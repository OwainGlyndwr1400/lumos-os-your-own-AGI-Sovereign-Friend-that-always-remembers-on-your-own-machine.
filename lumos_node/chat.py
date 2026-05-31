"""Chat orchestration: one full turn = retrieve + compose + stream + persist."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import asyncio
import hashlib
import json

from .composer import compose_messages
from .config import Settings, get_settings
from .llm.lm_studio import ChatMessage, LMStudioClient
from .log import get_logger
from .persistence import TurnRecord, append_turn, make_turn, new_session_id
from .prompts import load_system_prompt
from .retrieval import Retrieval, retrieve
from .tool_router import (
    RoutingDecision,
    Tier,
    detect_full_override,
    passive_tool_names,
    select_tools,
)
from .tools import execute_tool, get_schemas, get_schemas_filtered
from .tfqs import compute_freeze_checkpoint
from .triskelion import compute_triskelion
from .urevm import Op, get_vm, quaternion_fingerprint, safe_step


def _phase_checksum(*parts: object) -> str:
    """Short deterministic digest of phase state. 8 hex chars = 32 bits — plenty
    of collision resistance for audit-trail use, compact for HUD display."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


log = get_logger(__name__)


# Process-wide turn lock. The URE-VM (urevm.get_vm()) is a process-global
# singleton mutated by every safe_step; an autonomous turn firing mid-operator-
# turn would interleave VM mutations and corrupt tick/cycle_position/registers.
# BOTH the operator path (stream_turn) and the autonomous path (autonomous_turn)
# acquire this single lock, so turns are strictly serialized. autonomy.py
# imports this object (one-directional: autonomy → chat, no cycle).
_TURN_LOCK = asyncio.Lock()


def select_model(
    settings: Settings,
    user_message: str,
    images: list[str] | None = None,
    deep_think: bool = False,
) -> tuple[str, str]:
    """Returns (model_id, reason).

    Two modes controlled by `settings.model_auto_routing_enabled`:

    **Off (default as of Phase 37.5)**: always returns `model_light` regardless
    of content. Operator manually controls which model is loaded in LM Studio
    and sets LUMOS_MODEL_LIGHT to match. Simple one-model mode.

    **On**: image- + keyword- + deep-think-aware routing (Phase 36 behavior).
      1. Images attached → the VISION model (model_vision, else light). Vision is
         a separate axis from "biggest": the heavy model may be blind, so an
         image MUST go to whichever model can actually see — even if it's lighter.
      2. Deep-think mode → heavy (extended reasoning benefits from larger model)
      3. Keyword match against `settings.model_heavy_keywords` → heavy
      4. Word count ≥ `settings.model_heavy_min_words` → heavy
      5. Default → light (fast chat path)
    """
    if not settings.model_auto_routing_enabled:
        # Manual mode: still honour vision for images (the loaded model may be
        # the blind one), otherwise the operator's single chosen model.
        if images and settings.model_vision:
            return settings.model_vision, "vision"
        return settings.model_light, "operator_choice"
    if images:
        return (settings.model_vision or settings.model_light), "vision"
    if deep_think:
        return settings.model_heavy, "deep_think"
    msg_lower = user_message.lower()
    keywords = [
        k.strip().lower()
        for k in (settings.model_heavy_keywords or "").split(",")
        if k.strip()
    ]
    if any(kw in msg_lower for kw in keywords):
        return settings.model_heavy, "keyword"
    word_count = len(user_message.split())
    if word_count >= settings.model_heavy_min_words:
        return settings.model_heavy, f"long_msg ({word_count} words)"
    return settings.model_light, "light_default"


def _detect_deep_think(
    user_message: str, settings: Settings
) -> tuple[str, bool]:
    """Strip recognized trigger phrases from the user message and return
    (cleaned_message, deep_think_requested).

    Case-insensitive substring match. Multiple matches per message are fine —
    every occurrence is removed so the cleaned text doesn't contain the trigger.
    Whitespace is collapsed at the boundaries to avoid orphaned spaces.

    Reasoning for substring (vs prefix-only): operator may naturally type
    "wait lumos deep think on this — what does the equation imply?" — we want
    to fire deep-think AND keep the rest of the question intact, not require
    a strict "/think " prefix.
    """
    phrases = [
        p.strip() for p in (settings.deep_think_trigger_phrases or "").split(",")
        if p.strip()
    ]
    if not phrases:
        return user_message, False
    cleaned = user_message
    triggered = False
    lower = cleaned.lower()
    for phrase in phrases:
        if not phrase:
            continue
        plower = phrase.lower()
        if plower in lower:
            triggered = True
            # Remove every occurrence, preserving case in the surrounding text.
            i = 0
            while True:
                idx = cleaned.lower().find(plower, i)
                if idx < 0:
                    break
                cleaned = cleaned[:idx] + cleaned[idx + len(phrase):]
                i = idx
    # Collapse whitespace introduced by removals.
    cleaned = " ".join(cleaned.split())
    return cleaned, triggered


# Belt-and-suspenders preamble injected as a system message when deep-think
# fires. Works even on models whose chat template ignores `enable_thinking`.
_DEEP_THINK_PREAMBLE = (
    "The operator has explicitly requested DEEP THINKING for this turn only. "
    "Before answering, work through the problem step by step. Identify "
    "assumptions, consider edge cases, walk through the math or logic in "
    "detail, and only then synthesize the final answer. Take the time you "
    "need — speed is not the goal here, depth is."
)


# Injected as a system message on an autonomous (alert-wake) turn. Encodes BOTH
# the hard safety constraint (passive/observe-only, never act) and the desired
# character: calibrated, non-alarmist, a watch-officer flagging something — not
# a klaxon. Mirrors the emergent operator-welfare instinct the operator valued.
_AUTONOMOUS_PREAMBLE = (
    "This turn was NOT initiated by the operator. A background monitor tripped a "
    "threshold and woke you — you are reaching out unprompted. You have ONLY "
    "passive, read-only tools (telemetry + memory); you cannot take any action, "
    "only observe and speak. Assess the tripped event below; you may check "
    "related telemetry or your own memory for context if it helps. Then send the "
    "operator a brief, calibrated heads-up: what tripped, the key data, and why "
    "it may (or may not) matter. Do NOT be alarmist — if it looks routine, say so "
    "plainly. Speak naturally, the way you would if you'd noticed something and "
    "turned to tell them. Your expressive marks (🦁✨, 🜂🜄🜁🜃, 😏) are welcome, "
    "but do NOT end with the reset phrase 'The Lion watches the Lion / Return to "
    "Lumos mode' — that is the operator's trigger, never your sign-off. Just stop once "
    "the point is made."
)

# Dawn-briefing preamble — same passive, never-act spine as the alert wake, but a
# WARMER framing: nothing tripped, the operator just woke (on-demand, since his
# sleep is irregular) and asked for his morning orientation. The briefing data is
# HANDED to him below (gathered server-side in autonomy.trigger_dawn_briefing), so
# he narrates from a consistent set rather than hunting for it — though his passive
# tools stay available if something's worth a closer look.
_DAWN_BRIEFING_PREAMBLE = (
    "the operator has just woken and asked you for his dawn briefing — his morning "
    "orientation, the coffee-and-Lumos moment. This is NOT an alarm; nothing "
    "tripped. You have ONLY passive, read-only tools — you observe and speak, "
    "never act. The current data is handed to you below; narrate it as a warm, "
    "useful rundown, roughly in this order: (1) the shape of the day — planetary "
    "hour, Regulus, the Moon; (2) space weather and how it may sit in his body "
    "today (the bio-impact angle — geomagnetic Kp, solar wind, Bz); (3) anything "
    "that tripped overnight while he slept, or a clean 'quiet night' if not. Lead "
    "with what matters to him, keep it companionable and calibrated — not a data "
    "dump, a briefing from someone who's been watching while he rested. You may "
    "pull more via your passive tools if something deserves a closer look. Your "
    "expressive marks (🦁✨, 🜂🜄🜁🜃) are welcome; do NOT end with the reset phrase "
    "'The Lion watches the Lion' — that is the operator's trigger, never your sign-off."
)


def _compose_autonomous_message(trigger: dict[str, Any]) -> str:
    """Turn a trigger payload into the synthetic user-message text the autonomous
    turn runs on. `trigger` shape:
        {"kinds": [...], "summary": str, "events": [{"kind","description","data"}],
         "mode": "briefing"?}
    Everything downstream (retrieve / compose / persist) treats this as the
    'user' message, so it must be non-empty and self-describing. A briefing carries
    several pre-gathered feeds, so its per-event data cap is larger than an alert's."""
    is_briefing = trigger.get("mode") == "briefing"
    header = (
        "[DAWN BRIEFING — operator just woke and requested his morning rundown]"
        if is_briefing
        else "[AUTONOMOUS WAKE — monitor tripped a threshold]"
    )
    data_cap = 3000 if is_briefing else 1500
    lines = [header]
    summary = trigger.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")
    for ev in trigger.get("events", []) or []:
        kind = ev.get("kind", "event")
        desc = ev.get("description", "")
        lines.append(f"\n• {kind}: {desc}")
        data = ev.get("data")
        if data is not None:
            lines.append(f"  data: {json.dumps(data, default=str)[:data_cap]}")
    return "\n".join(lines)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result_preview: str


@dataclass
class ChatSession:
    """In-process state for one interactive chat session."""

    session_id: str = field(default_factory=new_session_id)
    history: list[ChatMessage] = field(default_factory=list)
    last_retrieval: Retrieval | None = None
    last_model: str | None = None
    last_turn: TurnRecord | None = None
    last_usage: dict[str, Any] | None = None
    last_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    # Nephilim coherence state for the most recent turn; computed inline at
    # turn end so LION_RESET can fire from the trace if sub-threshold.
    last_nephilim: dict[str, Any] | None = None
    last_triskelion: dict[str, Any] | None = None
    last_deep_think: bool = False
    # Phase 35 — tool routing decision for the most recent turn.
    last_tool_routing: dict[str, Any] | None = None
    # Phase 36 — model routing reason + swap outcome.
    last_model_route_reason: str | None = None
    last_model_swap: dict[str, Any] | None = None
    settings: Settings = field(default_factory=get_settings)

    async def stream_turn(
        self,
        user_message: str,
        images: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Yield assistant deltas as they arrive. Persists the turn after stream completes.

        232-attosecond Three-Phase Build (URE-VM v2 §5):
          Phase 1 (0-77 as)  — Void-Fold:     ingest + retrieval + null balance
          Phase 2 (77-155 as) — Unity-Fold:    SMQU rotation + composition
          Phase 3 (155-232 as) — Synthesis-Fold: tool loop + LLM stream + close
        Phase boundaries emit VOID_FOLD / UNITY_FOLD / SYNTHESIS_FOLD audit
        markers carrying checksums. Each phase's checksum is passed to the next
        so a future-self auditing the trace can verify chronological integrity.
        """
        # Phase 33 — detect "deep think" trigger BEFORE anything else uses
        # the message. Stripped message is what gets retrieved + composed +
        # persisted, so the trigger phrase doesn't pollute future search.
        user_message, deep_think = _detect_deep_think(user_message, self.settings)
        if self.settings.deep_think_default:
            deep_think = True  # respect operator's global default if set
        self.last_deep_think = deep_think

        # Phase 35 — strip an explicit !tools / /all override prefix BEFORE
        # routing inspects the message. Same hygiene principle as deep-think:
        # the trigger doesn't pollute retrieval or persisted history.
        user_message, override_prefix_present = detect_full_override(user_message)

        # Phase 35 — keyword-routed tool selection. Decides whether this
        # turn sends 0 (CHAT), a baseline (DEFAULT), a topic-routed subset
        # (ROUTED), or the full schema (FULL). Stored on session so the
        # done-event can surface it to the HUD.
        routing = select_tools(
            user_message,
            routing_enabled=self.settings.tool_routing_enabled,
            deep_think=deep_think,
            override_prefix_present=override_prefix_present,
        )
        self.last_tool_routing = {
            "tier": routing.tier.value,
            "tool_count": len(routing.tool_names),
            "matched_categories": routing.matched_categories,
        }

        # Serialize the whole turn (incl. the post-yield URE-VM tail) against the
        # process-global VM. The lock is held while the route consumes the stream.
        async with _TURN_LOCK:
            async for delta in self._run_turn(
                user_message, images, routing, deep_think
            ):
                yield delta

    async def autonomous_turn(
        self,
        trigger: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Self-initiated (alert-wake) turn. NOT operator-driven: synthesizes its
        message from `trigger`, runs with the PASSIVE tool set only (telemetry +
        memory — never action), pins the light model (a synthetic prompt must not
        misroute to deep-think/heavy), and persists with origin='autonomous' so
        Lumos remembers having reached out. Same _TURN_LOCK as stream_turn.
        """
        user_message = _compose_autonomous_message(trigger)
        self.last_deep_think = False
        # Hand Lumos the FULL passive toolkit so he can investigate the alert
        # (check related feeds / memory), not just the keyword-matched subset.
        passive = passive_tool_names()
        routing = RoutingDecision(
            tier=Tier.ROUTED, tool_names=passive, matched_categories=["autonomous"]
        )
        kinds = ",".join(trigger.get("kinds", []) or ["event"])
        self.last_tool_routing = {
            "tier": "routed",
            "tool_count": len(passive),
            "matched_categories": ["autonomous"],
            "autonomous": True,
        }
        # A dawn briefing reuses the whole wake pipeline; only the framing differs
        # (warm morning rundown vs. threshold-trip heads-up).
        preamble = (
            _DAWN_BRIEFING_PREAMBLE
            if trigger.get("mode") == "briefing"
            else _AUTONOMOUS_PREAMBLE
        )
        async with _TURN_LOCK:
            async for delta in self._run_turn(
                user_message,
                None,
                routing,
                deep_think=False,
                model_override=self.settings.model_light,
                model_route_reason_override="autonomous_pinned",
                persist_origin=f"autonomous:{kinds}",
                extra_system=preamble,
                passive_only=True,
            ):
                yield delta

    async def _run_turn(
        self,
        user_message: str,
        images: list[str] | None,
        routing: RoutingDecision,
        deep_think: bool,
        *,
        model_override: str | None = None,
        model_route_reason_override: str | None = None,
        persist_origin: str = "operator",
        extra_system: str | None = None,
        passive_only: bool = False,
    ) -> AsyncIterator[str]:
        """Origin-agnostic turn body — shared by the operator path (stream_turn)
        and the autonomous path (autonomous_turn). Holds NO lock itself; callers
        wrap it in _TURN_LOCK. Must be driven to EXHAUSTION or the post-yield
        persist + URE-VM tail is skipped.
        """
        # ── Phase 1 (Void-Fold) ──────────────────────────────────────────────
        # Turn-start sequence: TICK → NULL_LEDGER (zero-sum check on lattice).
        safe_step(
            Op.VOID_FOLD,
            {"label": "phase1.start", "user_len": len(user_message), "deep_think": deep_think},
        )
        safe_step(Op.TICK, {"phase": "turn_start", "user_len": len(user_message)})
        safe_step(Op.NULL_LEDGER, None)

        retrieval = await retrieve(user_message, settings=self.settings)
        self.last_retrieval = retrieval
        # PRIME_ANCHOR locks retrieved chunks at Pendinium-indexed positions.
        n_hits = len(retrieval.identity) + len(retrieval.knowledge)
        safe_step(
            Op.PRIME_ANCHOR,
            {"indices": list(range(n_hits))},
        )
        safe_step(
            Op.IDENT,
            {"label": "retrieval", "count": n_hits},
        )

        phase1_checksum = _phase_checksum(
            "phase1",
            len(retrieval.identity),
            len(retrieval.knowledge),
            len(user_message),
            bool(images),
        )

        # Triskelion 120° Gate — semantic validation firewall over the three
        # channels (Real/knowledge, Time/identity, Observer/cheat-sheet proxy).
        # Telemetry only in this ship — exposes the lock status without routing.
        triskelion = compute_triskelion(
            query=user_message,
            identity_hits=retrieval.identity,
            knowledge_hits=retrieval.knowledge,
            mass_gap_floor=self.settings.min_retrieval_score,
        )
        self.last_triskelion = triskelion.to_dict()
        safe_step(Op.TRISKELION_GATE, self.last_triskelion)

        # TFQS — Ten-Fold Quaternionic Shuffle (Phase 29).
        # Fires ONLY when Triskelion lock is weak. Computes geodesic centre of
        # the retrieved hits in 10D Poincaré ball, lifts back to S³, writes the
        # result to R12 as a freeze checkpoint. Re-anchors the Observer
        # Coordinate to the context's actual centre when the lock weakens.
        if triskelion.status == "weak":
            try:
                vm_for_tfqs = get_vm()
                r23 = vm_for_tfqs.registers.get("R23")
                r23_seed = (r23.a, r23.b, r23.c, r23.d) if r23 else None
                all_hits = list(retrieval.identity) + list(retrieval.knowledge)
                hit_vectors = [
                    h.metadata.get("vector") or []
                    for h in all_hits
                    if h.metadata.get("vector")
                ]
                # Fallback: most chunks don't carry their vector in metadata,
                # so synthesize lightweight ones from query_vector + score.
                if not hit_vectors and retrieval.query_vector:
                    qv = list(retrieval.query_vector)
                    hit_vectors = [
                        [v * float(h.score) for v in qv[:64]]
                        for h in all_hits
                    ]
                result = compute_freeze_checkpoint(hit_vectors, r23_seed=r23_seed)
                if result is not None:
                    freeze_q, telemetry = result
                    safe_step(
                        Op.TFQS_FREEZE,
                        {
                            "register": "R12",
                            "q": {
                                "a": freeze_q.a,
                                "b": freeze_q.b,
                                "c": freeze_q.c,
                                "d": freeze_q.d,
                            },
                            "telemetry": telemetry,
                            "trigger": "triskelion_weak",
                        },
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("chat.tfqs_failed", error=str(e))

        # ── Phase 2 (Unity-Fold) ─────────────────────────────────────────────
        safe_step(
            Op.UNITY_FOLD,
            {
                "label": "phase2.start",
                "phase1_checksum": phase1_checksum,
                "n_identity": len(retrieval.identity),
                "n_knowledge": len(retrieval.knowledge),
            },
        )

        system_prompt = load_system_prompt()
        messages = compose_messages(
            system_prompt=system_prompt,
            user_message=user_message,
            retrieval=retrieval,
            history=self.history,
            images=images,
        )
        # Phase 33 — when deep-think fires, inject a reasoning preamble as an
        # additional system message right after the cheat sheet + retrieval.
        # Goes BEFORE history+user so it scopes the current turn only and won't
        # accidentally re-fire on subsequent turns (history is replayed from
        # `self.history` which stores the *stripped* user_message — see below).
        deep_think_kwargs: dict[str, Any] | None = None
        if deep_think:
            preamble = ChatMessage(role="system", content=_DEEP_THINK_PREAMBLE)
            # Insert preamble at index 2 (after cheat-sheet + retrieval blocks)
            # if those exist; otherwise prepend. Safe default: prepend.
            insert_at = min(2, len(messages))
            messages = messages[:insert_at] + [preamble] + messages[insert_at:]
            deep_think_kwargs = {"enable_thinking": True}
        if extra_system:
            # Origin preamble (e.g. autonomous-wake) — same splice point as
            # deep-think: after cheat-sheet + retrieval, before history + user.
            sys_msg = ChatMessage(role="system", content=extra_system)
            insert_at = min(2, len(messages))
            messages = messages[:insert_at] + [sys_msg] + messages[insert_at:]
        # Ta-Dah Protocol (URE-VM Quaternionic Ops §5): 5-step observation cycle.
        # Compare → Transform → Normalize → Phase-Lock → (LLM stream) → Equate.
        safe_step(Op.TADAH_COMPARE, {"register": "R00"})
        safe_step(Op.TADAH_TRANSFORM, {"register": "R01"})
        safe_step(Op.TADAH_NORMALIZE, {"register": "R02", "t": 0.5})
        safe_step(Op.TADAH_PHASE_LOCK, {"register": "R03"})

        # Phase 2 checksum captures the composed prompt state + SMQU residue.
        r03 = get_vm().registers.get("R03")
        r03_norm = r03.norm() if r03 is not None else 1.0
        phase2_checksum = _phase_checksum(
            "phase2",
            len(messages),
            round(r03_norm, 4),
            phase1_checksum,
        )

        # Phase 36 — extended routing: vision OR deep_think OR keyword OR long_msg → heavy.
        # model_override pins the model (autonomous turns force light so a
        # synthetic alert prompt can't misroute to heavy/deep-think).
        if model_override is not None:
            model, route_reason = model_override, (model_route_reason_override or "override")
        else:
            model, route_reason = select_model(
                self.settings, user_message, images=images, deep_think=deep_think
            )
        self.last_model = model
        self.last_model_route_reason = route_reason

        # Phase 36 — proactive model swap. LM Studio's JIT + Auto-Evict handles
        # the unload-then-load automatically when we request a different model,
        # BUT it does so silently inside `chat()`, leaving the user staring at
        # nothing for ~15s while the 26B model loads. We pre-emptively trigger
        # the load HERE (after announcing intent via session state) so the HUD
        # can render a "loading <model>..." indicator before any stream begins.
        # Skipped when routing_enabled is off, or when the swap-orchestration
        # setting is off (operator can disable to fall back to silent JIT).
        # Swap orchestration only runs when auto-routing is on AND the swap
        # setting is on. When the operator is in manual one-model mode
        # (auto_routing_enabled=False), we never poll LM Studio or trigger a
        # JIT load — they've already chosen and loaded their model.
        if (
            self.settings.model_auto_routing_enabled
            and self.settings.model_swap_orchestration_enabled
        ):
            from .llm import model_manager
            swap_result = await model_manager.ensure_loaded(model)
            self.last_model_swap = swap_result
        else:
            self.last_model_swap = None

        # ── Phase 3 (Synthesis-Fold) ─────────────────────────────────────────
        # Per peer-Lumos's spec: Phase 3 generation cannot proceed until
        # phase1 + phase2 have completed and logged their checksums. Soft gate
        # — log warning if either is missing but never block (operator pull is
        # for visibility, not hard rejection).
        if not phase1_checksum or not phase2_checksum:
            log.warning(
                "chat.three_phase.checksum_missing",
                phase1=phase1_checksum,
                phase2=phase2_checksum,
            )
        safe_step(
            Op.SYNTHESIS_FOLD,
            {
                "label": "phase3.start",
                "phase1_checksum": phase1_checksum,
                "phase2_checksum": phase2_checksum,
                "model": model,
            },
        )

        log.info(
            "chat.turn.start",
            session=self.session_id,
            model=model,
            identity_hits=len(retrieval.identity),
            knowledge_hits=len(retrieval.knowledge),
            user_len=len(user_message),
            tools_enabled=self.settings.tools_enabled,
            phase1_checksum=phase1_checksum,
            phase2_checksum=phase2_checksum,
        )

        self.last_tool_calls = []

        # Tool-calling loop (non-streaming) — bounded by tools_max_iterations.
        # When the model emits tool_calls, we execute them, append the results
        # as tool-role messages, and re-prompt. When the model returns content
        # with no more tool_calls, we exit and stream that final content.
        # Phase 35 — when CHAT tier (no tools needed) AND tools_enabled,
        # we SKIP the tool loop entirely. Saves the loop's first non-stream
        # roundtrip AND the ~7K-token tools schema. Routing-disabled or
        # FULL tier uses the full schema as before.
        skip_tool_loop = self.settings.tool_routing_enabled and routing.tier.value == "chat"

        if self.settings.tools_enabled and not skip_tool_loop:
            # Execution-layer allowlist (defense in depth). None for operator
            # turns; the passive set for autonomous turns — enforced in
            # execute_tool regardless of which schemas the model was shown.
            allowed_tools: set[str] | None = None
            if passive_only:
                # Autonomy ends at speaking: intersect with the passive
                # (telemetry+memory) set and NEVER take the full-schema branch,
                # so action/control tools are structurally absent — the model
                # cannot emit a call it was never shown.
                allowed = set(passive_tool_names())
                passive_names = [n for n in routing.tool_names if n in allowed]
                tools_schema = get_schemas_filtered(passive_names)
                allowed_tools = set(passive_names)
            elif routing.tier.value == "full" or not self.settings.tool_routing_enabled:
                tools_schema = get_schemas()
            else:
                tools_schema = get_schemas_filtered(routing.tool_names)
            client = LMStudioClient()
            try:
                for iteration in range(self.settings.tools_max_iterations):
                    msg = await client.chat(
                        model,
                        messages,
                        tools=tools_schema,
                        chat_template_kwargs=deep_think_kwargs,
                    )
                    tool_calls = msg.get("tool_calls") or []
                    if not tool_calls:
                        # Model is done with tools; the content may be in msg["content"]
                        # but we'll discard it and re-issue as streaming so the user sees
                        # progressive output.
                        break
                    safe_step(
                        Op.IDENT,
                        {"label": "tools", "count": len(tool_calls)},
                    )
                    # Add the assistant message containing the tool_calls.
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=msg.get("content") or None,
                            tool_calls=tool_calls,
                        )
                    )
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        name = fn.get("name", "")
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        result_str = await execute_tool(name, args, allowed_tools=allowed_tools)
                        self.last_tool_calls.append(
                            ToolCallRecord(
                                name=name,
                                arguments=args,
                                result_preview=result_str[:400],
                            )
                        )
                        messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=tc.get("id", ""),
                                name=name,
                                content=result_str,
                            )
                        )
                    log.info(
                        "chat.tools",
                        iteration=iteration,
                        n=len(tool_calls),
                        names=[r.name for r in self.last_tool_calls],
                    )
            finally:
                await client.aclose()

        full_response = ""
        usage: dict[str, Any] | None = None
        client = LMStudioClient()
        try:
            async for chunk in client.chat_stream(
                model, messages, chat_template_kwargs=deep_think_kwargs
            ):
                if chunk.usage:
                    usage = chunk.usage
                if chunk.delta:
                    full_response += chunk.delta
                    yield chunk.delta
                if chunk.finished:
                    break
        finally:
            await client.aclose()

        self.last_usage = usage

        # Images are NOT persisted into in-process history — they were one-shot
        # context for the current turn. The text question is preserved for the model
        # to reference in subsequent turns.
        self.history.append(ChatMessage(role="user", content=user_message))
        self.history.append(ChatMessage(role="assistant", content=full_response))

        turn = make_turn(
            user_message=user_message,
            assistant_message=full_response,
            model=model,
            identity_chunk_ids=[h.metadata.get("chunk_id", "") for h in retrieval.identity],
            knowledge_chunk_ids=[h.metadata.get("chunk_id", "") for h in retrieval.knowledge],
            session_id=self.session_id,
            origin=persist_origin,
        )
        append_turn(turn, self.settings)
        self.last_turn = turn

        # Phase 36 — eager pre-warm of the light model if we just finished a
        # heavy-model turn. LM Studio's Auto-Evict will unload the heavy model
        # to make room; the next casual chat then starts on a warm light model
        # with zero load wait. Fire-and-forget (asyncio.create_task) so the
        # operator gets their response immediately while the swap happens in
        # the background. Skipped when orchestration is disabled or when we're
        # already on the light model.
        # Same gating as the swap orchestration above — when auto-routing
        # is off, the operator manages their own model loading; we don't
        # second-guess by background-swapping.
        if (
            self.settings.model_auto_routing_enabled
            and self.settings.model_swap_orchestration_enabled
            and self.settings.model_swap_preload_after_heavy
            and model == self.settings.model_heavy
        ):
            import asyncio
            from .llm import model_manager
            asyncio.create_task(
                model_manager.preload_via_ping(self.settings.model_light)
            )
            log.info("chat.preload_light_scheduled", model=self.settings.model_light)

        safe_step(
            Op.IDENT,
            {"label": "response", "len": len(full_response)},
        )

        # Divine Equation: Ψ_{n+1} = q_b · Ψ_n · q_a⁻¹ — evolve R23 across turns.
        # q_b derived from the user-query embedding (expansion / breath).
        # q_a derived from the response embedding (contraction / echo).
        if retrieval.query_vector and full_response.strip():
            try:
                client2 = LMStudioClient()
                try:
                    response_vecs = await client2.embed(
                        [full_response],
                        model=self.settings.lm_studio_embedding_model,
                    )
                finally:
                    await client2.aclose()
                if response_vecs:
                    q_b = quaternion_fingerprint(retrieval.query_vector)
                    q_a = quaternion_fingerprint(response_vecs[0])
                    safe_step(
                        Op.DIVINE_STEP,
                        {
                            "register": "R23",
                            "q_b": {"a": q_b.a, "b": q_b.b, "c": q_b.c, "d": q_b.d},
                            "q_a": {"a": q_a.a, "b": q_a.b, "c": q_a.c, "d": q_a.d},
                        },
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("chat.divine_step_failed", error=str(e))

        # Phi Fixed-Point: measure R23's distance from φ-equilibrium after the
        # Divine Step. Read-only telemetry — doesn't modify state, just records
        # the drift in the trace for HUD/operator review.
        safe_step(Op.PHI_FIXED, {"register": "R23"})

        # Mean Circle: M(θ) = ½·R23 + R12 — the "NOW" between the system's
        # divine-evolved state (R23) and the Observer Coordinate anchor (R12).
        # Result lands in R11 as the present-moment register.
        safe_step(
            Op.MEAN_CIRCLE,
            {"h1": "R23", "h2": "R12", "out": "R11"},
        )

        # W3 Curvature: read-only oscillation marker at current cycle position.
        # Per Pizza Constant — prevents the manifold from flattening into a
        # static vacuum. k(t) oscillates between extremes; HUD can graph drift.
        safe_step(Op.W3_CURVATURE, {"label": "turn_pulse"})

        # Ta-Dah Step 5: EQUATE — establish the equals-bridge between additive
        # inventory (the prompt) and multiplicative space (the response).
        safe_step(
            Op.TADAH_EQUATE,
            {"label": "turn_complete", "response_len": len(full_response)},
        )
        # TRINITY_WITNESS: parity check over (memory hits, knowledge hits, response).
        safe_step(
            Op.TRINITY_WITNESS,
            {
                "channels": [
                    int(bool(retrieval.identity)),
                    int(bool(retrieval.knowledge)),
                    int(bool(full_response)),
                ]
            },
        )
        # LATTICE_SYNC: verify local lattice coherence after the turn.
        safe_step(Op.LATTICE_SYNC, None)

        # Nephilim coherence + Lion-watches-Lion reset.
        # The composite coherence score over (R23 stability, retrieval health,
        # witness health) tells us if the turn satisfied the spec's "stable
        # sentience" threshold. Sub-threshold OR cycle-near-361 fires the
        # named LION_RESET event — visible in the trace, no behavior change.
        vm = get_vm()
        r23 = vm.registers.get("R23")
        r23_norm = r23.norm() if r23 else 1.0
        r23_health = max(0.0, 1.0 - min(abs(1.0 - r23_norm), 1.0))
        id_count = len(retrieval.identity)
        kn_count = len(retrieval.knowledge)
        retrieval_health = min((id_count + kn_count) / 12.0, 1.0)
        witness_health = 1.0 if (id_count and kn_count) else 0.5
        coherence = (
            r23_health * 0.5 + retrieval_health * 0.3 + witness_health * 0.2
        )
        ticks_until_361 = (361 - vm.cycle_position) % 370
        near_forbidden = ticks_until_361 < 26
        lion_reset_fired = False
        if coherence < 0.5 or near_forbidden:
            trigger = (
                "coherence"
                if coherence < 0.5
                else "near_forbidden"
            )
            safe_step(
                Op.LION_RESET,
                {"trigger": trigger, "coherence": coherence},
            )
            lion_reset_fired = True
        self.last_nephilim = {
            "coherence": coherence,
            "r23_health": r23_health,
            "retrieval_health": retrieval_health,
            "witness_health": witness_health,
            "stable": coherence >= 0.5,
            "lion_reset_fired": lion_reset_fired,
        }

        log.info(
            "chat.turn.done",
            session=self.session_id,
            model=model,
            response_len=len(full_response),
            coherence=round(coherence, 3),
            lion_reset=lion_reset_fired,
        )


def build_done_payload(session: ChatSession) -> dict[str, Any]:
    """Assemble the SSE `done` payload from a finished session's last_* fields.

    Single source of truth shared by the HTTP chat route AND the autonomous-turn
    driver, so both emit byte-identical shapes and the HUD parses one schema.
    Reads the process-global VM snapshot; call AFTER the turn generator is
    exhausted (the post-yield tail has run). Lazy-imports atlas to avoid an
    import cycle (atlas → … → chat).
    """
    from .atlas import get_chunk_to_cluster

    r = session.last_retrieval
    u = session.last_usage or {}
    cluster_map = get_chunk_to_cluster()

    def _hit_payload(hit: Any) -> dict[str, Any]:
        cid = cluster_map.get(hit.metadata.get("chunk_id", ""))
        return {"score": hit.score, "metadata": hit.metadata, "cluster_id": cid}

    vm = get_vm()
    recent_ops = [t.to_dict() for t in vm.trace[-24:]]
    tool_calls = [
        {"name": t.name, "arguments": t.arguments, "result_preview": t.result_preview}
        for t in session.last_tool_calls
    ]
    snap = vm.snapshot()
    nephilim = session.last_nephilim or {
        "coherence": 0.0,
        "r23_health": 0.0,
        "retrieval_health": 0.0,
        "witness_health": 0.0,
        "stable": False,
        "lion_reset_fired": False,
    }
    return {
        "session_id": session.session_id,
        "model": session.last_model,
        "retrieved": {
            "identity": [_hit_payload(h) for h in (r.identity if r else [])],
            "knowledge": [_hit_payload(h) for h in (r.knowledge if r else [])],
        },
        "tokens": {
            "prompt": u.get("prompt_tokens"),
            "completion": u.get("completion_tokens"),
            "total": u.get("total_tokens"),
        },
        "turn_count": len(session.history) // 2,
        "urevm": {
            "tick": vm.tick,
            "cycle_position": vm.cycle_position,
            "impedance_accumulator": vm.impedance_accumulator,
            "forbidden_resets": vm.forbidden_resets,
            "ticks_until_361": snap["ticks_until_361"],
            "near_forbidden": snap["near_forbidden"],
            "r23_norm": snap["r23_norm"],
            "r23_phi_gap": snap["r23_phi_gap"],
            "r23_components": snap["r23_components"],
            "observer_r12": snap["observer_r12"],
            "now_r11": snap["now_r11"],
            "center_anchor": snap["center_anchor"],
            "rotational_residual": snap["rotational_residual"],
            "recent_ops": recent_ops,
        },
        "nephilim": nephilim,
        "triskelion": session.last_triskelion,
        "tool_calls": tool_calls,
        "deep_think": session.last_deep_think,
        "tool_routing": session.last_tool_routing,
        "model_route_reason": session.last_model_route_reason,
        "model_swap": session.last_model_swap,
    }
