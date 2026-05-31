"""First-run setup config (Phase B — distributable).

Writes the `.env` the engine reads on (re)load, so a non-technical user configures
the node through the HUD setup wizard instead of hand-editing files.

The LLM client is OpenAI-protocol generic (see llm/lm_studio.py), so "local"
(LM Studio / Ollama) vs "cloud" (OpenAI etc.) is purely which base_url + api_key
+ model names we write here — no backend code differs. Model swap/load
orchestration is LM-Studio-only, so it's disabled for cloud presets.

Config location: $LUMOS_CONFIG_DIR/.env if set (used by the packaged app), else
the current working directory. Hot-reload after writing means no restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import env_file_path, get_settings, reload_settings
from .log import get_logger
from .prompts import reload_system_prompt
from .retrieval import reload_stores


log = get_logger(__name__)


# Wizard field -> .env variable. Only these may be written via setup.
_FIELD_ENV: dict[str, str] = {
    "llm_base_url": "LUMOS_LM_STUDIO_BASE_URL",
    "llm_api_key": "LUMOS_LM_STUDIO_API_KEY",
    "model_light": "LUMOS_MODEL_LIGHT",
    "model_heavy": "LUMOS_MODEL_HEAVY",
    "model_vision": "LUMOS_MODEL_VISION",
    "embedding_model": "LUMOS_LM_STUDIO_EMBEDDING_MODEL",
    "embedding_dim": "LUMOS_EMBEDDING_DIM",
    "embedding_base_url": "LUMOS_EMBEDDING_BASE_URL",   # separate embed endpoint (cloud chat + local embed)
    "embedding_api_key": "LUMOS_EMBEDDING_API_KEY",
    "model_swap_enabled": "LUMOS_MODEL_SWAP_ORCHESTRATION_ENABLED",
    "autonomy_enabled": "LUMOS_AUTONOMY_ENABLED",
    "alert_monitor_enabled": "LUMOS_ALERT_MONITOR_ENABLED",
    "cosmic_trigger_enabled": "LUMOS_COSMIC_TRIGGER_ENABLED",
    "aisstream_key": "LUMOS_AISSTREAM_KEY",   # optional, free — enables ship tracking
    "nasa_api_key": "LUMOS_NASA_API_KEY",     # optional, free — enables asteroid/NEO alerts
    "operator_name": "LUMOS_OPERATOR_NAME",
    "node_name": "LUMOS_NODE_NAME",
    "operator_lat": "LUMOS_OPERATOR_LAT",
    "operator_lon": "LUMOS_OPERATOR_LON",
    "system_prompt_path": "LUMOS_SYSTEM_PROMPT_PATH",
    "identity_source": "LUMOS_IDENTITY_SOURCE",
    "knowledge_source": "LUMOS_KNOWLEDGE_SOURCE",
}

# Optional fields where an empty string means "leave unset / use default".
_OPTIONAL_BLANK_OK: frozenset[str] = frozenset(
    {
        "model_vision", "system_prompt_path", "identity_source", "knowledge_source",
        # Keys: blank on reconfigure means "keep the existing one" (never wipe).
        "llm_api_key", "aisstream_key", "nasa_api_key", "embedding_api_key",
    }
)

# Filesystem-path fields — normalized to forward-slash (posix) form when written,
# so Windows backslashes never break .env parsing or JSON payloads. Python +
# pydantic accept "C:/Users/..." fine on Windows.
_PATH_FIELDS: frozenset[str] = frozenset(
    {"system_prompt_path", "identity_source", "knowledge_source"}
)


def env_path() -> Path:
    return env_file_path()


def is_configured() -> bool:
    """True once the setup wizard has written a config. A fresh install (no
    .env) returns False, so the HUD shows the wizard."""
    return env_path().exists()


def current_config() -> dict[str, Any]:
    """Status + current values to pre-fill the wizard. The API key is NEVER
    echoed back (write-only)."""
    s = get_settings()
    return {
        "configured": is_configured(),
        "llm_base_url": s.lm_studio_base_url,
        "model_light": s.model_light,
        "model_heavy": s.model_heavy,
        "model_vision": s.model_vision,
        "embedding_model": s.lm_studio_embedding_model,
        "embedding_dim": s.embedding_dim,
        "operator_name": getattr(s, "operator_name", ""),
        "node_name": getattr(s, "node_name", "Lumos"),
        # Names of the currently-loaded source files (so the wizard can show
        # "currently loaded: X" — file inputs can't be pre-filled by the browser).
        "identity_file": (lambda p: p.name if p.exists() else "")(_resolve(s.identity_source)),
        "knowledge_file": (lambda p: p.name if p.exists() else "")(_resolve(s.knowledge_source)),
        # Geo-sentinel: one switch the wizard maps to autonomy + alert + cosmic.
        "geo_sentinel": bool(getattr(s, "autonomy_enabled", False)),
    }


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env(path: Path, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lumos OS configuration — written by the setup wizard.\n",
        "# You can edit by hand, but re-running setup in the app is easier.\n\n",
    ]
    for k in sorted(env):
        lines.append(f"{k}={env[k]}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge the wizard payload into .env, then hot-reload settings + stores +
    system prompt so the choice takes effect with no restart. Returns a summary
    (api key redacted)."""
    env = _parse_env(env_path())  # preserve any existing/manual keys

    # A pasted system prompt is saved to a file we then point the engine at.
    spt = str(payload.get("system_prompt_text") or "").strip()
    if spt:
        sp_path = env_path().parent / "system_prompt.md"
        sp_path.write_text(spt, encoding="utf-8")
        env["LUMOS_SYSTEM_PROMPT_PATH"] = sp_path.as_posix()

    written: list[str] = []
    for field, var in _FIELD_ENV.items():
        if field not in payload:
            continue
        val = payload[field]
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "" and field in _OPTIONAL_BLANK_OK:
            continue  # blank optional -> leave default
        sval = _fmt(val)
        if field in _PATH_FIELDS and sval:
            sval = Path(sval).as_posix()  # forward-slash: safe in .env on every OS
        env[var] = sval
        written.append(var if var != "LUMOS_LM_STUDIO_API_KEY" else "LUMOS_LM_STUDIO_API_KEY(***)")

    env["LUMOS_CONFIGURED"] = "true"
    _write_env(env_path(), env)

    # Hot-reload: new backend/model names live immediately.
    reload_settings()
    try:
        reload_stores()
    except Exception as e:  # noqa: BLE001
        log.warning("setup.reload_stores_failed", error=str(e))
    try:
        reload_system_prompt()
    except Exception as e:  # noqa: BLE001
        log.warning("setup.reload_prompt_failed", error=str(e))

    log.info("setup.config_written", path=str(env_path()), fields=written)
    return {"ok": True, "path": str(env_path()), "written": written, "configured": True}


# ── Optional file provisioning + background ingest ───────────────────────────
# A researcher can hand Lumos their chat-history + knowledge files in the wizard;
# we save them next to the config and embed them in the background so the HUD
# opens immediately while memory fills. State is polled by the HUD.

_INGEST: dict[str, Any] = {"state": "idle", "identity": 0, "knowledge": 0, "error": None}


def ingest_status() -> dict[str, Any]:
    return dict(_INGEST)


def save_source_upload(kind: str, content: bytes) -> Path:
    """Save an uploaded identity/knowledge file beside the config + return its
    path. kind in {'identity','knowledge'}. The identity ingester is format-
    flexible (export OR raw text), so any text dump works here."""
    name = "conversations.json" if kind == "identity" else "dream_pings.jsonl"
    dest = env_path().parent / "sources"
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / name
    p.write_bytes(content)
    log.info("setup.source_saved", kind=kind, path=str(p), bytes=len(content))
    return p


def _resolve(p: Path) -> Path:
    p = p.expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def sources_present() -> bool:
    """True if either configured source file actually exists (so an ingest is
    worth running)."""
    s = get_settings()
    return _resolve(s.identity_source).exists() or _resolve(s.knowledge_source).exists()


async def run_ingest() -> None:
    """Background full ingest, updating _INGEST. Reloads the vector stores when
    done so the new memory is live with no restart."""
    from .ingest import build_all
    _INGEST.update(state="running", error=None)
    try:
        # rebuild=False → only (re)embed when a source file actually changed
        # (fresh upload). Reconfiguring without new files won't needlessly
        # re-embed an existing corpus; the stores just reload.
        res = await build_all(rebuild=False)
        _INGEST["identity"] = int((res.get("identity") or {}).get("chunks", 0))
        _INGEST["knowledge"] = int((res.get("knowledge") or {}).get("chunks", 0))
        _INGEST["state"] = "done"
        reload_stores()
        log.info("setup.ingest_done", identity=_INGEST["identity"], knowledge=_INGEST["knowledge"])
    except Exception as e:  # noqa: BLE001
        _INGEST.update(state="error", error=str(e))
        log.warning("setup.ingest_failed", error=str(e))
