"""Alert monitor (Phase 3) — event-driven threshold wakes.

Generalizes the cosmic-trigger scaffold (telemetry/worker.py) to the full intel
layer: polls each source on a cadence, evaluates NUMERIC thresholds in PURE CODE
(no LLM), and on a fresh trip wakes Lumos via autonomy.trigger_autonomous_turn
with ONLY the tripped events as context. Autonomy ends at speaking.

Design (locked 2026-05-29):
  • Event-driven, not a timed dump: the poll + threshold check is tokenless code;
    the LLM is invoked only on a trip, and sees only what tripped.
  • Per-(source, identity) dedup: a given aircraft hex / ship MMSI / satellite /
    GPS zone re-alerts only after `alert_cooldown_minutes`; a daily cap bounds
    total wakes; a new distinct entity is a new alert.
  • Bundled wake: all FRESH trips in one poll cycle become ONE wake ("here's
    what's around"), not N separate pings.
  • Gated: runs when alert_monitor_enabled; only WAKES when autonomy_enabled.
    alert_monitor_enabled + autonomy OFF = DRY RUN (logs what WOULD trip without
    pinging) — handy for tuning thresholds first.
  • Ships ride the persistent AIS cache (maritime.ais_monitor_loop), started as a
    child task here, so the naval-type + anomalous filters are reliable.

Thresholds (the operator's locked values, from config):
  Kp/flare/quake/NEO (reuse cosmic) · mil-air ≤40 mi · ships naval|anomalous
  ≤50 mi · GPS zone ≤150 km · mil-recon sat ≥60° elevation.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from ..config import Settings, get_settings
from ..events import EventBus
from ..log import get_logger
from . import cosmic, gpsjam, maritime, military, satellites
from .worker import (
    _chat_idle_seconds,
    _data_dir,
    _evaluate_thresholds as _evaluate_cosmic,
    _today_iso,
)


log = get_logger(__name__)

_LOG_FILE = "alert_events.jsonl"
_STATE_FILE = "alert_state.json"
_LOG_CAP = 1000


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _state_path(s: Settings) -> Path:
    return _data_dir(s) / _STATE_FILE


def _log_path(s: Settings) -> Path:
    return _data_dir(s) / _LOG_FILE


def _read_state(s: Settings) -> dict[str, Any]:
    p = _state_path(s)
    base = {"identities": {}, "fires_today": 0, "day_iso": ""}
    if not p.exists():
        return base
    try:
        st = orjson.loads(p.read_bytes())
        for k, v in base.items():
            st.setdefault(k, v)
        return st
    except (orjson.JSONDecodeError, OSError):
        return base


def _write_state(s: Settings, st: dict[str, Any]) -> None:
    try:
        _state_path(s).write_bytes(orjson.dumps(st, option=orjson.OPT_INDENT_2))
    except OSError as e:
        log.warning("alert.state_write_failed", error=str(e))


def _append_log(s: Settings, entry: dict[str, Any]) -> None:
    p = _log_path(s)
    try:
        with p.open("ab") as f:
            f.write(orjson.dumps(entry))
            f.write(b"\n")
    except OSError as e:
        log.warning("alert.log_write_failed", error=str(e))
        return
    try:
        with p.open("rb") as f:
            lines = f.readlines()
        if len(lines) > _LOG_CAP * 2:
            with p.open("wb") as f:
                f.writelines(lines[-_LOG_CAP:])
    except OSError:
        pass


async def _evaluate_alerts(settings: Settings) -> list[dict[str, Any]]:
    """Gather current threshold trips across all sources. Each trip is
    {id, kind, description, data}. `id` is the dedup identity (per aircraft /
    vessel / satellite / GPS zone / cosmic kind)."""
    lat, lon = settings.operator_lat, settings.operator_lon
    trips: list[dict[str, Any]] = []

    # ── Cosmic (Kp / flare / quake / NEO) — reuse the cosmic evaluator verbatim.
    try:
        snap = await cosmic.snapshot_all()
        for ev in _evaluate_cosmic(snap, settings):
            trips.append(
                {"id": f"cosmic:{ev['kind']}", "kind": ev["kind"],
                 "description": ev["description"], "data": ev}
            )
    except Exception as e:  # noqa: BLE001
        log.info("alert.cosmic_failed", error=str(e))

    # ── Military aircraft within radius.
    try:
        mil = await military.fetch_military_aircraft(
            lat=lat, lon=lon, radius_km=settings.alert_military_air_radius_km
        )
        if mil.get("ok"):
            for ac in mil.get("aircraft", []):
                hexid = ac.get("hex") or ac.get("callsign") or "?"
                cs = ac.get("callsign") or hexid
                tc = ac.get("type_code") or "?"
                trips.append(
                    {"id": f"mil:{hexid}", "kind": "military_air",
                     "description": (
                         f"Military aircraft {cs} ({tc}) within "
                         f"{settings.alert_military_air_radius_km:.0f} km"
                     ),
                     "data": ac}
                )
    except Exception as e:  # noqa: BLE001
        log.info("alert.mil_failed", error=str(e))

    # ── GPS-jamming zones whose centroid is within the alert radius.
    try:
        gps = await gpsjam.fetch_gps_jamming(lat=lat, lon=lon)
        if gps.get("ok"):
            for z in gps.get("zones", []):
                d = _haversine_km(lat, lon, z["lat"], z["lon"])
                if d <= settings.alert_gps_jam_radius_km:
                    trips.append(
                        {"id": f"gps:{z['lat']:.1f}_{z['lon']:.1f}", "kind": "gps_jamming",
                         "description": (
                             f"GPS-jamming zone {z['severity_pct']}% severity, "
                             f"{z['degraded_count']} aircraft, ~{d:.0f} km away"
                         ),
                         "data": {**z, "distance_km": round(d, 1)}}
                    )
    except Exception as e:  # noqa: BLE001
        log.info("alert.gps_failed", error=str(e))

    # ── Military-recon satellites at high elevation (near-overhead pass).
    try:
        sats = await satellites.fetch_satellites_overhead(
            lat=lat, lon=lon, min_elevation=settings.alert_sat_min_elevation_deg, limit=50
        )
        if sats.get("ok"):
            for st in sats.get("satellites", []):
                if st.get("mission") == "military_recon":
                    trips.append(
                        {"id": f"sat:{st['name']}", "kind": "recon_satellite",
                         "description": (
                             f"Recon satellite {st['name']} overhead at "
                             f"{st['elevation_deg']:.0f}° elevation"
                         ),
                         "data": st}
                    )
    except Exception as e:  # noqa: BLE001
        log.info("alert.sat_failed", error=str(e))

    # ── Ships — persistent AIS cache; naval-type OR anomalous-behaviour within radius.
    try:
        for v in maritime.snapshot_vessels(lat, lon, settings.alert_ship_radius_km):
            if v.get("is_naval") or v.get("is_anomalous"):
                if v.get("is_naval"):
                    tag = "naval/military"
                else:
                    ns = v.get("nav_status") or f"nav code {v.get('nav_status_code')}"
                    tag = f"anomalous ({ns})"
                nm = v.get("name") or f"MMSI {v['mmsi']}"
                trips.append(
                    {"id": f"ship:{v['mmsi']}", "kind": "vessel",
                     "description": f"Vessel {nm} — {tag} — ~{v['distance_km']:.0f} km",
                     "data": v}
                )
    except Exception as e:  # noqa: BLE001
        log.info("alert.ship_failed", error=str(e))

    # ── Operator-defined custom watches (Phase 39) — same dedup/cooldown/wake
    # path. Trip ids are prefixed 'watch:<id>:' so a watch fires independently of
    # the home thresholds (one feed failing for one watch never sinks the rest).
    try:
        from . import watches as _watches
        trips.extend(await _watches.evaluate_watches(settings))
    except Exception as e:  # noqa: BLE001
        log.info("alert.watches_failed", error=str(e))

    return trips


def _global_eligible(settings: Settings, state: dict[str, Any]) -> tuple[bool, str]:
    """Daily-cap + chat-active gates (per-identity cooldown handled separately)."""
    today = _today_iso()
    if (
        state.get("day_iso") == today
        and int(state.get("fires_today") or 0) >= settings.alert_daily_cap
    ):
        return False, "daily_cap"
    skip = settings.alert_skip_if_chat_active_minutes * 60
    if skip > 0:
        idle = _chat_idle_seconds(settings)
        if idle < skip:
            # Don't self-suppress: an autonomous wake's own append_turn bumps
            # identity_events.jsonl mtime, which would otherwise read as "chat
            # active" for the next 1-2 polls. If the recent write IS our own
            # last wake (within a few seconds), it's not the operator chatting.
            last_wake = float(state.get("last_wake_unix") or 0.0)
            activity_mtime = time.time() - idle
            if abs(activity_mtime - last_wake) > 10.0:
                return False, "chat_active"
    return True, ""


async def _poll_once(settings: Settings, bus: EventBus) -> None:
    trips = await _evaluate_alerts(settings)
    # Heartbeat: the monitor is otherwise silent on a quiet cycle, so log one
    # line per poll showing it's alive + how many vessels the persistent AIS
    # feed has cached (climbs as ships broadcast). Lets the operator SEE it
    # watching. (Raise alert_poll_interval_seconds if the cadence feels noisy.)
    log.info("alert.poll", trips=len(trips), vessels_cached=len(maritime._vessel_cache))
    if not trips:
        return

    state = _read_state(settings)
    now = time.time()
    cooldown = settings.alert_cooldown_minutes * 60
    identities: dict[str, Any] = state.get("identities", {})
    fresh = [t for t in trips if (now - float(identities.get(t["id"], 0.0))) >= cooldown]

    audit: dict[str, Any] = {
        "ts_unix": now,
        "ts_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trips": [{"id": t["id"], "kind": t["kind"]} for t in trips],
        "fresh": [t["id"] for t in fresh],
        "woke": False,
    }

    if not fresh:
        audit["skipped_reason"] = "all_on_cooldown"
        _append_log(settings, audit)
        return

    eligible, reason = _global_eligible(settings, state)
    if not eligible:
        audit["skipped_reason"] = reason
        _append_log(settings, audit)
        log.info("alert.skipped", reason=reason, fresh=[t["id"] for t in fresh])
        return

    # Force dry-run when autonomy is off OR the operator location is unset
    # (0,0 = null-island sentinel; waking against it would describe a wrong-
    # region world). Dry-run logs what WOULD wake and does NOT consume the dedup
    # (so trips still fire once the gate clears).
    coords_unset = settings.operator_lat == 0.0 and settings.operator_lon == 0.0
    if not settings.autonomy_enabled or coords_unset:
        reason = "no_operator_location" if coords_unset else "autonomy_disabled_dry_run"
        audit["skipped_reason"] = reason
        _append_log(settings, audit)
        log.info("alert.dry_run", reason=reason, would_wake=[t["id"] for t in fresh])
        return

    # Bundle all fresh trips into ONE wake.
    kinds = sorted({t["kind"] for t in fresh})
    summary = (
        f"{len(fresh)} alert{'s' if len(fresh) != 1 else ''} near you: "
        + "; ".join(t["description"] for t in fresh[:5])
    )
    trigger = {
        "kinds": kinds,
        "summary": summary,
        "events": [
            {"kind": t["kind"], "description": t["description"], "data": t["data"]}
            for t in fresh
        ],
    }

    from ..autonomy import trigger_autonomous_turn
    try:
        await trigger_autonomous_turn(trigger, bus)
        audit["woke"] = True
    except Exception as e:  # noqa: BLE001
        audit["wake_error"] = str(e)
        log.warning("alert.wake_failed", error=str(e))

    # Consume dedup + daily cap; prune identities older than a day to bound size.
    today = _today_iso()
    for t in fresh:
        identities[t["id"]] = now
    identities = {k: v for k, v in identities.items() if now - float(v) < 86400.0}
    state["identities"] = identities
    state["fires_today"] = (
        int(state.get("fires_today") or 0) + 1 if state.get("day_iso") == today else 1
    )
    state["day_iso"] = today
    # Stamp the wake time (post-wake) so the chat-active gate can tell our own
    # append_turn write apart from a real operator turn on the next poll.
    state["last_wake_unix"] = time.time()
    _write_state(settings, state)
    _append_log(settings, audit)
    log.info("alert.woke", kinds=kinds, n=len(fresh))


async def alert_monitor_loop(bus: EventBus) -> None:
    """Background loop. Starts the persistent AIS monitor (ships), then polls all
    sources every `alert_poll_interval_seconds`, waking on fresh trips. Self-
    cancellable. No-op when alert_monitor_enabled is False."""
    settings = get_settings()
    if not settings.alert_monitor_enabled:
        log.info("alert.monitor.disabled")
        return

    interval = max(30, settings.alert_poll_interval_seconds)
    if settings.operator_lat == 0.0 and settings.operator_lon == 0.0:
        log.warning(
            "alert.monitor.no_operator_location",
            note="operator_lat/lon unset (0,0) — geo alerts would evaluate null island; "
            "set LUMOS_OPERATOR_LAT/LON. Running dry-run only until configured.",
        )
    log.info("alert.monitor.started", interval_s=interval, autonomy=settings.autonomy_enabled)

    # Persistent AIS cache child task (ships). Best-effort; no key → no-op.
    ais_task = asyncio.create_task(
        maritime.ais_monitor_loop(
            settings.operator_lat, settings.operator_lon, settings.alert_ship_radius_km
        )
    )
    try:
        while True:
            try:
                await asyncio.sleep(interval)
                await _poll_once(settings, bus)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("alert.monitor.iter_failed", error=str(e))
    finally:
        ais_task.cancel()
        try:
            await ais_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
