"""Anticipatory forecasts (Phase 39) — what's ABOUT to happen, not just what is.

Three look-ahead layers, each cheap on its own:
  * celestial — reuses grimoire's already-computed "next" timestamps (Regulus
    transit/rise/set — the RHC anchor; next sunrise/sunset; when the current
    planetary hour ends). No new astro code.
  * sat passes — skyfield `find_events` over the next N hours for military-recon
    satellites (the "someone's watching" heads-up). Filtered to that mission so
    the propagation sweep stays bounded; result cached 10 min.
  * Kp forecast — NOAA SWPC planetary-K-index 3-day product, so Lumos can warn
    "Kp could climb to 5 this evening" BEFORE the storm lands (bio-impact
    look-ahead — Bz/solar-wind lead Kp by hours; the forecast leads by more).

Surfaced via the passive `get_forecast` tool and folded into the dawn briefing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache
from .satellites import _classify, fetch_tle


log = get_logger(__name__)

_KP_FORECAST_URL = (
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
)
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_MAX_SATS_SCANNED = 500  # bound the find_events sweep (mil-recon subset is small)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def fetch_kp_forecast() -> dict[str, Any]:
    """NOAA planetary K-index 3-day forecast. Returns {ok, peak, upcoming[], fetched_at}.

    `peak` is the highest PREDICTED Kp in the upcoming window (with its time);
    `upcoming` is the predicted series (3-hour cadence). Cached 30 min."""
    cache_key = "kp_forecast"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached
    fetched_at = _now_utc().isoformat(timespec="seconds")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_KP_FORECAST_URL, timeout=_TIMEOUT)
            r.raise_for_status()
            rows = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.info("forecast.kp_fetch_failed", error=str(e))
        result = {"ok": False, "error": str(e), "fetched_at": fetched_at}
        tcache.put(cache_key, result, ttl_seconds=1800)
        return result

    # NOAA returns a list of dicts: {time_tag, kp, observed: 'observed'|'predicted',
    # noaa_scale}. No header row.
    series: list[dict[str, Any]] = []
    for row in (rows if isinstance(rows, list) else []):
        if not isinstance(row, dict):
            continue
        try:
            series.append({
                "time_tag": row.get("time_tag"),
                "kp": round(float(row.get("kp")), 2),
                "kind": str(row.get("observed") or ""),
            })
        except (ValueError, TypeError):
            continue

    predicted = [s for s in series if s["kind"].lower() == "predicted"]
    peak = max(predicted, key=lambda s: s["kp"], default=None) if predicted else None
    result = {
        "ok": True,
        "peak": peak,
        "upcoming": predicted[:16],  # ~2 days at 3h cadence
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=1800)
    return result


def _predict_passes(
    tles: list[dict[str, str]],
    lat: float,
    lon: float,
    hours: float,
    min_elevation: float,
    missions: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """SYNC (runs in a thread) — next culminating passes over the observer in the
    next `hours`, restricted to `missions`. One bad TLE never kills the sweep."""
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale(builtin=True)  # offline — no leap-second download
    now_dt = _now_utc()
    t0 = ts.from_datetime(now_dt)
    t1 = ts.from_datetime(now_dt + timedelta(hours=hours))
    observer = wgs84.latlon(lat, lon)

    candidates = [t for t in tles if _classify(t["name"]) in missions][:_MAX_SATS_SCANNED]
    passes: list[dict[str, Any]] = []
    for tle in candidates:
        try:
            sat = EarthSatellite(tle["line1"], tle["line2"], tle["name"], ts)
            times, events = sat.find_events(
                observer, t0, t1, altitude_degrees=min_elevation
            )
            for i in range(len(events)):
                if int(events[i]) != 1:  # 0=rise, 1=culminate, 2=set
                    continue
                ti = times[i]
                alt, az, _dist = (sat - observer).at(ti).altaz()
                passes.append({
                    "name": tle["name"],
                    "mission": _classify(tle["name"]),
                    "culmination_utc": ti.utc_iso(),
                    "peak_elevation_deg": round(alt.degrees, 1),
                    "azimuth_deg": round(az.degrees, 1),
                })
        except Exception:  # noqa: BLE001 — decayed/malformed TLE; skip it
            continue

    passes.sort(key=lambda p: p["culmination_utc"])
    return passes[:limit]


async def fetch_sat_passes(
    lat: float | None = None,
    lon: float | None = None,
    hours: float = 8.0,
    min_elevation: float = 30.0,
    missions: tuple[str, ...] = ("military_recon",),
    limit: int = 12,
) -> dict[str, Any]:
    """Upcoming culminating passes over the observer. Defaults to operator
    location + military-recon over the next 8 h above 30°. Cached 10 min."""
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"sat_fc_{lat:.3f}_{lon:.3f}_{int(hours)}_{int(min_elevation)}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    tles = await fetch_tle()
    fetched_at = _now_utc().isoformat(timespec="seconds")
    if not tles:
        result = {"ok": False, "error": "no TLE data available (SatNOGS)",
                  "count": 0, "passes": [], "fetched_at": fetched_at}
        tcache.put(cache_key, result, ttl_seconds=600)
        return result

    try:
        passes = await asyncio.to_thread(
            _predict_passes, tles, lat, lon, hours, min_elevation, set(missions), limit
        )
    except Exception as e:  # noqa: BLE001
        log.warning("forecast.pass_predict_failed", error=str(e))
        result = {"ok": False, "error": f"pass prediction failed: {e}",
                  "count": 0, "passes": [], "fetched_at": fetched_at}
        tcache.put(cache_key, result, ttl_seconds=600)
        return result

    result = {
        "ok": True,
        "count": len(passes),
        "passes": passes,
        "window_hours": hours,
        "missions": list(missions),
        "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=600)
    return result


async def build_forecast(
    lat: float | None = None, lon: float | None = None
) -> dict[str, Any]:
    """Composite look-ahead: celestial (grimoire) + sat passes + Kp forecast.
    Each layer fails independently — a dead feed just drops its section."""
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon
    out: dict[str, Any] = {
        "ok": True,
        "center": {"lat": lat, "lon": lon},
        "fetched_at": _now_utc().isoformat(timespec="seconds"),
    }

    # Celestial look-ahead — reuse grimoire's already-computed "next" timestamps.
    try:
        from . import grimoire
        gt = await grimoire.fetch_grid_timing(lat, lon)
        if gt.get("ok"):
            ph = gt.get("planetary_hour", {}) or {}
            solar = gt.get("solar", {}) or {}
            reg = (gt.get("fixed_stars", {}) or {}).get("Regulus", {}) or {}
            out["celestial"] = {
                "current_planetary_hour_ruler": ph.get("ruler"),
                "planetary_hour_ends_local": ph.get("hour_end_local"),
                "sunset_local": solar.get("sunset_local"),
                "next_sunrise_local": solar.get("next_sunrise_local"),
                "regulus_above_horizon": reg.get("above_horizon"),
                "regulus_next_transit_utc": reg.get("next_transit_utc"),
                "regulus_next_rising_utc": reg.get("next_rising_utc"),
                "regulus_next_setting_utc": reg.get("next_setting_utc"),
            }
    except Exception as e:  # noqa: BLE001
        log.info("forecast.celestial_failed", error=str(e))

    try:
        out["sat_passes"] = await fetch_sat_passes(lat, lon)
    except Exception as e:  # noqa: BLE001
        log.info("forecast.sat_passes_failed", error=str(e))

    try:
        out["kp_forecast"] = await fetch_kp_forecast()
    except Exception as e:  # noqa: BLE001
        log.info("forecast.kp_failed", error=str(e))

    return out
