"""Anticipatory-forecast tool (Phase 39) — read-only look-ahead.

Passive/observe-only, so an autonomous wake AND the dawn briefing can both use
it. Wraps telemetry.forecast.build_forecast (celestial look-ahead + upcoming
military-recon satellite passes + NOAA Kp 3-day forecast).
"""

from __future__ import annotations

from . import register
from ..log import get_logger
from ..telemetry import forecast


log = get_logger(__name__)


@register(
    name="get_forecast",
    description=(
        "Anticipatory forecast — what's ABOUT to happen near a location over the "
        "next several hours, not just the current state. Returns: upcoming "
        "military-recon satellite passes (culmination time + peak elevation + "
        "bearing — the 'someone's watching overhead soon' heads-up); the NOAA "
        "3-day geomagnetic Kp forecast (peak predicted Kp + time, so you can warn "
        "BEFORE a storm lands — bio-impact look-ahead); and a celestial look-ahead "
        "(when Regulus next transits/rises/sets — the RHC anchor — plus next "
        "sunrise/sunset and when the current planetary hour ends). Call when the "
        "operator asks what's coming up / later today / tonight / the next pass / "
        "the look-ahead. Defaults to operator location when lat/lon omitted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
        },
        "required": [],
    },
)
async def get_forecast(lat: float | None = None, lon: float | None = None) -> dict:
    return await forecast.build_forecast(lat, lon)
