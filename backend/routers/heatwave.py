r"""
backend.routers.heatwave
=======================

``/api/heatwave`` -- anticipatory route planning (pure planning, no rollout).

Kept deliberately separate from :mod:`backend.routers.simulate` so that the
dashboard can ask "what *would* you do?" ("advisory" authority, displayed as a
dashed pre-emption trace) before committing to "what did you do?" ("enforced").
That separation is the single most important UI distinction in this project:
a predictive recommendation and a commanded actuator move must never be
rendered with the same visual weight.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..config import MAX_COOLANT_FLOW_LPM, MIN_COOLANT_FLOW_LPM, get_chemistry
from ..schemas import (
    ActuationEvent,
    ForecastRequest,
    ForecastResponse,
    HeatwaveZone,
    utcnow_iso,
)
from ..services.heatwave import HeatwavePlanner, dt_or
from ..services.telemetry import effective_ambient_series, solar_overshoot

router = APIRouter(prefix="/api/heatwave", tags=["heatwave"])


def _to_plan_inputs(req: ForecastRequest) -> tuple:
    """Flatten the pydantic request into the dict lists the planner expects."""
    route = [
        {"lat": p.lat, "lon": p.lon, "t_s": p.t_s, "speed_mps": p.speed_mps}
        for p in req.route
    ]
    weather = [
        {
            "t_s": w.t_s, "ambient_c": w.ambient_c,
            "relative_humidity": w.relative_humidity, "solar_w_m2": w.solar_w_m2,
            "wind_mps": w.wind_mps, "road_surface_c": w.road_surface_c,
        }
        for w in req.weather
    ]
    return route, weather


@router.post("/forecast", response_model=ForecastResponse, summary="Anticipatory forecast")
def forecast(req: ForecastRequest) -> ForecastResponse:
    """Compute the anticipatory coolant schedule for a route without applying it.

    Returns the detected zones, the advisory schedule (all events marked
    ``authority="advisory"``) and the total ramp distance -- the energy-proxy
    metric used to justify that the controller does not over-cool.
    """
    try:
        route, weather = _to_plan_inputs(req)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(422, str(exc)) from exc

    if len(route) != len(weather):
        raise HTTPException(
            422,
            f"route ({len(route)}) and weather ({len(weather)}) must have equal length; "
            "the demo weather service is co-indexed with the GPS fixes",
        )

    spec = get_chemistry(req.chemistry)
    planner = HeatwavePlanner(req.chemistry)
    current = (req.pack_heat_w ** 0.5) if req.pack_heat_w > 0 else 140.0

    try:
        plan = planner.plan(
            route, weather,
            base_current_a=current,
            base_flow_lpm=6.0,
            initial_temp_c=34.0,
            soc=req.state_of_charge,
            enable_anticipatory=True,
            pack_heat_w=req.pack_heat_w if req.pack_heat_w > 0 else None,
        )
    except Exception as exc:
        raise HTTPException(500, f"Heatwave planning failed: {exc!r}") from exc

    zones: List[HeatwaveZone] = [
        HeatwaveZone(
            zone_id=z.zone_id,
            t_enter_s=round(z.t_enter_s, 2),
            t_exit_s=round(z.t_exit_s, 2),
            peak_ambient_c=round(z.peak_ambient_c, 2),
            risk_index=round(float(z.risk_index), 4),
            predicted_peak_cell_c=round(z.predicted_peak_cell_c, 3),
            predicted_margin_c=round(z.predicted_margin_c, 2),
            required_flow_lpm=round(float(z.required_flow_lpm), 2),
            dominant_mode=z.dominant_mode,
        )
        for z in plan.zones
    ]

    schedule: List[ActuationEvent] = [
        ActuationEvent(
            t_s=round(a.t_s, 2),
            command_lpm=round(a.command_lpm, 3),
            delta_lpm=round(a.delta_lpm, 3),
            lead_time_s=round(a.lead_time_s, 2),
            trigger=a.trigger,
            confidence=round(a.confidence, 3),
            authority="advisory",
        )
        for a in plan.actuation
    ]

    return ForecastResponse(
        zones=zones,
        schedule=schedule,
        total_ramp_distance_lpm=round(plan.total_ramp_distance_lpm, 3),
        max_predicted_cell_temp_c=round(plan.peak_cell_c, 3),
        advisory=plan.advisory,
    )


@router.get("/explain", summary="Explain the controller's current decision")
def explain(
    chemistry: str = "nmc",
    index: int = 0,
    route_samples: int = 60,
    duration_s: float = 180.0,
) -> dict:
    """Return the deciding terms for one timestep of a canonical hot route.

    This exists so the UI never has to invent an explanation: every number it
    shows in the "why" panel comes from :meth:`HeatwavePlanner.explain`.
    """
    spec = get_chemistry(chemistry)
    planner = HeatwavePlanner(chemistry)
    dt_s = duration_s / max(route_samples - 1, 1)
    route = [
        {
            "lat": 19.0760 + 0.01 * i / route_samples,
            "lon": 72.8777 + 0.012 * i / route_samples,
            "t_s": i * dt_s, "speed_mps": 22.0,
        }
        for i in range(route_samples)
    ]
    weather = [
        {
            "t_s": i * dt_s,
            # Ramp from 34 degC to 52 degC across the route: a canonical
            # "drive into a heatwave" scenario.
            "ambient_c": 34.0 + 18.0 * (i / max(route_samples - 1, 1)) ** 1.6,
            "solar_w_m2": 850.0 + 380.0 * (i / max(route_samples - 1, 1)),
            "wind_mps": 2.4, "road_surface_c": 46.0 + 22.0 * (i / max(route_samples - 1, 1)),
            "relative_humidity": 0.3,
        }
        for i in range(route_samples)
    ]
    plan = planner.plan(route, weather, base_current_a=180.0, base_flow_lpm=6.0,
                        initial_temp_c=34.0, soc=0.7)
    return {
        "chemistry": chemistry,
        "exotherm_onset_c": spec.runaway_c,
        "decision": planner.explain(plan, index),
        "advisory": plan.advisory,
        "decision_log": plan.decision_log,
        "zones": [
            {
                "zone_id": z.zone_id,
                "t_enter_s": z.t_enter_s, "t_exit_s": z.t_exit_s,
                "reason": z.reason, "dominant_mode": z.dominant_mode,
                "risk_index": z.risk_index,
                "required_flow_lpm": z.required_flow_lpm,
            }
            for z in plan.zones
        ],
        "timestamp": utcnow_iso(),
    }


@router.get("/ambient", summary="Effective ambient forecast series")
def ambient_series(
    chemistry: str = "nmc",
    duration_s: float = 300.0,
    n: int = 120,
    ambient_start_c: float = 32.0,
    ambient_end_c: float = 50.0,
) -> dict:
    r"""Expose the lag-filtered ambient the *pack* sees, not the raw weather.

    The difference between the two is exactly the anticipation window: at a
    420 s lag, a rising ambient is already known to the pack ~7 minutes before
    the cabin thermometer reads it.
    """
    spec = get_chemistry(chemistry)
    dt_s = duration_s / max(n - 1, 1)
    raw = [
        {
            "t_s": i * dt_s,
            "ambient_c": ambient_start_c + (ambient_end_c - ambient_start_c)
            * (i / max(n - 1, 1)) ** 1.5,
            "solar_w_m2": 700.0 + 500.0 * (i / max(n - 1, 1)),
            "wind_mps": 2.5, "road_surface_c": ambient_start_c + 12.0
            + 20.0 * (i / max(n - 1, 1)),
            "relative_humidity": 0.3,
        }
        for i in range(n)
    ]
    eff = effective_ambient_series(raw, dt_s)
    solar_over = [solar_overshoot(w, spec) for w in raw]
    return {
        "chemistry": chemistry,
        "t_s": [round(i * dt_s, 2) for i in range(n)],
        "raw_ambient_c": [round(w["ambient_c"], 3) for w in raw],
        "effective_ambient_c": [round(float(e), 3) for e in eff],
        "solar_overshoot_k": [round(float(s), 3) for s in solar_over],
        "pack_ambient_c": [round(float(eff[i] + 0.55 * solar_over[i]), 3)
                           for i in range(n)],
        "comfort_ceiling_c": planner_ceiling(chemistry),
        "exotherm_onset_c": spec.runaway_c,
        "note": (
            "First-order lag (tau = 420 s) models the enclosure thermal mass; the "
            "pack therefore anticipates the rise ~7 min before the raw ambient does."
        ),
    }


def planner_ceiling(chemistry: str) -> float:
    """Comfort ceiling for a chemistry (imported lazily to avoid a cycle)."""
    from ..services.heatwave import COMFORT_CEILING_C

    return COMFORT_CEILING_C[chemistry]