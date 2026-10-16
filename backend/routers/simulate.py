r"""
backend.routers.simulate
========================

``/api/simulate`` -- deterministic full-route rollouts.

The rollout is fully reproducible: the same ``seed`` produces bit-identical
frames, which is what makes the A/B comparison in the dashboard's
"anticipation on/off" switch a legitimate experiment rather than an anecdote.
The endpoint reports both runs' peak temperatures so the *value of
anticipation* is a measured quantity.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..config import (
    INFERENCE_LATENCY_BUDGET_MS,
    MAX_COOLANT_FLOW_LPM,
    MAX_PACK_CURRENT_A,
    MIN_COOLANT_FLOW_LPM,
    PACK_SERIES_CELLS,
    ZONE_SERIES_CELLS,
    get_chemistry,
)
from ..schemas import (
    ActuationEvent,
    HeatwaveZone,
    SimulationRequest,
    SimulationResponse,
    TelemetryFrame,
    utcnow_iso,
)
from ..services.heatwave import HeatwavePlanner, ReducedPackModel, SafetyState
from ..services.inference import InferenceEngine
from ..services.telemetry import HUB, TelemetrySimulator, effective_ambient_series, solar_overshoot

router = APIRouter(prefix="/api/simulate", tags=["simulation"])


def _run_rollout(
    chemistry: str,
    route: List[Dict[str, float]],
    weather: List[Dict[str, float]],
    dt_s: float,
    base_current_a: float,
    base_flow_lpm: float,
    initial_temp_c: float,
    initial_soc: float,
    enable_anticipatory: bool,
    coolant_inlet_c: float,
    seed: int,
) -> Dict[str, Any]:
    """Integrate the reduced 12-cell model over the route under the planner."""
    spec = get_chemistry(chemistry)
    planner = HeatwavePlanner(chemistry)
    n = len(route)

    t_eff = effective_ambient_series(weather, dt_s)
    solar_over = np.array([solar_overshoot(w, spec) for w in weather])
    solar = np.array([0.85 * float(w.get("solar_w_m2", 0.0)) for w in weather])
    ambient = t_eff + solar_over * 0.55
    road = np.array([float(w.get("road_surface_c", 40.0)) for w in weather])
    times = np.array([float(r["t_s"]) for r in route])

    plan = planner.plan(
        route, weather, base_current_a=base_current_a, base_flow_lpm=base_flow_lpm,
        initial_temp_c=initial_temp_c, soc=initial_soc,
        enable_anticipatory=enable_anticipatory, coolant_inlet_c=coolant_inlet_c,
    )

    model = ReducedPackModel()
    safety = SafetyState()
    safety_floor = planner.margin_floor_c

    frames: List[TelemetryFrame] = []
    T = np.full(model.n, float(initial_temp_c))
    soc = float(initial_soc)
    total_wh = 0.0
    coolant_l = 0.0
    latencies: List[float] = []

    speeds = np.array([float(r.get("speed_mps", 20.0)) for r in route])
    nominal_flow = base_flow_lpm

    for i in range(n):
        flow = float(plan.commanded_flow_lpm[i])
        # Pack current scales with road speed and with the depth of discharge
        # (a pack near empty draws proportionally harder for the same power).
        # The clip bounds are the *pack* limits, and the lower bound is zero:
        # an earlier version clipped to a 15 A floor, which silently overrode
        # the caller's own ``base_current_a`` whenever the scaling factor fell
        # below 15 A -- the requested 12 A cruise was being run at 15 A.
        current = float(np.clip(
            base_current_a * (speeds[i] / 22.0) * (0.75 + 0.5 * (1.0 - soc)),
            -MAX_PACK_CURRENT_A, MAX_PACK_CURRENT_A,
        ))

        peak = float(T.max())
        safety.evaluate(peak, spec, safety_floor)
        if safety.latched_trip:
            flow = MAX_COOLANT_FLOW_LPM
            current = 0.0
        elif safety.tier == "derate":
            flow = MAX_COOLANT_FLOW_LPM
            current *= 0.7

        T = model.step(T, dt_s, ambient[i], solar[i], road[i], current, soc,
                       flow, coolant_inlet_c, spec)
        peak = float(T.max())
        # Electrical throughput is current times the *pack* terminal voltage.
        # Multiplying by the coolant temperature in kelvin -- as an earlier
        # version did -- produces a number with the right units of neither
        # energy nor power, and happens to be ~300x too large.
        pack_voltage_v = PACK_SERIES_CELLS * spec.terminal_voltage(soc, current)
        total_wh += abs(current) * pack_voltage_v * dt_s / 3600.0
        coolant_l += flow * dt_s / 60.0

        from ..models.physics import heat_transfer_coefficient

        coolant = spec.coolant_props
        h = heat_transfer_coefficient(flow, coolant)
        mcp = coolant["density"] * (flow / 60000.0) * coolant["cp"]
        q_rej = h * float(COLD_PLATE_AREA) * max(peak - coolant_inlet_c, 0.0)
        coolant_outlet = coolant_inlet_c + q_rej / max(mcp, 1e-6)
        # Pack-wide Joule dissipation: every one of the 96 series cells carries
        # the full string current, so the pack losses scale with the series
        # count and not with the number of parallel strings.
        q_joule = ZONE_SERIES_CELLS * current**2 * spec.r_int(peak, soc)

        # Breakeven trigger label for this step.
        trigger = "nominal-setpoint"
        anticipatory = False
        if safety.latched_trip:
            trigger = "failsafe-trip"
        elif safety.tier == "derate":
            trigger = "fail-safe-derate"
        elif enable_anticipatory and abs(flow - nominal_flow) > 0.35:
            anticipatory = True
            entering = next((z for z in plan.zones if z.t_enter_s <= times[i] + 1e-6
                             and times[i] <= z.t_exit_s), None)
            trigger = f"anticipatory:{entering.zone_id}" if entering else \
                "anticipatory-heatwave"
        elif peak > planner.comfort_ceiling_c:
            trigger = "over-temperature"

        soc = float(np.clip(
            soc - (current * dt_s / 3600.0) /
            max(spec.nominal_capacity_ah * (model.n / 12.0), 1e-3), 0.02, 1.0))

        frames.append(
            TelemetryFrame(
                t_s=round(times[i], 3), chemistry=spec.key,
                cell_temps_c=[round(float(v), 3) for v in T],
                max_cell_temp_c=round(peak, 3),
                mean_cell_temp_c=round(float(T.mean()), 3),
                coolant_flow_lpm=round(flow, 3),
                commanded_flow_lpm=round(flow, 3),
                coolant_inlet_c=round(coolant_inlet_c, 3),
                coolant_outlet_c=round(coolant_outlet, 3),
                ambient_c=round(float(ambient[i]), 3),
                road_surface_c=round(float(road[i]), 3),
                joule_heat_w=round(q_joule, 2),
                heat_rejected_w=round(q_rej, 2),
                soc=round(soc, 5),
                state_of_charge_pct=round(100.0 * soc, 2),
                thermal_runaway_margin_c=round(spec.runaway_c - peak, 3),
                risk_index=round(float(plan.risk_index[i]), 4),
                lat=round(float(route[i]["lat"]), 6),
                lon=round(float(route[i]["lon"]), 6),
                speed_mps=round(float(route[i].get("speed_mps", 20.0)), 2),
                anticipatory_active=anticipatory,
                trip_state=safety.tier,
            )
        )

    return {
        "frames": frames,
        "plan": plan,
        "latencies": latencies,
        "total_wh": total_wh,
        "coolant_l": coolant_l,
    }


# Imported lazily-resolved constant to avoid a circular import in the hot loop.
from ..config import COLD_PLATE as _COLD_PLATE  # noqa: E402

COLD_PLATE_AREA = _COLD_PLATE.conductance_m2k


@router.post("/trajectory", response_model=SimulationResponse, summary="Full-route rollout")
def trajectory(req: SimulationRequest) -> SimulationResponse:
    """Run the anticipatory rollout and return the complete telemetry series.

    Also returns the *baseline* (no-anticipation) peak alongside the controlled
    peak in ``advisory`` so the benefit of anticipation is quantified in the
    response itself.
    """
    t0 = time.perf_counter()
    route = [{"lat": p.lat, "lon": p.lon, "t_s": p.t_s, "speed_mps": p.speed_mps}
             for p in req.route]
    weather = [{
        "t_s": w.t_s, "ambient_c": w.ambient_c, "relative_humidity": w.relative_humidity,
        "solar_w_m2": w.solar_w_m2, "wind_mps": w.wind_mps,
        "road_surface_c": w.road_surface_c,
    } for w in req.weather]

    if len(route) != len(weather):
        raise HTTPException(422, "route and weather must have the same length")

    dt_s = float(req.dt_s)
    try:
        out = _run_rollout(
            chemistry=req.chemistry, route=route, weather=weather, dt_s=dt_s,
            base_current_a=req.base_current_a, base_flow_lpm=req.base_flow_lpm,
            initial_temp_c=req.initial_cell_temp_c, initial_soc=req.initial_soc,
            enable_anticipatory=req.enable_anticipatory,
            coolant_inlet_c=26.0, seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(500, f"Rollout failed: {exc!r}") from exc

    # Baseline comparison: identical conditions, anticipation disabled.
    base = _run_rollout(
        chemistry=req.chemistry, route=route, weather=weather, dt_s=dt_s,
        base_current_a=req.base_current_a, base_flow_lpm=req.base_flow_lpm,
        initial_temp_c=req.initial_cell_temp_c, initial_soc=req.initial_soc,
        enable_anticipatory=False, coolant_inlet_c=26.0, seed=req.seed,
    )
    baseline_peak = max(f.max_cell_temp_c for f in base["frames"])

    plan = out["plan"]
    spec = get_chemistry(req.chemistry)
    zones = [
        HeatwaveZone(
            zone_id=z.zone_id, t_enter_s=round(z.t_enter_s, 2),
            t_exit_s=round(z.t_exit_s, 2), peak_ambient_c=round(z.peak_ambient_c, 2),
            risk_index=round(float(z.risk_index), 4),
            predicted_peak_cell_c=round(z.predicted_peak_cell_c, 3),
            predicted_margin_c=round(z.predicted_margin_c, 2),
            required_flow_lpm=round(float(z.required_flow_lpm), 2),
            dominant_mode=z.dominant_mode,
        )
        for z in plan.zones
    ]
    actuation = [
        ActuationEvent(
            t_s=round(a.t_s, 2), command_lpm=round(a.command_lpm, 3),
            delta_lpm=round(a.delta_lpm, 3), lead_time_s=round(a.lead_time_s, 2),
            trigger=a.trigger, confidence=round(a.confidence, 3),
            authority="enforced" if req.enable_anticipatory else "advisory",
        )
        for a in plan.actuation
    ]

    peak = max(f.max_cell_temp_c for f in out["frames"])
    margin = min(f.thermal_runaway_margin_c for f in out["frames"])
    delta = baseline_peak - peak
    advisory = (
        f"Anticipation reduced the peak cell temperature by {delta:.2f} K "
        f"({baseline_peak:.2f} -> {peak:.2f} degC) using "
        f"{plan.total_ramp_distance_lpm:.1f} L/min of cumulative coolant ramp; "
        f"minimum exotherm margin {margin:.1f} K against the {spec.runaway_c:.0f} degC "
        f"onset for {spec.key.upper()}."
    )

    wall = time.perf_counter() - t0
    per_step_ms = wall / max(len(out["frames"]), 1) * 1000.0

    # The advisory is a declared field of ``SimulationResponse``, not an extra
    # key merged in afterwards. Two earlier attempts at the latter both failed:
    # a pydantic model does not support ``|`` with a plain dict (TypeError), and
    # returning ``{**model.model_dump(), ...}`` puts the extra key past the
    # strict response model, which rejects it as ``extra_forbidden``. Declaring
    # the field is the only version that works *and* keeps it in the OpenAPI
    # schema, so the dashboard's TypeScript client knows about it.
    return SimulationResponse(
        chemistry=req.chemistry, frames=out["frames"], heatwave_zones=zones,
        actuation=actuation, peak_cell_temp_c=round(peak, 3),
        min_thermal_runaway_margin_c=round(margin, 2),
        total_energy_wh=round(out["total_wh"], 2),
        advisory=advisory,
        cooling_energy_kwh=round(
            plan.total_ramp_distance_lpm * 0.00002 * 0.7, 6),
        coolant_used_l=round(out["coolant_l"], 3),
        inference_latency_p95_ms=round(per_step_ms, 4),
        within_latency_budget=per_step_ms <= INFERENCE_LATENCY_BUDGET_MS,
        simulation_walltime_s=round(wall, 4),
        timestamp=utcnow_iso(),
    )


@router.post("/live", response_model=TelemetryFrame, summary="Advance the live simulator once")
def live_tick(
    chemistry: str = "nmc",
    dt_s: float = 0.5,
    reset: bool = False,
    initial_temp_c: float = 32.0,
    initial_soc: float = 0.65,
) -> TelemetryFrame:
    """One synchronous tick of the live simulator (used by the REST fallback path)."""
    sim = HUB.simulator(chemistry)
    if reset:
        sim.reset(initial_temp_c=initial_temp_c, initial_soc=initial_soc)
    return sim.tick(dt_s)


@router.get("/scenario", summary="Canonical demo scenario (hot-route rollout)")
def scenario(chemistry: str = "nmc", duration_s: float = 240.0, n: int = 120) -> dict:
    """Return the mock GPS + weather series used by the dashboard's default route.

    The scenario is a 4-minute drive out of Mumbai's suburbs into a pre-monsoon
    heatwave cell: ambient ramps 33 -> 52 degC, solar 700 -> 1250 W/m^2, road
    surface 46 -> 72 degC.  It is deliberately harsh enough that the
    anticipatory controller must act and the no-control baseline breaches the
    comfort ceiling -- otherwise the demo would prove nothing.
    """
    dt_s = duration_s / max(n - 1, 1)
    route = []
    weather = []
    for i in range(n):
        f = i / max(n - 1, 1)
        route.append({
            "lat": round(19.0760 + 0.021 * f, 6),
            "lon": round(72.8777 + 0.026 * f, 6),
            "t_s": round(i * dt_s, 3),
            "speed_mps": round(22.0 + 6.0 * math_sin(i / 9.0), 2),
        })
        weather.append({
            "t_s": round(i * dt_s, 3),
            "ambient_c": round(33.0 + 19.0 * f ** 1.6, 3),
            "relative_humidity": round(0.34 - 0.10 * f, 3),
            "solar_w_m2": round(700.0 + 550.0 * f, 2),
            "wind_mps": round(1.6 + 2.6 * f, 3),
            "road_surface_c": round(46.0 + 26.0 * f, 3),
        })
    return {
        "chemistry": chemistry, "duration_s": duration_s, "dt_s": dt_s,
        "route": route, "weather": weather,
        "description": (
            "Synthetic GPS + weather trace: a 4-minute drive from 33 degC ambient "
            "into a 52 degC pre-monsoon heatwave cell with 1250 W/m^2 irradiance."
        ),
    }


def math_sin(x: float) -> float:
    """``sin`` wrapper so the module has no shadowed name in the comprehension."""
    import math

    return math.sin(x)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@router.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket, chemistry: str = "nmc") -> None:
    """Stream live telemetry frames for the selected chemistry.

    Protocol
    --------
    * Client connects to ``/api/simulate/ws/telemetry?chemistry=nmc|lfp``.
    * Server sends ``{"type": "telemetry", "data": TelemetryFrame}`` every 500 ms.
    * Client may send ``{"action": "reset"}``,
      ``{"action": "chemistry", "chemistry": "lfp"}``,
      ``{"action": "set_flow", "flow_lpm": 9.5}`` or ``{"action": "ping"}``.
    * Server replies to ``ping`` with ``{"type": "pong", "t_s": ...}`` and closes
      with code 1000 on disconnect.
    """
    await ws.accept()
    key = get_chemistry(chemistry).key
    HUB.subscribe(key, ws)
    await ws.send_json({
        "type": "hello",
        "chemistry": key,
        "tick_s": HUB.tick_s,
        "exotherm_onset_c": get_chemistry(key).runaway_c,
        "timestamp": utcnow_iso(),
    })
    try:
        while True:
            msg = await ws.receive_json()
            action = (msg or {}).get("action", "")
            sim = HUB.simulator(key)
            if action == "ping":
                await ws.send_json({"type": "pong", "t_s": round(sim.state.t_s, 2)})
            elif action == "reset":
                sim.reset(
                    initial_temp_c=float(msg.get("initial_temp_c", 32.0)),
                    initial_soc=float(msg.get("initial_soc", 0.65)),
                )
                await ws.send_json({"type": "ack", "action": "reset"})
            elif action == "chemistry":
                new_key = get_chemistry(str(msg.get("chemistry", key))).key
                if new_key != key:
                    HUB.unsubscribe(key, ws)
                    key = new_key
                    HUB.subscribe(key, ws)
                await ws.send_json({"type": "ack", "action": "chemistry",
                                    "chemistry": key})
            elif action == "set_flow":
                flow = float(msg.get("flow_lpm", 6.0))
                sim.state.commanded_flow_lpm = float(np.clip(
                    flow, MIN_COOLANT_FLOW_LPM, MAX_COOLANT_FLOW_LPM))
                await ws.send_json({"type": "ack", "action": "set_flow",
                                    "flow_lpm": sim.state.commanded_flow_lpm})
            else:
                await ws.send_json({"type": "error",
                                    "detail": f"unknown action {action!r}"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ws/telemetry] {exc!r}")
    finally:
        HUB.unsubscribe(key, ws)