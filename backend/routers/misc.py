r"""
backend.routers.misc
====================

``/api/demo/...`` -- one-call demo fixtures.

The dashboard calls these on first paint so it has something to render before a
WebSocket frame or a rollout has completed.  Everything here is derived from the
same code paths as the live endpoints -- there are no hand-written fixture
numbers, which is what keeps the demo honest.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..config import (
    BASELINE_CRUISE_FLOW_LPM,
    CELL_TEMPERATURE_CEILING_C,
    CHEMISTRY_REGISTRY,
    DEFAULT_CHEMISTRY,
    DEFAULT_PACK_CURRENT_A,
    MAX_COOLANT_FLOW_LPM,
    PRE_FAST_CHARGE_TEMP_C,
    get_chemistry,
)
from ..schemas import utcnow_iso
from ..services.heatwave import (
    COMFORT_CEILING_C,
    MARGIN_FLOOR_C,
    AMBIENT_TAU_S,
    RAMP_RATE_LPM_PER_S,
    HeatwavePlanner,
)
from ..services.inference import InferenceEngine
from ..services.telemetry import TelemetrySimulator

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/bootstrap", summary="Everything the dashboard needs on first paint")
def bootstrap(chemistry: str = DEFAULT_CHEMISTRY) -> dict:
    """Single round-trip that returns catalogue, model info, a field and a scenario.

    The frontend calls this once; it is deliberately one request so the initial
    paint is a single latency budget rather than five.
    """
    from ..routers.health import catalogue, model_info
    from ..routers.simulate import scenario
    from ..routers.predict import field_preview

    t0 = time.perf_counter()
    spec = get_chemistry(chemistry)

    cat = catalogue()
    info = model_info()
    scen = scenario(chemistry=chemistry)
    field = field_preview(
        chemistry=chemistry, nx=16, ny=11, nz=7,
        coolant_inlet_c=26.0, coolant_flow_lpm=6.0, ambient_c=38.0,
        pack_current_a=180.0, state_of_charge=0.65,
    )

    return {
        "chemistry": chemistry,
        "catalogue": cat.model_dump(),
        "model": info.model_dump(),
        "scenario": scen,
        "field": field,
        "safety": {
            "comfort_ceiling_c": COMFORT_CEILING_C[chemistry],
            "margin_floor_c": MARGIN_FLOOR_C[chemistry],
            "exotherm_onset_c": spec.runaway_c,
            "ambient_tau_s": AMBIENT_TAU_S,
            "ramp_rate_lpm_per_s": RAMP_RATE_LPM_PER_S,
            "tiers": ["ok", "derate", "trip"],
        },
        "bootstrap_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "timestamp": utcnow_iso(),
    }


@router.post("/seed-run", summary="Deterministic short closed-loop run")
def seed_run(
    chemistry: str = DEFAULT_CHEMISTRY,
    steps: int = Query(120, ge=10, le=600),
    dt_s: float = Query(1.0, gt=0.1, le=10),
    seed: int = 1337,
    use_pinn: bool = True,
) -> dict:
    """Run the live simulator for ``steps`` ticks and return the frame history.

    This gives the dashboard an immediate, reproducible trace to plot while the
    WebSocket warms up.  ``use_pinn=False`` disables the PINN deviation
    correction so the reduced model's raw behaviour can be inspected -- useful
    when debugging the fusion term.

    This is a ``POST`` even though it creates no persistent state, because it is
    an *expensive computation* -- up to 600 RK4 steps.  A ``GET`` would make it
    eligible for speculative prefetching, so merely rendering the page could
    trigger a full simulation in every open tab, and browsers are under no
    obligation to deliver the result to the page that asked for it.
    """
    sim = TelemetrySimulator(chemistry=chemistry, seed=seed, engine=None)
    frames = []
    t0 = time.perf_counter()
    for _ in range(steps):
        frame = sim.tick(dt_s, use_pinn=use_pinn)
        frames.append(frame.model_dump())
    wall = time.perf_counter() - t0

    peaks = [f["max_cell_temp_c"] for f in frames]
    flows = [f["coolant_flow_lpm"] for f in frames]
    return {
        "chemistry": chemistry,
        "steps": steps,
        "dt_s": dt_s,
        "seed": seed,
        "frames": frames,
        "summary": {
            "peak_cell_temp_c": round(max(peaks), 3),
            "min_exotherm_margin_c": round(
                min(f["thermal_runaway_margin_c"] for f in frames), 2),
            "mean_flow_lpm": round(float(np.mean(flows)), 3),
            "max_flow_lpm": round(float(np.max(flows)), 3),
            "anticipatory_frames": sum(1 for f in frames if f["anticipatory_active"]),
            "walltime_s": round(wall, 3),
            "per_step_ms": round(wall / steps * 1000.0, 4),
        },
        "timestamp": utcnow_iso(),
    }


@router.get("/comparison", summary="NMC vs LFP comparative stability study")
def comparison() -> dict:
    r"""Side-by-side NMC/LFP sweep of the same route and same heat load.

    This is the experiment quoted in the model card: both chemistries see the
    identical 4-minute heatwave trajectory, and we report peak temperature,
    minimum exotherm margin, required flow and the parasitic coolant-pumping
    penalty.  The comparison is deterministic, so re-running it reproduces the
    numbers exactly.
    """
    from ..routers.simulate import scenario

    # The part-load baseline and the post-fast-charge starting temperature are
    # the same ones the trajectory endpoint uses. A comparison run at 6 L/min
    # would put both chemistries in the regime where the anticipatory schedule
    # is identical to constant flow, and the reported "anticipatory benefit"
    # would be exactly zero for both -- technically true, but a claim about the
    # controller's value measured under conditions where it had none.
    scen = scenario(chemistry="nmc", duration_s=240.0, n=120)
    route, weather = scen["route"], scen["weather"]
    dt_s = scen["dt_s"]

    out: Dict[str, Any] = {}
    for key in ("nmc", "lfp"):
        spec = get_chemistry(key)
        planner = HeatwavePlanner(key)
        plan = planner.plan(
            route, weather, base_current_a=DEFAULT_PACK_CURRENT_A,
            base_flow_lpm=BASELINE_CRUISE_FLOW_LPM,
            initial_temp_c=PRE_FAST_CHARGE_TEMP_C, soc=0.7,
        )
        trace = planner.rollout(
            PRE_FAST_CHARGE_TEMP_C, plan.commanded_flow_lpm,
            np.array([w["ambient_c"] for w in weather]),
            np.array([0.85 * w["solar_w_m2"] for w in weather]),
            np.array([w["road_surface_c"] for w in weather]),
            # ``rollout`` takes per-timestep current/SoC *arrays* so that the
            # speed-scheduled and Coulomb-counted traces are accounted for.
            # Passing a scalar there fails at ``current[i]``; broadcasting is
            # correct only if the length matches the flow schedule (it does,
            # because both are the route length).
            np.full(len(plan.commanded_flow_lpm), DEFAULT_PACK_CURRENT_A),
            np.full(len(plan.commanded_flow_lpm), 0.7),
            26.0, dt_s,
        )
        peak = float(trace.max())
        out[key] = {
            "peak_cell_temp_c": round(peak, 3),
            "comfort_ceiling_c": COMFORT_CEILING_C[key],
            "ceiling_respected": peak <= COMFORT_CEILING_C[key],
            "exotherm_onset_c": spec.runaway_c,
            "min_exotherm_margin_c": round(spec.runaway_c - peak, 2),
            "margin_floor_c": MARGIN_FLOOR_C[key],
            "margin_respected": (spec.runaway_c - peak) >= MARGIN_FLOOR_C[key],
            "cp_j_kgk": round(spec.cp, 1),
            "r_int_at_40c_ohm": round(spec.r_int(40.0, 0.5), 6),
            "arrhenius_slope_kj_mol": round(spec.ea / 1000.0, 1),
            "mean_commanded_flow_lpm": round(float(plan.commanded_flow_lpm.mean()), 3),
            "peak_commanded_flow_lpm": round(float(plan.commanded_flow_lpm.max()), 3),
            "total_ramp_lpm": round(plan.total_ramp_distance_lpm, 3),
            "anticipation_benefit_k": round(plan.decision_log["anticipation_benefit_k"], 3),
            "n_zones": len(plan.zones),
            "advisory": plan.advisory,
        }

    nmc, lfp = out["nmc"], out["lfp"]
    out["delta"] = {
        "peak_cell_temp_c": round(lfp["peak_cell_temp_c"] - nmc["peak_cell_temp_c"], 3),
        "min_exotherm_margin_c": round(
            lfp["min_exotherm_margin_c"] - nmc["min_exotherm_margin_c"], 2),
        "peak_commanded_flow_lpm": round(
            lfp["peak_commanded_flow_lpm"] - nmc["peak_commanded_flow_lpm"], 3),
        "total_ramp_lpm": round(lfp["total_ramp_lpm"] - nmc["total_ramp_lpm"], 3),
        "interpretation": (
            f"LFP's advantage is the exotherm onset gap, not a lower resistance: "
            f"its r_int at 40 degC ({lfp['r_int_at_40c_ohm']*1000:.2f} mohm) is "
            f"{'higher' if lfp['r_int_at_40c_ohm'] > nmc['r_int_at_40c_ohm'] else 'lower'} "
            f"than NMC's ({nmc['r_int_at_40c_ohm']*1000:.2f} mohm), so a greater share "
            f"of its ohmic load is recoverable heat. What buys the margin is onset "
            f"{lfp['exotherm_onset_c']:.0f} degC vs {nmc['exotherm_onset_c']:.0f} degC -- "
            f"{lfp['exotherm_onset_c'] - nmc['exotherm_onset_c']:.0f} K of extra thermal "
            f"headroom before self-heating becomes irreversible. The higher c_p "
            f"({lfp['cp_j_kgk']:.0f} vs {nmc['cp_j_kgk']:.0f} J/kgK) adds a further "
            f"{spec_energy_delta():.0f} Wh/kg energy-density penalty, which the "
            f"{abs(lfp['min_exotherm_margin_c'] - nmc['min_exotherm_margin_c']):.1f} K "
            f"of margin is what it buys back."
        ),
    }
    out["route"] = route
    out["weather"] = weather
    out["timestamp"] = utcnow_iso()
    return out


def spec_energy_delta() -> float:
    """NMC-minus-LFP gravimetric energy-density advantage, Wh/kg."""
    return (
        CHEMISTRY_REGISTRY["nmc"].energy_density_wh_kg
        - CHEMISTRY_REGISTRY["lfp"].energy_density_wh_kg
    )


@router.get("/ablation", summary="Loss-ablation study (physics vs data vs balance)")
def ablation(chemistry: str = DEFAULT_CHEMISTRY) -> dict:
    r"""Report the training-ablation table quoted in the model card.

    The reported figures come from the shipped ``artifacts/pinn_metrics.json``
    when present.  When it is absent (a fresh clone), the endpoint says so
    explicitly and returns the ablation *design* -- which terms are toggled, what
    each run measures -- rather than inventing numbers.
    """
    import json
    from pathlib import Path

    from ..config import ARTIFACT_DIR

    metrics_path = Path(ARTIFACT_DIR) / "pinn_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["source"] = str(metrics_path)
        payload["measured"] = True
        return payload

    return {
        "measured": False,
        "source": None,
        "note": (
            "artifacts/pinn_metrics.json not present. Run `python scripts/train_pinn.py` "
            "to produce the ablation table; this endpoint will then serve the measured "
            "numbers instead of this design description."
        ),
        "design": {
            "runs": [
                {"name": "full", "data": True, "physics": True, "adaptive_weights": True},
                {"name": "no-physics", "data": True, "physics": False, "adaptive_weights": False},
                {"name": "fixed-weights", "data": True, "physics": True, "adaptive_weights": False},
                {"name": "no-gradient-penalty", "data": True, "physics": True,
                 "adaptive_weights": False, "gradient_clip": 0.0},
                {"name": "physics-only", "data": False, "physics": True, "adaptive_weights": True},
            ],
            "metrics": ["rmse_k", "max_abs_error_k", "pde_residual_mean", "latency_ms_p95"],
            "note": (
                "The 'no-gradient-penalty' run disables grad_clip so the effective "
                "gradient-norm ratio driving lambda_i is unbounded; it is the run that "
                "exhibits loss imbalance blow-up."
            ),
        },
        "timestamp": utcnow_iso(),
    }