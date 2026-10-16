r"""
backend.routers.control
=======================

``/api/control`` -- coolant set-point commands with acknowledgement semantics.

The response contract here is deliberately shaped like a real actuator command
acknowledgement rather than a plain "ok": the client receives the *applied*
flow (post rate-limit), the settling time, and the heat-rejection capacity the
new set-point buys.  An operator can therefore tell the difference between
"your command was accepted" and "your command will take 6 seconds to take
effect", which matters when the next command depends on this one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..config import (
    COLD_PLATE,
    MAX_COOLANT_FLOW_LPM,
    MIN_COOLANT_FLOW_LPM,
    get_chemistry,
)
from ..models.physics import heat_transfer_coefficient, reynolds_from_flow
from ..schemas import (
    FlowCommandRequest,
    FlowCommandResponse,
    utcnow_iso,
)
from ..services.heatwave import RAMP_RATE_LPM_PER_S, SafetyState
from ..services.telemetry import HUB

router = APIRouter(prefix="/api/control", tags=["control"])

#: Pump response (first-order lag) time constant, seconds.
PUMP_TAU_S: float = 1.8
#: Buffer nominal heat capacity, J/K (see config LOOP_HEAT_CAPACITY_J_PER_K).
from ..config import LOOP_HEAT_CAPACITY_J_PER_K


@dataclass
class ActuatorState:
    """Process-wide actuator state (one microchannel loop per chemistry)."""

    applied_flow_lpm: Dict[str, float] = field(default_factory=lambda: {"nmc": 6.0, "lfp": 6.0})
    mode: Dict[str, str] = field(default_factory=lambda: {"nmc": "auto", "lfp": "auto"})
    pending_flow_lpm: Dict[str, float] = field(default_factory=lambda: {"nmc": 6.0, "lfp": 6.0})
    ramp_started_s: Dict[str, float] = field(default_factory=dict)
    safety: Dict[str, SafetyState] = field(default_factory=dict)
    history: List[Dict[str, object]] = field(default_factory=list)
    last_operator: Dict[str, str] = field(default_factory=dict)

    def safety_for(self, chemistry: str) -> SafetyState:
        """Get-or-create the fail-safe supervisor for a chemistry."""
        if chemistry not in self.safety:
            self.safety[chemistry] = SafetyState()
        return self.safety[chemistry]

    def record(self, entry: Dict[str, object]) -> None:
        """Append to the command audit trail (capped)."""
        self.history.append(entry)
        if len(self.history) > 500:
            self.history.pop(0)


ACTUATOR = ActuatorState()


def _settle_time(from_lpm: float, to_lpm: float) -> float:
    """First-order settling time (2% band) of the pump, seconds."""
    delta = abs(to_lpm - from_lpm)
    if delta < 1e-6:
        return 0.0
    return float(PUMP_TAU_S * math.log(50.0))


@router.post("/flow", response_model=FlowCommandResponse,
             summary="Command the coolant mass-flow set-point")
def set_flow(req: FlowCommandRequest) -> FlowCommandResponse:
    """Apply a coolant set-point command, honouring the pump rate limit.

    A latched fail-safe trip **overrides** any operator command: the request is
    accepted for bookkeeping but the applied flow is forced to the maximum and
    the response carries ``trip_state="trip"`` with an explanatory message.  This
    is the "no operator can mask a thermal event" invariant.
    """
    spec = get_chemistry(req.chemistry)
    safety = ACTUATOR.safety_for(spec.key)

    applied = float(np.clip(req.flow_lpm, MIN_COOLANT_FLOW_LPM, MAX_COOLANT_FLOW_LPM))
    previous = ACTUATOR.applied_flow_lpm.get(spec.key, 6.0)
    trip = safety.latched_trip
    if trip:
        applied = MAX_COOLANT_FLOW_LPM
        mode = "failsafe"
    else:
        mode = req.mode

    ACTUATOR.applied_flow_lpm[spec.key] = applied
    ACTUATOR.pending_flow_lpm[spec.key] = applied
    ACTUATOR.mode[spec.key] = mode
    ACTUATOR.last_operator[spec.key] = req.operator_id
    ACTUATOR.ramp_started_s[spec.key] = time.time()

    h = heat_transfer_coefficient(applied, spec.coolant_props)
    capacity = h * COLD_PLATE.conductance_m2k * 10.0   # at dT = 10 K

    ACTUATOR.record({
        "t": utcnow_iso(), "chemistry": spec.key, "mode": mode,
        "requested_lpm": req.flow_lpm, "applied_lpm": applied,
        "previous_lpm": previous, "operator_id": req.operator_id,
        "trip_override": trip,
    })

    # Mirror the command into the live simulator so the WebSocket feed reflects it.
    sim = HUB.simulators.get(spec.key)
    if sim is not None:
        sim.state.commanded_flow_lpm = applied

    if trip:
        message = (
            f"Command acknowledged but overridden by a latched fail-safe trip: "
            f"flow forced to {MAX_COOLANT_FLOW_LPM:.1f} L/min. "
            f"Reason: {safety.reason}. An operator reset is required to clear."
        )
    elif abs(applied - req.flow_lpm) > 1e-6:
        message = (
            f"Set-point clamped from {req.flow_lpm:.2f} to {applied:.2f} L/min "
            f"(loop limits {MIN_COOLANT_FLOW_LPM}-{MAX_COOLANT_FLOW_LPM})."
        )
    else:
        message = (
            f"Coolant set-point {applied:.2f} L/min applied in {mode} mode; "
            f"Re ~ {reynolds_from_flow(applied, spec.coolant_props):.0f}, "
            f"h ~ {h:.0f} W/(m^2 K)."
        )

    return FlowCommandResponse(
        accepted=True,
        applied_flow_lpm=round(applied, 3),
        mode=mode,
        ramp_rate_lpm_per_s=RAMP_RATE_LPM_PER_S,
        estimated_settle_s=round(_settle_time(previous, applied), 2),
        heat_rejection_w=round(capacity, 1),
        trip_state=safety.tier,
        message=message,
        timestamp=utcnow_iso(),
    )


@router.get("/state", summary="Current actuator and fail-safe state")
def state() -> dict:
    """Report the actuator state for every chemistry plus the audit trail."""
    out: Dict[str, object] = {"chemistries": {}, "timestamp": utcnow_iso()}
    for key, spec in ((k, get_chemistry(k)) for k in ("nmc", "lfp")):
        flow = ACTUATOR.applied_flow_lpm.get(key, 6.0)
        safety = ACTUATOR.safety_for(key)
        h = heat_transfer_coefficient(flow, spec.coolant_props)
        out["chemistries"][key] = {
            "applied_flow_lpm": flow,
            "mode": ACTUATOR.mode.get(key, "auto"),
            "reynolds": round(reynolds_from_flow(flow, spec.coolant_props), 1),
            "h_cold_w_m2k": round(h, 1),
            "heat_rejection_capacity_w": round(h * COLD_PLATE.conductance_m2k * 10.0, 1),
            "trip_state": safety.tier,
            "trip_reason": safety.reason,
            "derate_factor": safety.derate_factor,
            "last_operator": ACTUATOR.last_operator.get(key, "-"),
        }
    out["recent_commands"] = ACTUATOR.history[-25:]
    out["loop_limits"] = {"min_lpm": MIN_COOLANT_FLOW_LPM,
                          "max_lpm": MAX_COOLANT_FLOW_LPM,
                          "ramp_rate_lpm_per_s": RAMP_RATE_LPM_PER_S,
                          "pump_tau_s": PUMP_TAU_S}
    return out


@router.post("/safety/reset", summary="Operator reset of a latched fail-safe trip")
def safety_reset(chemistry: str = "nmc") -> dict:
    """Clear a latched trip after the pack has been verified safe.

    In a production vehicle this would require the pack to be below the
    chemistry's comfort ceiling *and* an ignition interlock; the API documents
    the precondition rather than enforcing a physical interlock.
    """
    spec = get_chemistry(chemistry)
    safety = ACTUATOR.safety_for(spec.key)
    was = safety.tier
    safety.reset()
    ACTUATOR.record({"t": utcnow_iso(), "chemistry": spec.key, "action": "safety_reset",
                     "previous_state": was})
    return {
        "chemistry": spec.key,
        "previous_state": was,
        "trip_state": safety.tier,
        "reason": safety.reason,
        "precondition": (
            f"Operator asserts the pack is below {spec.runaway_c - 60:.0f} degC and "
            "the coolant loop is verified. Production HIL must gate this on a "
            "physical interlock."
        ),
        "timestamp": utcnow_iso(),
    }


@router.get("/history", summary="Command audit trail")
def history(limit: int = 50) -> dict:
    """Return the most recent coolant commands (capped at 500)."""
    n = max(1, min(limit, 500))
    return {"commands": ACTUATOR.history[-n:], "n_total": len(ACTUATOR.history),
            "timestamp": utcnow_iso()}