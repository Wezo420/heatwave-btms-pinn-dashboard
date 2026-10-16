r"""
backend.services.telemetry
==========================

Closed-loop telemetry simulator and the WebSocket broadcast hub.

The simulator advances the reduced 12-cell pack model under the anticipatory
controller, so the dashboard's live graphs are the output of an actual
closed-loop integration rather than a replayed fixture.  At each tick it:

1.  advances the pack state with the currently commanded flow;
2.  evaluates the PINN to obtain the 3-D field (optionally throttled), and blends
    the PINN's *shape* correction onto the reduced model's *magnitude* -- a
    cheap, explicit physics/model data fusion (see :meth:`TelemetrySimulator.tick`);
3.  evaluates the fail-safe supervisor;
4.  publishes a :class:`backend.schemas.TelemetryFrame` to all subscribers.

The blending in step 2 is the interesting part.  The reduced model is accurate
in the bulk mean (it carries the calibrated ``R_int``, the coolant ``\dot m c_p``
and the convective conductance) but cannot resolve the 3-D structure; the PINN is
structurally rich but its absolute level is uncertain at deployment.  The fusion

.. math::  \tilde T = \bar T_\text{reduced} + \big(\hat T_\text{PINN} - \overline{\hat T}_\text{PINN}\big) \cdot \gamma

keeps the reduced model's calibrated absolute level and imports only the PINN's
*deviation field*, with ``gamma`` the blend factor (0.55 by default).  This is
the standard "model output correction" form of physics-model data assimilation
and it means a PINN extrapolation can never shift the safety-relevant absolute
temperature on its own -- only its spatial pattern.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..config import (
    CHECKPOINT_PATH,
    COLD_PLATE,
    DEFAULT_CHEMISTRY,
    MAX_COOLANT_FLOW_LPM,
    MIN_COOLANT_FLOW_LPM,
    ChemistrySpec,
    get_chemistry,
)
from ..models.physics import (
    heat_transfer_coefficient,
    lumped_pack_rhs,
    required_flow_for_target,
)
from ..schemas import TelemetryFrame, utcnow_iso
from .heatwave import (
    COMFORT_CEILING_C,
    MARGIN_FLOOR_C,
    RAMP_RATE_LPM_PER_S,
    HeatwavePlanner,
    ReducedPackModel,
    SafetyState,
    effective_ambient_series,
    solar_overshoot,
)
from .inference import InferenceEngine

#: Blend weight for the PINN deviation field (see module docstring).
DEFAULT_BLEND_GAMMA = 0.55
#: Simulated tick period for the WebSocket feed, seconds.
DEFAULT_TICK_S = 0.5


@dataclass(slots=True)
class SimState:
    """Full internal state of one simulated vehicle."""

    chemistry: str
    cell_temp_c: np.ndarray
    soc: float
    coolant_inlet_c: float
    coolant_flow_lpm: float
    commanded_flow_lpm: float
    pack_current_a: float
    ambient_c: float
    road_surface_c: float
    solar_w_m2: float
    wind_mps: float
    lat: float
    lon: float
    speed_mps: float
    t_s: float
    initial_soc: float
    journey_frac_soc: float = 0.0
    trip: str = "ok"
    anticipatory_active: bool = False
    trigger: str = "idle"


class TelemetrySimulator:
    """Deterministic (seeded) closed-loop pack/coolant simulator.

    Parameters
    ----------
    chemistry:
        Starting chemistry.  The dashboard toggle switches this and *preserves*
        the state, so the user can flip NMC <-> LFP mid-stream and watch the
        controller re-plan -- which is the point of the dual-chemistry study.
    """

    def __init__(
        self,
        chemistry: str = DEFAULT_CHEMISTRY,
        initial_temp_c: float = 32.0,
        initial_soc: float = 0.65,
        coolant_inlet_c: float = 26.0,
        seed: int = 1337,
        blend_gamma: float = DEFAULT_BLEND_GAMMA,
        engine: Optional[InferenceEngine] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.spec: ChemistrySpec = get_chemistry(chemistry)
        self.chemistry = chemistry
        self.model = ReducedPackModel(COLD_PLATE.n_cells)
        self.blend_gamma = float(blend_gamma)
        self.safety = SafetyState()
        self.planner = HeatwavePlanner(chemistry)
        self.state = SimState(
            chemistry=chemistry,
            cell_temp_c=np.full(COLD_PLATE.n_cells, float(initial_temp_c)),
            soc=float(initial_soc),
            initial_soc=float(initial_soc),
            coolant_inlet_c=float(coolant_inlet_c),
            coolant_flow_lpm=6.0,
            commanded_flow_lpm=6.0,
            pack_current_a=140.0,
            ambient_c=34.0,
            road_surface_c=48.0,
            solar_w_m2=850.0,
            wind_mps=2.5,
            lat=19.0760,   # Mumbai -- default demo route
            lon=72.8777,
            speed_mps=22.0,
            t_s=0.0,
        )
        self._history: List[Dict[str, Any]] = []
        self._engine: Optional[InferenceEngine] = engine

    # -- accessors ---------------------------------------------------------
    @property
    def engine(self) -> InferenceEngine:
        """Lazily construct the inference engine (deferred until first use)."""
        if self._engine is None:
            self._engine = InferenceEngine.instance()
        return self._engine

    def set_chemistry(self, chemistry: str) -> SimState:
        """Switch chemistry while preserving pack state (dashboard toggle)."""
        self.spec = get_chemistry(chemistry)
        self.chemistry = chemistry
        self.state.chemistry = chemistry
        self.planner = HeatwavePlanner(chemistry)
        self.safety = SafetyState()
        return self.state

    def reset(self, **overrides: float) -> SimState:
        """Reset to the initial condition, applying ``overrides``."""
        temp = overrides.get("initial_temp_c", 32.0)
        soc = overrides.get("initial_soc", 0.65)
        self.state = SimState(
            chemistry=self.chemistry,
            cell_temp_c=np.full(COLD_PLATE.n_cells, float(temp)),
            soc=float(soc),
            initial_soc=float(soc),
            coolant_inlet_c=float(overrides.get("coolant_inlet_c", 26.0)),
            coolant_flow_lpm=float(overrides.get("base_flow_lpm", 6.0)),
            commanded_flow_lpm=float(overrides.get("base_flow_lpm", 6.0)),
            pack_current_a=float(overrides.get("pack_current_a", 140.0)),
            ambient_c=float(overrides.get("ambient_c", 34.0)),
            road_surface_c=float(overrides.get("road_surface_c", 48.0)),
            solar_w_m2=float(overrides.get("solar_w_m2", 850.0)),
            wind_mps=2.5,
            lat=float(overrides.get("lat", 19.0760)),
            lon=float(overrides.get("lon", 72.8777)),
            speed_mps=float(overrides.get("speed_mps", 22.0)),
            t_s=0.0,
        )
        self.safety = SafetyState()
        self._history.clear()
        return self.state

    # -- controller --------------------------------------------------------
    def _anticipatory_command(
        self,
        dt_s: float,
        ambient_forecast_c: float,
    ) -> Tuple[float, bool, str]:
        r"""One receding-horizon step of the anticipatory controller.

        The preview comes from the weather service's own short-range forecast
        (mocked here as ``ambient + ramp``); the planner converts it into the
        flow required to keep the cell *on* its trajectory rather than merely
        below the ceiling.
        """
        s = self.state
        spec = self.spec
        ceiling = COMFORT_CEILING_C[spec.key]
        margin_floor = MARGIN_FLOOR_C[spec.key]

        # If the trip is latched, the fail-safe owns the actuator.
        if self.safety.latched_trip:
            return MAX_COOLANT_FLOW_LPM, True, "failsafe-trip"

        # Preview ambient at t + lead, from the mock weather service.
        preview = ambient_forecast_c + max(0.0, ambient_forecast_c - s.ambient_c) * 0.8
        effective = preview + solar_overshoot(
            {"solar_w_m2": s.solar_w_m2}, spec
        )

        if effective > ceiling - 6.0 or s.cell_temp_c.max() > ceiling - 3.0:
            required = required_flow_for_target(
                spec, float(s.cell_temp_c.max()), ceiling, s.coolant_inlet_c,
                s.pack_current_a, s.soc, effective, s.solar_w_m2,
            )
            # Anticipatory bias: act before the temperature moves.
            required = required + 0.9
            return min(MAX_COOLANT_FLOW_LPM, max(required, s.commanded_flow_lpm)), \
                True, "anticipatory-heatwave"

        # Nominal set-point-tracking with a deadband.
        if s.cell_temp_c.max() > ceiling:
            required = required_flow_for_target(
                spec, float(s.cell_temp_c.max()), ceiling, s.coolant_inlet_c,
                s.pack_current_a, s.soc, effective, s.solar_w_m2,
            )
            return min(MAX_COOLANT_FLOW_LPM, required), True, "over-temperature"

        return 6.0, False, "nominal-setpoint"

    # -- main step ---------------------------------------------------------
    def tick(self, dt_s: float = DEFAULT_TICK_S, use_pinn: bool = True) -> TelemetryFrame:
        """Advance the closed loop by ``dt_s`` and return the frame."""
        s = self.state
        spec = self.spec

        # -- environment (mock GPS + weather drift) ----------------------
        s.t_s += dt_s
        s.lat += 0.00012 * dt_s * s.speed_mps / 22.0
        s.lon += 0.00016 * dt_s * s.speed_mps / 22.0
        s.solar_w_m2 = float(np.clip(
            880.0 + 260.0 * math.sin(s.t_s / 47.0) + self.np_rng.normal(0, 18.0), 0, 1350))
        s.ambient_c = float(np.clip(
            36.0 + 6.5 * math.sin(s.t_s / 210.0) + self.np_rng.normal(0, 0.25),
            15.0, 56.0))
        s.road_surface_c = s.ambient_c + 14.0 + 6.0 * math.sin(s.t_s / 90.0)
        s.wind_mps = float(np.clip(2.4 + 1.4 * abs(math.sin(s.t_s / 130.0)), 0.3, 9.0))
        s.pack_current_a = float(np.clip(
            140.0 * (s.speed_mps / 22.0) * (0.8 + 0.4 * (1.0 - s.soc)), 20.0, 620.0))

        # -- controller ----------------------------------------------------
        preview_ambient = float(np.clip(
            s.ambient_c + 3.2 * math.sin((s.t_s + 150.0) / 210.0), 15.0, 60.0))
        target_flow, anticipatory, trigger = self._anticipatory_command(
            dt_s, preview_ambient
        )

        # -- fail-safe supervisor -----------------------------------------
        peak_now = float(s.cell_temp_c.max())
        self.safety.evaluate(peak_now, spec)
        if self.safety.tier == "derate":
            target_flow = MAX_COOLANT_FLOW_LPM
            trigger = f"derate:{self.safety.reason.split(';')[0]}"
        s.trip = self.safety.tier

        if self.safety.derate_factor < 1.0:
            s.pack_current_a *= self.safety.derate_factor

        # -- actuator rate limit ------------------------------------------
        max_step = RAMP_RATE_LPM_PER_S * dt_s
        delta = float(np.clip(target_flow - s.commanded_flow_lpm, -max_step, max_step))
        s.commanded_flow_lpm = float(np.clip(
            s.commanded_flow_lpm + delta, MIN_COOLANT_FLOW_LPM, MAX_COOLANT_FLOW_LPM))
        s.coolant_flow_lpm = s.commanded_flow_lpm
        s.anticipatory_active = anticipatory or self.safety.tier != "ok"
        s.trigger = trigger

        # -- plant integration ---------------------------------------------
        s.cell_temp_c = self.model.step(
            s.cell_temp_c, dt_s,
            s.ambient_c, 0.85 * s.solar_w_m2, s.road_surface_c,
            s.pack_current_a, s.soc, s.coolant_flow_lpm, s.coolant_inlet_c, spec,
        )

        # -- PINN field fusion (deviation correction) ----------------------
        blend = self.blend_gamma
        if use_pinn and blend > 0:
            blend = self._blend_field(blend)

        # -- derived quantities --------------------------------------------
        from ..models.physics import heat_transfer_coefficient, nusselt_microchannel

        coolant = spec.coolant_props
        h = heat_transfer_coefficient(s.coolant_flow_lpm, coolant)
        mcp = coolant["density"] * (s.coolant_flow_lpm / 60000.0) * coolant["cp"]
        q_joule = float(s.pack_current_a**2 * spec.r_int(float(s.cell_temp_c.max()),
                                                           s.soc))
        q_rej = h * COLD_PLATE.conductance_m2k * max(
            float(s.cell_temp_c.max()) - s.coolant_inlet_c, 0.0)
        coolant_outlet = s.coolant_inlet_c + q_rej / max(mcp, 1e-6)
        # The channel warms the inlet over the pack length.
        s.coolant_inlet_c = 0.86 * s.coolant_inlet_c + 0.14 * coolant_outlet

        # -- SOC decay -------------------------------------------------------
        pack_energy_wh = spec.nominal_capacity_ah * s.soc * (
            spec.energy_density_wh_kg / max(spec.density / 1000.0, 1e-6)
        ) * (COLD_PLATE.n_cells / 12.0)
        s.journey_frac_soc += (s.pack_current_a * dt_s / 3600.0) / max(
            s.initial_soc * spec.nominal_capacity_ah * (COLD_PLATE.n_cells / 12.0), 1e-3)
        s.soc = float(np.clip(s.soc - (s.pack_current_a * dt_s / 3600.0) / max(
            spec.nominal_capacity_ah * (COLD_PLATE.n_cells / 12.0), 1e-3), 0.02, 1.0))

        peak = float(s.cell_temp_c.max())
        frame = TelemetryFrame(
            t_s=round(s.t_s, 3),
            chemistry=spec.key,
            cell_temps_c=[round(float(v), 3) for v in s.cell_temp_c],
            max_cell_temp_c=round(peak, 3),
            mean_cell_temp_c=round(float(s.cell_temp_c.mean()), 3),
            coolant_flow_lpm=round(s.coolant_flow_lpm, 3),
            commanded_flow_lpm=round(s.commanded_flow_lpm, 3),
            coolant_inlet_c=round(s.coolant_inlet_c, 3),
            coolant_outlet_c=round(coolant_outlet, 3),
            ambient_c=round(s.ambient_c, 3),
            road_surface_c=round(s.road_surface_c, 3),
            joule_heat_w=round(q_joule, 2),
            heat_rejected_w=round(q_rej, 2),
            soc=round(s.soc, 5),
            state_of_charge_pct=round(100.0 * s.soc, 2),
            thermal_runaway_margin_c=round(spec.runaway_c - peak, 3),
            risk_index=round(self._risk(peak), 4),
            lat=round(s.lat, 6),
            lon=round(s.lon, 6),
            speed_mps=round(s.speed_mps, 2),
            anticipatory_active=s.anticipatory_active,
            trip_state=self.safety.tier,
        )
        self._history.append(frame.model_dump())
        if len(self._history) > 4000:
            self._history.pop(0)
        return frame

    def _blend_field(self, gamma: float) -> float:
        r"""Fuse the PINN's *spatial shape* into the reduced model's level.

        The fusion is explicit and one-directional:

        .. math::
           T_\text{fused} = \bar T_\text{reduced}
                          + \gamma\,\sigma_T\bigl[\hat T_\text{PINN}(x)\bigr]
                          + \gamma\,\varepsilon\, p(x)

        The reduced model carries the calibrated absolute level (``R_int`` from
        the NASA fit, the convective conductance, the coolant rise); the PINN
        supplies only its mean-removed deviation field.  That ordering is the
        safety property: a PINN extrapolation can reshape the profile but can
        never move the absolute temperature, so it cannot talk the controller
        out of a thermal event.

        The deviation is reduced to one value per zone by sampling the PINN on a
        ``(nx, ny)`` line at mid-thickness and averaging over ``y`` -- giving
        ``nx`` streamwise values that are then interpolated onto the 12 zones.
        An earlier version reshaped the flat collocation vector with a tuple
        index, which silently indexed a *tuple* rather than an array and
        contributed nothing; the reduction is now explicit and its shape is
        asserted.
        """
        s = self.state
        nx = 9
        try:
            xs = torch.linspace(0.0, COLD_PLATE.length_m, nx, dtype=torch.float32)
            y_mid = torch.full((nx,), 0.5 * COLD_PLATE.width_m, dtype=torch.float32)
            z_mid = torch.full((nx,), 0.5 * COLD_PLATE.thickness_m, dtype=torch.float32)
            pts = torch.stack([xs, y_mid, z_mid], dim=-1)

            t_k = self.engine.model.temperature(pts.to(self.engine.device), s.chemistry)
            deviation = (t_k - t_k.mean()).detach().cpu().numpy()
            assert deviation.shape == (nx,), deviation.shape

            # Streamwise pattern: the inlet end is hottest (the coolant is
            # coldest there) and the outlet end coolest.
            pattern = (
                0.6 * np.cos(np.pi * xs.numpy() / COLD_PLATE.length_m)
                + 0.4 * np.linspace(1.0, 0.2, self.model.n)
            )

            # Map the nx streamwise samples onto the n zones by interpolation.
            x_zone = np.linspace(0.0, COLD_PLATE.length_m, self.model.n)
            dev_zone = np.interp(x_zone, xs.numpy(), deviation)

            s.cell_temp_c = s.cell_temp_c + gamma * dev_zone * 0.35 \
                + gamma * 0.05 * pattern
        except Exception as exc:
            # The simulator must never die on a model hiccup -- the auto-heal
            # monitor owns recovery, and a degraded field is preferable to a
            # dead stream.  One warning, then the fusion is disabled for good so
            # the hot loop does not re-raise on every tick.
            if self.blend_gamma > 0.0:
                warnings.warn(
                    f"PINN field fusion disabled after an engine error: {exc!r}. "
                    f"The telemetry stream continues on the reduced model alone.",
                    RuntimeWarning, stacklevel=2,
                )
            self.blend_gamma = 0.0
        return gamma

    def _risk(self, peak_cell_c: float) -> float:
        ceiling = COMFORT_CEILING_C[self.chemistry]
        util = peak_cell_c / ceiling
        base = 1.0 / (1.0 + math.exp(-6.0 * (util - 0.82)))
        if self.safety.tier == "derate":
            base = max(base, 0.85)
        elif self.safety.tier == "trip":
            base = 1.0
        if self.state.anticipatory_active:
            base = min(1.0, base + 0.08)
        return float(np.clip(base, 0.0, 1.0))

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Recent frames as plain dicts."""
        return list(self._history)


# ---------------------------------------------------------------------------
# WebSocket hub
# ---------------------------------------------------------------------------


class TelemetryHub:
    """Fan-out hub broadcasting :class:`TelemetryFrame` to WebSocket clients.

    Each client may select its own chemistry and tick rate.  The hub owns one
    simulator per chemistry so that two browsers on NMC and LFP do not share a
    plant, and a single asyncio task drives every simulator at the shortest
    requested period (a tick that nobody needs is not simulated).
    """

    def __init__(self, tick_s: float = DEFAULT_TICK_S) -> None:
        self.tick_s = float(tick_s)
        self.simulators: Dict[str, TelemetrySimulator] = {}
        self._subscribers: Dict[str, Set[Any]] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.frames_sent = 0

    def simulator(self, chemistry: str = DEFAULT_CHEMISTRY) -> TelemetrySimulator:
        """Get-or-create the simulator for ``chemistry``."""
        key = get_chemistry(chemistry).key
        if key not in self.simulators:
            self.simulators[key] = TelemetrySimulator(chemistry=key)
        return self.simulators[key]

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Start the broadcast loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="telemetry-hub")

    async def stop(self) -> None:
        """Stop the broadcast loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            started = time.perf_counter()
            try:
                for key, sim in list(self.simulators.items()):
                    subs = self._subscribers.get(key)
                    if not subs:
                        continue
                    frame = sim.tick(self.tick_s)
                    self.frames_sent += 1
                    payload = json.dumps({"type": "telemetry", "data": frame.model_dump()})
                    dead = []
                    for ws in list(subs):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        subs.discard(ws)
            except Exception as exc:  # pragma: no cover - keep the hub alive
                print(f"[telemetry-hub] tick error: {exc!r}")
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.02, self.tick_s - elapsed))

    # -- pub/sub -----------------------------------------------------------
    def subscribe(self, chemistry: str, ws: Any) -> None:
        """Register ``ws`` for frames of ``chemistry``."""
        key = get_chemistry(chemistry).key
        self.simulator(key)
        self._subscribers.setdefault(key, set()).add(ws)

    def unsubscribe(self, chemistry: str, ws: Any) -> None:
        """Deregister ``ws``."""
        key = get_chemistry(chemistry).key
        self._subscribers.get(key, set()).discard(ws)

    def subscriber_count(self) -> int:
        """Total connected clients."""
        return sum(len(v) for v in self._subscribers.values())


#: Module-level hub shared by the routers.
HUB = TelemetryHub()