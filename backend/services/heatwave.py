r"""
backend.services.heatwave
========================

Anticipatory heatwave planning: the logic that ramps the microchannel coolant
**before** the vehicle reaches a thermally hostile region.

Why anticipation beats reaction
-------------------------------
A cold plate with 2.2 mm channels has a coolant transit time of

.. math::  \tau_\text{loop} = \frac{V_\text{loop}}{\dot V}
                 \approx \frac{0.34\ \text{L}}{6\ \text{L/min}} \approx 3.4\ \text{s}

while the pack's thermal time constant is

.. math::  \tau_\text{pack} = \frac{C_\theta}{\dot U}
                 = \frac{12 \times 118\ \text{J/K}}{1.1\ \text{W/K}} \approx 1300\ \text{s}

so the loop can slew the flow rate *fifty times faster* than the pack responds.
The optimum is therefore not "react when T rises" but **"ramp so the pack's
approach to the equilibrium trajectory is tangent to its comfort limit"** -- a
one-sided preview-constraint problem, solved here by a receding-horizon
schedule over the known route.

Method
------
1.  **Route -> thermal forecast.**  For every waypoint the ambient is drawn from
    the weather series, inflated by (a) solar irradiance on the pack shell and
    (b) road-surface heating of the lower face, then a first-order lag
    (time constant ``T_amb_tau = 420 s``, fitted to cabin measurements) gives the
    effective ambient the pack actually sees at time ``t``.
2.  **Feasibility sweep.**  For a grid of candidate flow rates the forward
    lumped model (:func:`backend.models.physics.lumped_pack_rhs`) is integrated
    over the route.  For each candidate we record the peak cell temperature and
    the minimum runaway margin.
3.  **Constraint selection.**  The smallest flow rate whose peak cell temperature
    respects the chemistry's comfort ceiling **and** preserves at least
    ``MARGIN_FLOOR`` kelvin of exotherm onset margin is selected.  This is a
    one-dimensional constraint search, so the result is the provably minimal
    actuation -- the controller never over-cools.
4.  **Ramp shaping.**  The selected envelope is rate-limited to
    ``RAMP_RATE_LPM_PER_S`` and aligned to a ``RAMP_LEAD_S`` early start so the
    flow is already at its target when the vehicle enters the zone.
5.  **Risk index.**  A dimensionless score in ``[0, 1]`` from peak projected
    temperature, rate of change, and lateral (cell-to-cell) spread; it drives the
    dashboard overlay and the fail-safe tier.

Every step is explainable: :meth:`HeatwavePlanner.explain` returns the deciding
terms for a given zone, which the API exposes verbatim so the UI can show *why*
the controller acted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (
    CELL_GEOMETRY,
    CELL_TEMPERATURE_CEILING_C,
    COLD_PLATE,
    DEFAULT_AUX_HEAT_W,
    DEFAULT_PACK_CURRENT_A,
    EMISSIVITY,
    SIGMA_SB,
    COOLANT_SETPOINT_C,
    HEATWAVE_LOOKAHEAD_S,
    HEATWAVE_RAMP_LEAD_S,
    MAX_COOLANT_FLOW_LPM,
    MAX_PACK_CURRENT_A,
    MIN_COOLANT_FLOW_LPM,
    PACK_PARALLEL_STRINGS,
    ZONE_PARALLEL_STRINGS,
    ZONE_SERIES_CELLS,
    ChemistrySpec,
    get_chemistry,
    kelvin,
)
from ..models.physics import (
    lumped_pack_rhs,
    reynolds_from_flow,
    required_flow_for_target,
    thermal_runaway_margin,
)

#: Comfort ceiling per chemistry (degC).  NMC gets a tighter cell because of its
#: lower exotherm onset; LFP can run cooler for the same safety factor.
COMFORT_CEILING_C: Dict[str, float] = {
    "nmc": CELL_TEMPERATURE_CEILING_C - 3.0,   # 42 degC
    "lfp": CELL_TEMPERATURE_CEILING_C + 2.0,   # 47 degC
}

#: Re-exported for callers of the planner; the canonical definition (and its
#: rationale) lives in :mod:`backend.config` so the API layer and the controller
#: cannot disagree about the demo scenario.
DEFAULT_AUX_HEAT_W = DEFAULT_AUX_HEAT_W

#: Iterations of the search/shape/ramp fixed-point loop in :meth:`HeatwavePlanner.plan`.
#: Each pass measures the thermal cost of rate-limiting the schedule and asks the
#: constant-flow search for that much additional headroom.  Three passes are
#: enough for the observed overshoot to converge to zero; the loop also exits
#: early once the ceiling is satisfied.
_SEARCH_REFINE_ITERS: int = 4
#: Cap on the headroom the loop will ask for, degC.  Beyond this the load is
#: simply unmanageable at maximum flow, and further iterations cannot help.
_SEARCH_MAX_MARGIN_C: float = 25.0

#: Required clearance below the adiabatic exotherm onset, K.
MARGIN_FLOOR_C: Dict[str, float] = {"nmc": 45.0, "lfp": 55.0}

#: Maximum commanded ramp rate, L/min per second (pump + inverter limits).
RAMP_RATE_LPM_PER_S: float = 0.22
#: Pump flow ripple tolerance used by the deadband controller, L/min.
FLOW_DEADBAND_LPM: float = 0.35
#: First-order lag of the effective ambient the pack sees, seconds.
AMBIENT_TAU_S: float = 420.0

#: Exposed shell area of one thermal zone, m^2 -- the fraction of the zone's
#: cell envelope that sees ambient air rather than being bonded to the cold
#: plate.
ZONE_SHELL_AREA_M2: float = 0.0056

#: Thermal capacitance of one thermal zone, J/K.  A zone holds
#: :data:`~backend.config.ZONE_SERIES_CELLS` prismatic cells (8 in the 96s1p
#: pack), so it is ~8x a single bare cell's ~365 J/K.  Getting this wrong
#: compresses every transient by the same factor and destroys the anticipatory
#: lead time the controller exists to exploit -- with a small capacitance the
#: pack reaches equilibrium in seconds and there is nothing to anticipate.
#: Derived from the chemistry inside :class:`ReducedPackModel`; this constant is
#: only the fallback used when no chemistry has been supplied.
CELL_C_TH_J_PER_K: float = 2_950.0

#: Thermal coupling between adjacent zones, W/K, as a 1-D ring coupling.
#: Sized with the zone thermal mass so the lateral spreading time constant
#: ``C_th / G`` stays a few minutes -- a plausible timescale for conduction
#: through a bonded interface, and slow enough that a blocked channel shows up
#: as a developing gradient rather than a step.
CELL_COUPLING_J_PER_K: float = 45.0
#: Conductive coupling between a zone and the coolant channel plane, W/K.
CELL_TO_CHANNEL_J_PER_K: float = 0.85


@dataclass(slots=True)
class ZoneForecast:
    """One geofenced thermally-hostile segment on the route."""

    zone_id: str
    t_enter_s: float
    t_exit_s: float
    peak_ambient_c: float
    peak_solar_w_m2: float
    lat_centre: float
    lon_centre: float
    risk_index: float
    predicted_peak_cell_c: float
    predicted_margin_c: float
    required_flow_lpm: float
    baseline_flow_lpm: float
    dominant_mode: str
    reason: str


@dataclass(slots=True)
class Actuation:
    """A commanded coolant set-point change."""

    t_s: float
    command_lpm: float
    delta_lpm: float
    lead_time_s: float
    trigger: str
    confidence: float
    authority: str = "enforced"


@dataclass(slots=True)
class Plan:
    """Complete anticipatory schedule for a route."""

    chemistry: str
    time_s: np.ndarray
    baseline_flow_lpm: np.ndarray
    commanded_flow_lpm: np.ndarray
    predicted_cell_c: np.ndarray
    predicted_margin_c: np.ndarray
    risk_index: np.ndarray
    zones: List[ZoneForecast]
    actuation: List[Actuation]
    peak_cell_c: float
    min_margin_c: float
    total_ramp_distance_lpm: float
    advisory: str
    decision_log: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Weather -> effective ambient
# ---------------------------------------------------------------------------


def effective_ambient_series(
    weather: Sequence[Dict[str, float]],
    dt_s: float,
) -> np.ndarray:
    r"""Lag-filter the ambient to what the *pack* experiences.

    The first-order lag models the thermal mass of the cabin/pack enclosure:
    ``T_eff(t) = T_amb(t - tau)``.  Implemented as an exact discrete step response
    so that it is stable for any ``dt``:

    .. math::
       T_{\text{eff}}^{n+1} = T_{\text{eff}}^n e^{-\Delta t/\tau}
                              + T_{\text{amb}}^{n+1}\big(1 - e^{-\Delta t/\tau}\big)
    """
    tau = AMBIENT_TAU_S
    alpha = math.exp(-dt_s / tau)
    t_eff = np.empty(len(weather), dtype=np.float64)
    acc = weather[0]["ambient_c"]
    for i, w in enumerate(weather):
        acc = acc * alpha + w["ambient_c"] * (1.0 - alpha)
        t_eff[i] = acc
    return t_eff


def solar_overshoot(w: Dict[str, float], spec: ChemistrySpec) -> float:
    r"""Pack temperature overshoot above ambient from solar heating, K.

    **The pack is underfloor, so it does not see the sky.**  A traction battery
    sits in a sealed tray beneath the cabin, shielded by the vehicle floor and
    by the cabin itself.  Direct irradiance reaches it only through the
    transmitted fraction of the roof/floor structure and via the road surface it
    sits above.  Applying a dark-absorber coefficient (``alpha ~ 0.85``) to the
    pack shell would overstate the solar path by an order of magnitude -- it is
    the *cabin* that runs 15--20 K above ambient in full sun, and the pack
    follows the cabin air, not the sky.

    The steady-state relation is therefore applied to the transmitted
    irradiance with a small effective absorptance:

    .. math::
       \Delta T_\odot = \frac{\alpha_\text{eff} G_\text{solar}}
                             {h_\text{conv} + h_\text{rad}},
       \qquad \alpha_\text{eff} = \tau_\text{floor}\,\alpha_\text{shell} \approx 0.10

    giving 1--3 K at ``G = 1000 W/m^2``.  The dominant solar heating of the pack
    arrives indirectly, through the road surface, and is captured by the
    ``road_surface_c`` channel rather than by this term.

    The structural point: a steady-state overshoot is *intensity over
    conductance* and cannot depend on thermal mass.  An implementation that
    multiplies by ``m c_p / (h A)`` silently converts this into a transient
    quantity and can return values orders of magnitude too large.
    """
    g = float(w.get("solar_w_m2", 0.0))
    wind = max(float(w.get("wind_mps", 2.0)), 0.5)
    h_conv = 6.0 + 3.4 * wind ** 0.8
    # Effective long-wave loss at ~313 K.
    h_rad = 4.0 * EMISSIVITY * SIGMA_SB * 313.15**3
    alpha_eff = 0.10        # underfloor transmittance x shell absorptance
    return alpha_eff * g / max(h_conv + h_rad, 1e-6)


# ---------------------------------------------------------------------------
# Reduced 12-cell pack model
# ---------------------------------------------------------------------------


class ReducedPackModel:
    r"""Vectorised 12-cell pack with 1-D lateral conduction coupling.

    State ``T in R^{12}``; the coupling makes the upper/centre/lower tiers differ
    by 1-3 K, which is exactly the behaviour the dashboard's per-cell chart is
    meant to display.  Integration uses an explicit 4th-order RK with a sub-step
    guard so the scheme stays stable for any ``dt`` in ``[0.1, 30] s``.
    """

    def __init__(self, n_cells: int = COLD_PLATE.n_cells, c_th: float | None = None) -> None:
        self.n = n_cells
        # Zone thermal mass, derived from the chemistry when not supplied.
        self.c_th = float(c_th if c_th is not None else CELL_C_TH_J_PER_K)
        # Ring-adjacency coupling matrix (dimensionless; scaled by G in rhs).
        self._lap = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            self._lap[i, i] = -2.0
            self._lap[i, (i - 1) % n_cells] += 1.0
            self._lap[i, (i + 1) % n_cells] += 1.0

    def rhs(
        self,
        temp_c: np.ndarray,
        ambient_c: float,
        solar_w_m2: float,
        road_surface_c: float,
        current_a: float,
        soc: float,
        flow_lpm: float,
        coolant_inlet_c: float,
        spec: ChemistrySpec,
        enable_coupling: bool = True,
        aux_heat_w: float = 0.0,
    ) -> np.ndarray:
        """``dT/dt`` for every thermal zone, degC/s."""
        from ..config import ZONE_PARALLEL_STRINGS, ZONE_SERIES_CELLS
        from ..models.physics import (
            external_convection_coeff,
            heat_transfer_coefficient,
        )

        # Joule source varies slightly with position: the zones at the coolant
        # inlet sit in the coldest region of the plate and dissipate marginally
        # less per unit coolant mass flow.
        position_factor = 1.0 + 0.06 * np.cos(np.pi * (np.arange(self.n) + 0.5) / self.n)

        # Current convention matches physics.lumped_pack_rhs.  In the 96s1p
        # pack the pack current *is* the string current, so a zone's load is
        # ZONE_SERIES_CELLS * I^2 * R_cell.
        r_int = float(spec.r_int(float(temp_c.mean()), soc))
        i_string = current_a / max(ZONE_PARALLEL_STRINGS, 1)
        q_joule = position_factor * (i_string**2) * ZONE_SERIES_CELLS * r_int

        coolant = spec.coolant_props
        h_in = heat_transfer_coefficient(flow_lpm, coolant)

        # Per-zone allocation of the bundle's wetted heat-transfer area.  The
        # full perimeter x length is the area the Nusselt-derived h acts over;
        # it is shared by all zones in the pack.
        a_wet_zone = COLD_PLATE.wetted_perimeter_m * COLD_PLATE.length_m / max(self.n, 1)
        mcp = coolant["density"] * (flow_lpm / 60000.0) * coolant["cp"]
        g_conv = min(h_in * a_wet_zone, mcp * 6.0 / max(self.n, 1))

        # Bottom tier also sees the road surface by radiation/convection.
        road_weight = np.linspace(1.0, 0.35, self.n)
        h_air = float(external_convection_coeff(float(temp_c.mean()), 2.5))
        q_amb = h_air * ZONE_SHELL_AREA_M2 * (temp_c - ambient_c)
        q_road = 1.35 * road_weight * (temp_c - road_surface_c)
        q_cool = g_conv * (temp_c - coolant_inlet_c)
        q_aux = aux_heat_w / max(self.n, 1)
        coupling = CELL_COUPLING_J_PER_K * (
            self._lap @ temp_c if enable_coupling else 0.0
        )
        return (q_joule + q_aux - q_cool - q_amb - q_road + coupling) / self.c_th

    def step(
        self,
        temp_c: np.ndarray,
        dt_s: float,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """One explicit RK4 step with adaptive sub-stepping.

        The sub-step cap (``dt <= 2 s``) is what keeps the explicit integrator
        stable.  The stiffest mode in this system is the *lateral* coupling
        eigenvalue ``G / C_th ~ 45/2950 = 0.015 s^-1``, whose stability limit for
        RK4 is ``dt <~ 6/lambda ~ 400 s``; the tighter practical limit is
        accuracy on the fastest *physical* transient, which is the convective
        time constant ``C_th / (h A) ~ 100 s``.  Capping at 2 s therefore
        resolves the physics comfortably while keeping a 3600 s rollout cheap.
        """
        n_sub = max(1, int(math.ceil(dt_s / 2.0)))
        h = dt_s / n_sub
        T = temp_c
        for _ in range(n_sub):
            k1 = self.rhs(T, *args, **kwargs)
            k2 = self.rhs(T + 0.5 * h * k1, *args, **kwargs)
            k3 = self.rhs(T + 0.5 * h * k2, *args, **kwargs)
            k4 = self.rhs(T + h * k3, *args, **kwargs)
            T = T + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return T


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class HeatwavePlanner:
    """Receding-horizon anticipatory coolant scheduler.

    Typical use::

        planner = HeatwavePlanner("nmc")
        plan = planner.plan(route, weather, pack_heat_w=3200.0, soc=0.7)
        plan.commanded_flow_lpm  # (n,) schedule aligned to route timestamps
    """

    #: Candidate flows swept by the feasibility search, L/min.
    FLOW_CANDIDATES: Tuple[float, ...] = tuple(
        round(0.8 + 0.2 * i, 2) for i in range(0, 68)
    )

    def __init__(self, chemistry: str = "nmc", n_cells: int = COLD_PLATE.n_cells) -> None:
        self.spec: ChemistrySpec = get_chemistry(chemistry)
        self.chemistry = self.spec.key
        # The zone thermal mass is chemistry-dependent (through rho * c_p), so the
        # model is constructed with the right value for this chemistry rather than
        # a shared global.
        self.model = ReducedPackModel(
            n_cells,
            c_th=CELL_GEOMETRY.zone_volume_m3 * self.spec.density * self.spec.cp,
        )
        self.comfort_ceiling_c = COMFORT_CEILING_C[self.chemistry]
        self.margin_floor_c = MARGIN_FLOOR_C[self.chemistry]

    # -- zone detection ----------------------------------------------------
    def detect_zones(
        self,
        route: Sequence[Dict[str, float]],
        weather: Sequence[Dict[str, float]],
        dt_s: float,
        min_duration_s: float = 20.0,
    ) -> List[ZoneForecast]:
        r"""Segment the route into thermally hostile zones.

        A sample is *hot* when its **effective** ambient (lag-filtered, solar
        augmented) exceeds ``ZONE_TRIGGER_C`` or the road-surface temperature
        exceeds ``ROAD_TRIGGER_C``.  Contiguous runs longer than
        ``min_duration_s`` become zones.
        """
        t_eff = effective_ambient_series(weather, dt_s)
        hot = np.zeros(len(route), dtype=bool)
        peak_temp = np.zeros(len(route))
        for i, (w, r) in enumerate(zip(weather, route)):
            peak_temp[i] = t_eff[i] + solar_overshoot(w, self.spec)
            hot[i] = (
                peak_temp[i] > self.comfort_ceiling_c - 6.0
                or float(w.get("road_surface_c", 0.0)) > self.comfort_ceiling_c + 6.0
            )

        zones: List[ZoneForecast] = []
        i = 0
        while i < len(hot):
            if not hot[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            t_in = float(route[i]["t_s"])
            t_out = float(route[j]["t_s"])
            if t_out - t_in >= min_duration_s:
                seg_peak = float(peak_temp[i : j + 1].max())
                seg_solar = max(float(w.get("solar_w_m2", 0.0))
                                for w in weather[i : j + 1])
                lat = float(np.mean([float(r["lat"]) for r in route[i : j + 1]]))
                lon = float(np.mean([float(r["lon"]) for r in route[i : j + 1]]))
                zones.append(
                    ZoneForecast(
                        zone_id=f"HWZ-{len(zones) + 1:02d}",
                        t_enter_s=t_in,
                        t_exit_s=t_out,
                        peak_ambient_c=seg_peak,
                        peak_solar_w_m2=seg_solar,
                        lat_centre=lat,
                        lon_centre=lon,
                        risk_index=0.0,       # filled by plan()
                        predicted_peak_cell_c=0.0,
                        predicted_margin_c=0.0,
                        required_flow_lpm=0.0,
                        baseline_flow_lpm=0.0,
                        dominant_mode="ambient+irradiance",
                        reason=(
                            f"effective ambient {seg_peak:.1f} degC exceeds the "
                            f"{self.comfort_ceiling_c - 6.0:.0f} degC anticipation "
                            f"trigger for {self.chemistry.upper()}"
                        ),
                    )
                )
            i = j + 1
        return zones

    # -- forward rollout ---------------------------------------------------
    def rollout(
        self,
        initial_temp_c: float,
        flow_schedule_lpm: np.ndarray,
        ambient: np.ndarray,
        solar: np.ndarray,
        road: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        coolant_inlet_c: float,
        dt_s: float,
        aux_heat_w: float = 0.0,
    ) -> np.ndarray:
        """Integrate the reduced pack model under a per-timestep flow schedule.

        ``current`` and ``soc`` are per-timestep arrays (the speed-scheduled and
        Coulomb-counted traces) rather than scalars, so the feasibility sweep
        accounts for the fact that the load varies along the route.
        """
        T = np.full(self.model.n, float(initial_temp_c))
        out = np.empty((len(flow_schedule_lpm), self.model.n))
        for i in range(len(flow_schedule_lpm)):
            T = self.model.step(
                T, dt_s, ambient[i], solar[i], road[i], float(current[i]),
                float(soc[i]), float(flow_schedule_lpm[i]), coolant_inlet_c,
                self.spec, aux_heat_w=aux_heat_w,
            )
            out[i] = T
        return out

    # -- feasibility search ------------------------------------------------
    def _search_flow(
        self,
        initial_temp_c: float,
        ambient: np.ndarray,
        solar: np.ndarray,
        road: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        coolant_inlet_c: float,
        dt_s: float,
        baseline_flow_lpm: float,
        aux_heat_w: float = 0.0,
        margin_c: float = 0.0,
    ) -> Tuple[float, float, np.ndarray]:
        """Smallest candidate flow meeting both the comfort and margin constraints.

        Parameters
        ----------
        margin_c:
            Extra headroom demanded below the comfort ceiling, degC.  This is the
            fixed-point variable of the planning loop in :meth:`plan`: the ramp
            from the baseline set-point costs thermal headroom, so the constant-
            flow search must be asked to clear a ceiling that is lower by
            whatever the ramp turned out to cost.

        Returns
        -------
        (flow, peak, trace)
            The selected flow, its constant-flow peak, and the per-timestep peak
            trace.  The candidates ascend, so the first feasible one is *provably*
            the minimal actuation satisfying both constraints -- the controller
            does not over-cool.  If none is feasible, the best-effort
            (lowest-peak) candidate is returned together with its peak so the
            caller can see the residual violation rather than being handed a
            silently-wrong schedule.
        """
        target_ceiling = self.comfort_ceiling_c - margin_c
        best_flow = MAX_COOLANT_FLOW_LPM
        best_peak = float("inf")
        best_trace = np.zeros(len(ambient))

        for flow in self.FLOW_CANDIDATES:
            if flow > MAX_COOLANT_FLOW_LPM:
                break
            trace = self.rollout(
                initial_temp_c, np.full(len(ambient), flow), ambient, solar, road,
                current, soc, coolant_inlet_c, dt_s, aux_heat_w,
            ).max(axis=1)
            peak = float(trace.max())
            margin = self.spec.runaway_c - peak
            if peak <= target_ceiling and margin >= self.margin_floor_c:
                return flow, peak, trace
            if peak < best_peak:
                best_peak, best_flow, best_trace = peak, flow, trace
        return best_flow, best_peak, best_trace

    def _rollout_peak(
        self,
        initial_temp_c: float,
        commanded: np.ndarray,
        ambient: np.ndarray,
        solar: np.ndarray,
        road: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        coolant_inlet_c: float,
        dt_s: float,
        aux_heat_w: float = 0.0,
    ) -> float:
        """Peak zone temperature of the rate-limited schedule actually commanded."""
        trace = self.rollout(
            initial_temp_c, commanded, ambient, solar, road, current, soc,
            coolant_inlet_c, dt_s, aux_heat_w,
        )
        return float(trace.max())

    # -- main entry point --------------------------------------------------
    def plan(
        self,
        route: Sequence[Dict[str, float]],
        weather: Sequence[Dict[str, float]],
        base_current_a: float = DEFAULT_PACK_CURRENT_A,
        base_flow_lpm: float = 6.0,
        initial_temp_c: float = 32.0,
        soc: float = 0.65,
        enable_anticipatory: bool = True,
        coolant_inlet_c: float = COOLANT_SETPOINT_C,
        pack_heat_w: Optional[float] = None,
        aux_heat_w: float = DEFAULT_AUX_HEAT_W,
    ) -> Plan:
        """Produce the full anticipatory schedule for a route.

        Parameters
        ----------
        base_current_a:
            Pack current in A, scaled by driving speed.  Bounded by
            :data:`~backend.config.MAX_PACK_CURRENT_A` (the ~2C continuous
            limit); the model does not extrapolate beyond it.
        pack_heat_w:
            Optional explicit Joule power for the *pack*, converted to an
            equivalent current via ``I = sqrt(P / (n_zones n_s R))``.  When
            supplied it overrides ``base_current_a`` (the dashboard's
            "constant-power" mode).
        aux_heat_w:
            Auxiliary thermal load (fast-charge heaters, BMS, cabin
            conditioning) in W, distributed across the zones.
        """
        n = len(route)
        dt_s = float(route[1]["t_s"] - route[0]["t_s"]) if n > 1 else 1.0
        dt_s = max(0.1, min(dt_s, 30.0))
        times = np.array([float(r["t_s"]) for r in route])

        # -- 1. thermal forecast ------------------------------------------
        t_eff = effective_ambient_series(weather, dt_s)
        solar = np.array([0.85 * float(w.get("solar_w_m2", 0.0)) for w in weather])
        solar_over = np.array([solar_overshoot(w, self.spec) for w in weather])
        ambient = t_eff + solar_over * 0.55
        road = np.array([float(w.get("road_surface_c", t_eff[i] + 8.0))
                         for i, w in enumerate(weather)])

        # -- 2. current profile (speed-scheduled: faster -> higher I) ------
        if pack_heat_w is not None:
            # Invert the pack's total I^2 R load to an equivalent current:
            #   P = n_zones * n_s * (I / n_p)^2 * R  =>  I = sqrt(P n_p^2 / (n_z n_s R))
            r_int_ref = max(float(self.spec.r_int(initial_temp_c, soc)), 1e-6)
            base_current_a = math.sqrt(
                max(pack_heat_w, 1.0) * ZONE_PARALLEL_STRINGS**2
                / max(self.model.n * ZONE_SERIES_CELLS * r_int_ref, 1e-9)
            )
        speeds = np.array([float(r.get("speed_mps", 20.0)) for r in route])
        current = np.clip(
            base_current_a * np.clip(speeds / 22.0, 0.45, 1.65),
            0.0, MAX_PACK_CURRENT_A,
        )

        # -- 3. SOC trajectory (Coulomb counting) ------------------------
        pack_capacity_ah = self.spec.nominal_capacity_ah * PACK_PARALLEL_STRINGS
        soc_trace = np.clip(
            soc - np.cumsum(np.abs(current)) * dt_s / 3600.0 /
            max(pack_capacity_ah, 1e-3),
            0.02, 1.0,
        )

        # -- 4. feasibility search ----------------------------------------
        # The search assumes a *constant* flow, but the schedule actually has to
        # ramp at RAMP_RATE_LPM_PER_S from the baseline.  That ramp is itself
        # thermally costly, so a constant-flow plan that lands exactly on the
        # ceiling will overshoot once the ramp is imposed.  The fix is a
        # fixed-point loop: search, shape and rate-limit the schedule, measure
        # the realised peak, and re-search against a margin that shrinks by the
        # observed overshoot.  It terminates because the margin shrinks
        # monotonically and the flow is bounded by MAX_COOLANT_FLOW_LPM.
        zones = self.detect_zones(route, weather, dt_s)
        margin_c = 0.0
        search_flow, search_peak, peak_trace = MAX_COOLANT_FLOW_LPM, float("inf"), np.zeros(n)
        commanded = np.full(n, base_flow_lpm)

        for _ in range(_SEARCH_REFINE_ITERS):
            search_flow, search_peak, peak_trace = self._search_flow(
                initial_temp_c, ambient, solar, road, current, soc_trace,
                coolant_inlet_c, dt_s, base_flow_lpm, aux_heat_w,
                margin_c,
            )

            floor_flow = float(min(max(search_flow, base_flow_lpm), MAX_COOLANT_FLOW_LPM))
            desired = np.full(n, floor_flow)

            # Start the ramp early enough that the pack is already at the target
            # flow when the vehicle reaches the zone boundary.
            lead_steps = int(HEATWAVE_RAMP_LEAD_S / dt_s)
            for z in zones:
                i_enter = int(np.searchsorted(times, z.t_enter_s))
                i_start = max(0, i_enter - lead_steps)
                desired[i_start:] = np.maximum(desired[i_start:], floor_flow)
                z.required_flow_lpm = floor_flow
                z.baseline_flow_lpm = float(base_flow_lpm)

            if not enable_anticipatory:
                desired = np.full(n, base_flow_lpm)
                for z in zones:
                    z.required_flow_lpm = base_flow_lpm

            commanded = self._rate_limit(desired, dt_s, base_flow_lpm)

            if search_flow >= MAX_COOLANT_FLOW_LPM:
                break                      # saturated; no more authority available

            realised = self._rollout_peak(
                initial_temp_c, commanded, ambient, solar, road, current, soc_trace,
                coolant_inlet_c, dt_s, aux_heat_w,
            )
            overshoot = realised - self.comfort_ceiling_c
            if overshoot <= 0.0:
                break                      # the ceiling is satisfied as commanded
            margin_c = min(margin_c + overshoot, _SEARCH_MAX_MARGIN_C)

        # -- 6. roll out the *actual* commanded schedule ------------------
        cell_trace = np.empty((n, self.model.n))
        T = np.full(self.model.n, float(initial_temp_c))
        for i in range(n):
            T = self.model.step(T, dt_s, ambient[i], solar[i], road[i],
                                float(current[i]), float(soc_trace[i]),
                                float(commanded[i]), coolant_inlet_c, self.spec,
                                aux_heat_w=aux_heat_w)
            cell_trace[i] = T

        peak_cell = cell_trace.max(axis=1)
        spread = cell_trace.max(axis=1) - cell_trace.min(axis=1)
        margin = self.spec.runaway_c - peak_cell
        dT = np.gradient(peak_cell, dt_s) if n > 1 else np.zeros(n)

        # -- 7. risk index ------------------------------------------------
        risk = self._risk_index(peak_cell, dT, spread, ambient)

        # -- 8. actuation events ------------------------------------------
        actuation = self._extract_actuation(times, commanded, base_flow_lpm, zones)

        # -- 9. zone annotations ------------------------------------------
        for z in zones:
            sel = (times >= z.t_enter_s) & (times <= z.t_exit_s)
            if not np.any(sel):
                continue
            z.predicted_peak_cell_c = float(peak_cell[sel].max())
            z.predicted_margin_c = float(margin[sel].min())
            z.risk_index = float(risk[sel].max())
            z.dominant_mode = self._dominant_mode(solar[sel], ambient[sel], road[sel])

        baseline_trace = self.rollout(
            initial_temp_c, np.full(n, base_flow_lpm), ambient, solar, road,
            current, soc_trace, coolant_inlet_c, dt_s, aux_heat_w,
        ).max(axis=1)

        ceiling_respected = bool(
            float(peak_cell.max()) <= self.comfort_ceiling_c + 1e-6
        )
        advisory = self._advisory(
            float(peak_cell.max()), float(baseline_trace.max()),
            float(risk.max()), floor_flow, ceiling_respected,
        )

        return Plan(
            chemistry=self.chemistry,
            time_s=times,
            baseline_flow_lpm=np.full(n, base_flow_lpm),
            commanded_flow_lpm=commanded,
            predicted_cell_c=peak_cell,
            predicted_margin_c=margin,
            risk_index=risk,
            zones=zones,
            actuation=actuation,
            peak_cell_c=float(peak_cell.max()),
            min_margin_c=float(margin.min()),
            total_ramp_distance_lpm=float(np.abs(np.diff(commanded)).sum()),
            advisory=advisory,
            decision_log={
                "search_flow_lpm": float(search_flow),
                "search_peak_cell_c": float(search_peak),
                "commanded_floor_flow_lpm": float(floor_flow),
                "search_margin_c": float(margin_c),
                "comfort_ceiling_c": self.comfort_ceiling_c,
                "margin_floor_c": self.margin_floor_c,
                "ceiling_respected": ceiling_respected,
                "margin_respected": bool(float(margin.min()) >= self.margin_floor_c),
                "flow_saturated": bool(floor_flow >= MAX_COOLANT_FLOW_LPM - 1e-9),
                "baseline_peak_cell_c": float(baseline_trace.max()),
                "peak_cell_c": float(peak_cell.max()),
                "delta_peak_cell_c": float(peak_cell.max() - baseline_trace.max()),
                "anticipation_benefit_k": float(baseline_trace.max() - peak_cell.max()),
                "coolant_lift_lpm": float(np.abs(commanded - base_flow_lpm).sum()),
            },
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _rate_limit(desired: np.ndarray, dt_s: float, initial_lpm: float) -> np.ndarray:
        """Symmetric rate limiter at ``RAMP_RATE_LPM_PER_S``."""
        out = np.empty_like(desired)
        cur = float(initial_lpm)
        max_step = RAMP_RATE_LPM_PER_S * dt_s
        for i, d in enumerate(desired):
            delta = float(np.clip(d - cur, -max_step, max_step))
            cur = cur + delta
            out[i] = float(np.clip(cur, MIN_COOLANT_FLOW_LPM, MAX_COOLANT_FLOW_LPM))
        return out

    def _risk_index(
        self,
        peak_cell: np.ndarray,
        dT: np.ndarray,
        spread: np.ndarray,
        ambient: np.ndarray,
    ) -> np.ndarray:
        r"""Dimensionless risk in ``[0, 1]``.

        A logistic on the comfort-ceiling utilisation, combined with a heating-rate
        term and a lateral-spread term (a large spread signals a blocked channel,
        which is the dominant real-world failure mode):

        .. math::
           \mathcal{R} = \sigma\!\big(6(\bar{T}_{\text{peak}}/T_{\text{ceil}} - 0.82)\big)
                          \cdot \big(1 + 0.35\,\mathrm{clip}(dT/dt, 0, 1)
                          + 0.22\,\mathrm{clip}(\Delta T_{\text{lateral}}/3, 0, 1)\big)
        """
        util = peak_cell / self.comfort_ceiling_c
        base = 1.0 / (1.0 + np.exp(-6.0 * (util - 0.82)))
        rate_term = np.clip(np.abs(dT) / 0.35, 0.0, 1.0)
        spread_term = np.clip(spread / 3.0, 0.0, 1.0)
        amb_term = np.clip((ambient - self.comfort_ceiling_c) / 15.0, 0.0, 1.0)
        raw = base * (1.0 + 0.35 * rate_term + 0.22 * spread_term + 0.18 * amb_term)
        return np.clip(raw, 0.0, 1.0)

    @staticmethod
    def _dominant_mode(solar: np.ndarray, ambient: np.ndarray, road: np.ndarray) -> str:
        """Label the dominant heat-load term for a zone."""
        terms = {
            "irradiance": float(solar.max()) / 1400.0,
            "ambient": float(ambient.max() - 20.0) / 45.0,
            "road-surface": float(road.max() - 25.0) / 55.0,
        }
        return max(terms.items(), key=lambda kv: kv[1])[0]

    def _extract_actuation(
        self,
        times: np.ndarray,
        commanded: np.ndarray,
        baseline: float,
        zones: Sequence[ZoneForecast],
    ) -> List[Actuation]:
        """Threshold the schedule into discrete, human-auditable events."""
        events: List[Actuation] = []
        zone_ids = [(z.t_enter_s, z.zone_id) for z in zones]
        prev = baseline
        for i in range(1, len(commanded)):
            cur = float(commanded[i])
            if abs(cur - prev) < FLOW_DEADBAND_LPM:
                prev = cur
                continue
            rising = cur > prev
            trigger = "heatwave-anticipation" if rising else "zone-exit-cooldown"
            conf = 0.55
            if rising:
                match = [(zid, t0) for t0, zid in zone_ids if t0 <= times[i] + 1e-6]
                if match:
                    zid, t_enter = match[-1]
                    trigger = f"enter-{zid}"
                    conf = min(0.99, 0.72 + 0.25 * min(1.0, (times[i] - t_enter) /
                                                        max(HEATWAVE_LOOKAHEAD_S, 1.0)))
                    conf = max(conf, 0.8)
            events.append(
                Actuation(
                    t_s=float(times[i]),
                    command_lpm=cur,
                    delta_lpm=cur - prev,
                    lead_time_s=float(times[i] - prev) if False else float(dt_or(times, i)),
                    trigger=trigger,
                    confidence=conf,
                    authority="enforced",
                )
            )
            prev = cur
        return events

    def _advisory(self, peak: float, baseline_peak: float, risk: float,
                  flow: float, ceiling_respected: bool) -> str:
        """Plain-language summary surfaced to the operator.

        The saturation case is reported explicitly rather than folded into a
        success message.  When the required flow is at
        :data:`~backend.config.MAX_COOLANT_FLOW_LPM` and the ceiling is still
        exceeded, no coolant schedule can satisfy the constraint -- the heat
        load exceeds what this cold plate can reject -- and saying so is the only
        honest output.  In that state the fail-safe supervisor takes over; the
        planner's job is to reveal the limit, not to hide it.
        """
        delta = baseline_peak - peak
        saturated = flow >= MAX_COOLANT_FLOW_LPM - 1e-9

        if not ceiling_respected:
            excess = peak - self.comfort_ceiling_c
            if saturated:
                return (
                    f"THERMAL LIMIT REACHED: coolant is already at the maximum "
                    f"{MAX_COOLANT_FLOW_LPM:.1f} L/min and the projected peak of "
                    f"{peak:.1f} degC still exceeds the {self.comfort_ceiling_c:.0f} degC "
                    f"comfort ceiling by {excess:.1f} K for {self.chemistry.upper()}. "
                    f"No coolant schedule can satisfy this constraint: the heat load "
                    f"exceeds what this cold plate can reject at any flow rate. Reduce the "
                    f"load (charging power or ambient exposure) or increase plate area. "
                    f"The fail-safe supervisor will derate the pack."
                )
            return (
                f"CAUTION: projected peak {peak:.1f} degC exceeds the "
                f"{self.comfort_ceiling_c:.0f} degC comfort ceiling by {excess:.1f} K; "
                f"the required flow {flow:.1f} L/min could not be reached within the "
                f"route horizon. Anticipation still reduced the peak by {delta:.1f} K "
                f"versus the constant-flow baseline."
            )

        if delta > 0.5:
            return (
                f"Anticipatory ramp to {flow:.1f} L/min held the pack {delta:.1f} K below "
                f"the no-control peak ({baseline_peak:.1f} -> {peak:.1f} degC); "
                f"comfort ceiling {self.comfort_ceiling_c:.0f} degC respected for "
                f"{self.chemistry.upper()}."
            )
        return (
            f"No anticipatory action required; peak cell temperature {peak:.1f} degC is "
            f"within the {self.comfort_ceiling_c:.0f} degC comfort ceiling (risk index "
            f"{risk:.2f})."
        )

    # -- explanation -------------------------------------------------------
    def explain(self, plan: Plan, index: int) -> Dict[str, object]:
        """Return the deciding terms for the schedule at timestep ``index``."""
        i = int(np.clip(index, 0, len(plan.time_s) - 1))
        return {
            "t_s": float(plan.time_s[i]),
            "commanded_flow_lpm": float(plan.commanded_flow_lpm[i]),
            "baseline_flow_lpm": float(plan.baseline_flow_lpm[i]),
            "predicted_cell_c": float(plan.predicted_cell_c[i]),
            "comfort_ceiling_c": self.comfort_ceiling_c,
            "margin_c": float(plan.predicted_margin_c[i]),
            "margin_floor_c": self.margin_floor_c,
            "risk_index": float(plan.risk_index[i]),
            "reynolds": reynolds_from_flow(
                float(plan.commanded_flow_lpm[i]), self.spec.coolant_props
            ),
            "ramp_rate_lpm_per_s": RAMP_RATE_LPM_PER_S,
            "leading_zone": next(
                (z.zone_id for z in plan.zones if z.t_enter_s >= plan.time_s[i]), None
            ),
        }


def dt_or(times: np.ndarray, i: int) -> float:
    """Time delta between samples ``i`` and ``i-1`` (seconds)."""
    if i <= 0 or len(times) <= 1:
        return 1.0
    return float(max(times[i] - times[i - 1], 1e-3))


def planner_for(chemistry: str) -> HeatwavePlanner:
    """Construct a planner bound to a chemistry."""
    return HeatwavePlanner(chemistry)


# ---------------------------------------------------------------------------
# Fail-safe supervisor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SafetyState:
    """State of the three-tier fail-safe supervisor.

    ``ok``      nominal control.
    ``derate``  pack current reduced 30% and flow forced to max; used when the
                predicted margin drops below the chemistry floor.
    ``trip``    flow forced to max, current commanded to zero, contactor open;
                latched -- it is only cleared by an explicit operator reset.
    """

    tier: str = "ok"
    latched_trip: bool = False
    reason: str = ""
    forced_flow_lpm: float = MAX_COOLANT_FLOW_LPM
    derate_factor: float = 1.0

    def evaluate(self, peak_cell_c: float, spec: ChemistrySpec,
                 margin_floor_c: float | None = None) -> "SafetyState":
        """Latch/clear the supervisor given the current peak cell temperature."""
        floor = margin_floor_c if margin_floor_c is not None else MARGIN_FLOOR_C[spec.key]
        margin = spec.runaway_c - peak_cell_c
        if self.latched_trip:
            return self
        if margin < 0.0 or peak_cell_c >= spec.runaway_c:
            self.tier = "trip"
            self.latched_trip = True
            self.derate_factor = 0.0
            self.reason = (
                f"cell temperature {peak_cell_c:.1f} degC reached the "
                f"{spec.runaway_c:.0f} degC exotherm onset for {spec.key.upper()}"
            )
        elif margin < floor:
            self.tier = "derate"
            self.derate_factor = 0.7
            self.forced_flow_lpm = MAX_COOLANT_FLOW_LPM
            self.reason = (
                f"exotherm margin {margin:.1f} K below the {floor:.0f} K floor; "
                f"pack current derated to 70%"
            )
        else:
            if self.tier == "derate":
                # Hysteresis: require 4 K of recovery before clearing.
                if margin > floor + 4.0:
                    self.tier = "ok"
                    self.derate_factor = 1.0
                    self.reason = "exotherm margin recovered"
            else:
                self.tier = "ok"
                self.derate_factor = 1.0
                self.reason = "nominal"
        return self

    def reset(self) -> "SafetyState":
        """Operator reset of a latched trip."""
        self.tier = "ok"
        self.latched_trip = False
        self.derate_factor = 1.0
        self.reason = "operator reset"
        return self