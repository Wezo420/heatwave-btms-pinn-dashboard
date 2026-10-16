r"""
backend.schemas
===============

Pydantic v2 contracts shared by the FastAPI routers and (via the generated
OpenAPI schema) by the Next.js dashboard's typed client.

Conventions
-----------
* All temperatures crossing the API boundary are **degrees Celsius**.  Kelvin is
  used only inside :mod:`backend.config` / :mod:`backend.models.physics`.
* All flow rates are **litres per minute** (``lpm``).
* Every request model forbids unknown fields (``extra="forbid"``) so that a typo
  in the dashboard is surfaced as a 422 rather than silently ignored -- an
  important property for a safety-adjacent control surface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import (
    BASELINE_CRUISE_FLOW_LPM,
    DEFAULT_PACK_CURRENT_A,
    MAX_COOLANT_FLOW_LPM,
    MAX_PACK_CURRENT_A,
    MIN_COOLANT_FLOW_LPM,
    PRE_FAST_CHARGE_TEMP_C,
    SUPPORTED_CHEMISTRIES,
)

ChemistryKey = Literal["nmc", "lfp"]
OperatingMode = Literal["inference", "warmup", "cooldown", "fastcharge", "load"]


# ---------------------------------------------------------------------------
# Shared / catalogue
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CoolantSpec(StrictModel):
    """Thermophysical properties of the loop coolant."""

    density_kg_m3: float = Field(1051.0, gt=0, description="Coolant density, kg/m^3.")
    cp_j_kgk: float = Field(3400.0, gt=0, description="Specific heat capacity, J/(kg*K).")
    k_w_mk: float = Field(0.505, gt=0, description="Thermal conductivity, W/(m*K).")
    mu_pa_s: float = Field(2.05e-3, gt=0, description="Dynamic viscosity, Pa*s.")
    prandtl: float = Field(13.8, gt=0, description="Prandtl number.")


class ChannelSpec(StrictModel):
    """Microchannel cold-plate geometry exposed to the client."""

    n_channels: int = Field(24, ge=1, le=200)
    channel_width_mm: float = Field(2.2, gt=0)
    channel_depth_mm: float = Field(3.5, gt=0)
    length_mm: float = Field(360.0, gt=0)
    width_mm: float = Field(230.0, gt=0)


class ChemistryInfo(StrictModel):
    """Catalogue entry for one supported chemistry (drives the UI toggle)."""

    key: ChemistryKey
    label: str
    cathodes: str
    anode: str
    cp_j_kgk: float
    thermal_runaway_onset_c: float
    internal_resistance_reference_ohm: float
    activation_energy_j_mol: float
    nominal_capacity_ah: float
    energy_density_wh_kg: float
    thermal_diffusivity_m2_s: float
    coolant: CoolantSpec
    channels: ChannelSpec
    notes: str
    recommended_max_cell_temp_c: float
    default_flow_setpoint_lpm: float


class CatalogueResponse(StrictModel):
    """Response of ``GET /api/catalog/chemistries``."""

    chemistries: List[ChemistryInfo]
    default: ChemistryKey
    max_flow_lpm: float
    min_flow_lpm: float
    critical_temperature_c: float
    inference_latency_budget_ms: float
    physics_constants: Dict[str, float]


# ---------------------------------------------------------------------------
# Physics / prediction
# ---------------------------------------------------------------------------


class OperatingPoint(StrictModel):
    """Boundary conditions for a single quasi-steady pack operating point.

    ``pack_current_a`` and the coolant-flow window both carry the *pack's own*
    ratings rather than the sensor range they could represent. In the 96s1p
    topology of this study the string current is the pack current and its
    continuous ceiling is 20 A, so a request for 180 A is a state the hardware
    cannot be in: accepting it would make the Arrhenius resistance extrapolate
    ~90x past its fitted range and report a Joule load no instrumentation has
    ever validated. The 422 that comes back is the honest answer.
    """

    chemistry: ChemistryKey = "nmc"
    coolant_inlet_c: float = Field(26.0, ge=-40, le=120)
    coolant_flow_lpm: float = Field(6.0, ge=MIN_COOLANT_FLOW_LPM, le=MAX_COOLANT_FLOW_LPM)
    ambient_c: float = Field(34.0, ge=-60, le=65)
    pack_current_a: float = Field(
        DEFAULT_PACK_CURRENT_A, ge=-MAX_PACK_CURRENT_A, le=MAX_PACK_CURRENT_A,
    )
    state_of_charge: float = Field(0.5, ge=0.0, le=1.0)
    heat_watts: float = Field(
        0.0,
        ge=0,
        le=20_000,
        description="Auxiliary thermal load (fast-charge heaters, BMS, cabin).",
    )
    mode: OperatingMode = "inference"
    elapsed_s: float = Field(0.0, ge=0, le=86_400)
    thermal_ambient_s: float = Field(
        5.0, ge=0, le=200, description="Overshoot of ambient from solar irradiance, K."
    )

    @field_validator("state_of_charge")
    @classmethod
    def _soc_range(cls, v: float) -> float:
        if not 0.005 <= v <= 0.995:
            raise ValueError("state_of_charge must lie in [0.005, 0.995]")
        return v


class FieldRequest(StrictModel):
    """Request body of ``POST /api/predict/field``."""

    operating_point: OperatingPoint
    grid: List[int] = Field([28, 20, 14], description="[nx, ny, nz] collocation grid.")
    include_coefficients: bool = False
    query_points: Optional[List[List[float]]] = Field(
        None, description="Optional explicit query points in metres [[x,y,z], ...]."
    )


class FieldSlice(StrictModel):
    """Temperature on a constant-y plane."""

    y_m: float
    axis_x: List[float]
    axis_z: List[float]
    values_c: List[List[float]]


class ResidualReport(StrictModel):
    """Physics-residual diagnostics returned alongside every field evaluation."""

    pde_energy: float
    pde_momentum: float
    pde_continuity: float
    bc_interface_flux: float
    bc_wall_no_slip: float
    bc_inlet: float
    bc_outlet: float
    total_physics: float
    adaptive_weights: Dict[str, float]
    is_converged: bool
    nonfinite_guard: bool


class FieldResponse(StrictModel):
    """Response of ``POST /api/predict/field``."""

    chemistry: ChemistryKey
    grid: List[int]
    axes: Dict[str, List[float]]
    temperature_c: List[List[List[float]]]
    max_temp_c: float
    min_temp_c: float
    mean_temp_c: float
    cell_outlet_c: float
    coolant_outlet_c: float
    heat_rejected_w: float
    thermal_runaway_margin_c: float
    inference_latency_ms: float
    within_latency_budget: bool
    residuals: ResidualReport
    coefficients: Optional[Dict[str, float]] = None
    timestamp: str


class PredictRequest(StrictModel):
    """Request body of ``POST /api/predict/single``."""

    operating_point: OperatingPoint
    point: List[float] = Field(..., min_length=3, max_length=3)


class PredictResponse(StrictModel):
    """Single-point temperature query."""

    chemistry: ChemistryKey
    point_m: List[float]
    temperature_c: float
    uncertainty_k: float
    dT_dt_c_per_s: float
    local_heat_flux_w_m2: float
    timestamp: str


# ---------------------------------------------------------------------------
# Route / heatwave
# ---------------------------------------------------------------------------


class GeoPoint(StrictModel):
    """A route waypoint (mock GPS fix)."""

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    t_s: float = Field(..., ge=0, description="Seconds from simulation start.")
    speed_mps: float = Field(22.0, ge=0, le=90)


class WeatherSample(StrictModel):
    """Ambient weather observation attached to a route fix."""

    t_s: float
    ambient_c: float
    relative_humidity: float = Field(0.35, ge=0, le=1)
    solar_w_m2: float = Field(850.0, ge=0, le=1400)
    wind_mps: float = Field(2.5, ge=0, le=60)
    road_surface_c: float = Field(48.0, ge=-40, le=90)


class HeatwaveZone(StrictModel):
    """A geofenced high-temperature segment detected by the anticipatory planner."""

    zone_id: str
    t_enter_s: float
    t_exit_s: float
    peak_ambient_c: float
    risk_index: float = Field(..., ge=0, le=1)
    predicted_peak_cell_c: float
    predicted_margin_c: float
    required_flow_lpm: float
    dominant_mode: str


class ActuationEvent(StrictModel):
    """A commanded change of the coolant mass-flow set-point."""

    t_s: float
    command_lpm: float
    delta_lpm: float
    lead_time_s: float
    trigger: str
    confidence: float = Field(..., ge=0, le=1)
    authority: Literal["advisory", "enforced"] = "enforced"


class SimulationRequest(StrictModel):
    """Request body of ``POST /api/simulate/trajectory``.

    The defaults describe the scenario the controller is actually *for*: a pack
    that has just come off sustained DC fast charging and is already near its
    42 degC ceiling, entering a heatwave, with the coolant pump still at its
    part-load point because nothing has gone wrong yet.

    That combination matters.  A cool pack on a pump running at 6 L/min rejects
    ~4.7 kW at only 16 K above the coolant -- several times the ~1 kW this pack
    actually produces -- so *no* controller has anything to do, and an
    anticipatory schedule would be indistinguishable from constant flow.  The
    anticipatory claim is only meaningful against a part-load baseline and a
    pre-warmed pack, which is exactly when a reactive BMS would wait too long.
    """

    chemistry: ChemistryKey = "nmc"
    duration_s: float = Field(180.0, ge=5, le=3600)
    dt_s: float = Field(1.0, gt=0, le=30)
    route: List[GeoPoint] = Field(..., min_length=2, max_length=2000)
    weather: List[WeatherSample] = Field(..., min_length=2, max_length=2000)
    base_current_a: float = Field(
        DEFAULT_PACK_CURRENT_A,
        ge=-MAX_PACK_CURRENT_A, le=MAX_PACK_CURRENT_A,
        description="Cruise pack current; the pack is 96s1p so this is also the string current.",
    )
    base_flow_lpm: float = Field(
        BASELINE_CRUISE_FLOW_LPM,
        ge=MIN_COOLANT_FLOW_LPM, le=MAX_COOLANT_FLOW_LPM,
        description=(
            "Part-load cruise flow the BMS commands when nothing looks wrong. "
            "This is the *baseline* the anticipatory planner has to beat."
        ),
    )
    initial_cell_temp_c: float = Field(
        PRE_FAST_CHARGE_TEMP_C,
        ge=-20, le=200,
        description=(
            "Pack temperature at scenario start. The default is the state after "
            "sustained DC fast charging, which is what makes the following "
            "heatwave a problem."
        ),
    )
    initial_soc: float = Field(0.65, ge=0.005, le=0.995)
    enable_anticipatory: bool = True
    enable_learned_field: bool = True
    seed: int = 1337

    @model_validator(mode="after")
    def _align_series(self) -> "SimulationRequest":
        if abs(self.duration_s - self.route[-1].t_s) > 1.0:
            object.__setattr__(
                self, "route",
                [*self.route[:-1], self.route[-1].model_copy(update={"t_s": self.duration_s})],
            )
        return self


class TelemetryFrame(StrictModel):
    """One time-step of the streaming telemetry feed (also used by the REST sim)."""

    t_s: float
    chemistry: ChemistryKey
    cell_temps_c: List[float] = Field(..., description="Per-cell temperatures.")
    max_cell_temp_c: float
    mean_cell_temp_c: float
    coolant_flow_lpm: float
    commanded_flow_lpm: float
    coolant_inlet_c: float
    coolant_outlet_c: float
    ambient_c: float
    road_surface_c: float
    joule_heat_w: float
    heat_rejected_w: float
    soc: float
    state_of_charge_pct: float
    thermal_runaway_margin_c: float
    risk_index: float
    lat: float
    lon: float
    speed_mps: float
    anticipatory_active: bool
    trip_state: Literal["ok", "derate", "trip"] = "ok"


class SimulationResponse(StrictModel):
    """Response of ``POST /api/simulate/trajectory``."""

    chemistry: ChemistryKey
    frames: List[TelemetryFrame]
    heatwave_zones: List[HeatwaveZone]
    actuation: List[ActuationEvent]
    peak_cell_temp_c: float
    min_thermal_runaway_margin_c: float
    total_energy_wh: float
    #: Human-readable summary of what the anticipatory schedule bought, and what
    #: it cost in coolant.  Declared on the schema rather than bolted on as an
    #: extra dict key so FastAPI's response validation accepts it.
    advisory: str = Field(
        ...,
        description=(
            "One-paragraph statement of the peak-temperature reduction and the "
            "coolant lift that achieved it."
        ),
    )
    cooling_energy_kwh: float
    coolant_used_l: float
    inference_latency_p95_ms: float
    within_latency_budget: bool
    simulation_walltime_s: float
    timestamp: str


class ForecastRequest(StrictModel):
    """Request body of ``POST /api/heatwave/forecast`` (pure planning, no rollout)."""

    route: List[GeoPoint] = Field(..., min_length=2, max_length=2000)
    weather: List[WeatherSample] = Field(..., min_length=2, max_length=2000)
    chemistry: ChemistryKey = "nmc"
    pack_heat_w: float = Field(3200.0, ge=0, le=40_000)
    state_of_charge: float = Field(0.7, ge=0.005, le=0.995)


class ForecastResponse(StrictModel):
    """Response of ``POST /api/heatwave/forecast``."""

    zones: List[HeatwaveZone]
    schedule: List[ActuationEvent]
    total_ramp_distance_lpm: float
    max_predicted_cell_temp_c: float
    advisory: str


# ---------------------------------------------------------------------------
# NASA prognostics
# ---------------------------------------------------------------------------


class NacaCalibrationCell(StrictModel):
    """Per-cell calibration record extracted from a NASA run-to-failure cycle."""

    cell_id: str
    cycles_available: int
    n_operating_points: int
    r_int_ref_ohm: float
    r_int_std_ohm: float
    r_int_25c_ohm: float
    ea_j_mol: float
    rmse_voltage: float
    capacity_fade_pct: float
    eol_capacity_pct: float


class NacaCalibrationResponse(StrictModel):
    """Response of ``GET /api/nasa/calibration``."""

    dataset_root: str
    dataset_present: bool
    synthetic_fallback: bool
    cells: List[NacaCalibrationCell]
    aggregate: Dict[str, float]
    coolant_heat_capacity_w_k: float
    notes: str
    timestamp: str


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


class FlowCommandRequest(StrictModel):
    """Request body of ``POST /api/control/flow``."""

    chemistry: ChemistryKey = "nmc"
    flow_lpm: float = Field(..., ge=MIN_COOLANT_FLOW_LPM, le=MAX_COOLANT_FLOW_LPM)
    mode: Literal["manual", "auto", "anticipatory", "failsafe"] = "auto"
    operator_id: str = Field("dashboard-user", max_length=64)

    @field_validator("flow_lpm")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("flow_lpm must be finite")
        return v


class FlowCommandResponse(StrictModel):
    """Acknowledgement of a coolant set-point command."""

    accepted: bool
    applied_flow_lpm: float
    mode: str
    ramp_rate_lpm_per_s: float
    estimated_settle_s: float
    heat_rejection_w: float
    trip_state: Literal["ok", "derate", "trip"]
    message: str
    timestamp: str


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------


class ModelInfoResponse(StrictModel):
    """Response of ``GET /api/model/info``."""

    architecture: str
    total_parameters: int
    conditioning_dim: int
    hidden_dim: int
    n_hidden_layers: int
    fourier_features: int
    checkpoint_path: str
    checkpoint_present: bool
    device: str
    dtype: str
    gradient_penalty: Dict[str, float]
    trained_chemistries: List[str]
    latency_budget_ms: float
    notes: str


class HealthResponse(StrictModel):
    """Response of ``GET /api/health``."""

    status: Literal["ok", "degraded", "error"]
    version: str
    uptime_s: float
    device: str
    torch_version: str
    numpy_version: str
    checkpoint_present: bool
    nasa_dataset_present: bool
    chemistry_defaults: Dict[str, float]
    timestamp: str
    detail: Dict[str, Any] = Field(default_factory=dict)


def utcnow_iso() -> str:
    """Timezone-aware UTC ISO-8601 timestamp used on every response."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


#: Re-exported so the routers can validate the chemistry literal consistently.
VALID_CHEMISTRIES = SUPPORTED_CHEMISTRIES
Annotated  # noqa: B018  (kept for typing re-export compatibility)