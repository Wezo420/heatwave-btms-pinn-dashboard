/**
 * Shared TypeScript types mirroring `backend/schemas.py`.
 *
 * These are hand-maintained counterparts to the Pydantic models. The backend
 * publishes its OpenAPI schema at `/openapi.json`; a codegen step could replace
 * this file, but keeping it hand-written means the frontend compiles without
 * a Python runtime -- which matters for CI on the frontend alone.
 */

export type ChemistryKey = 'nmc' | 'lfp';

export type OperatingMode =
  | 'inference'
  | 'warmup'
  | 'cooldown'
  | 'fastcharge'
  | 'load';

export type TripState = 'ok' | 'derate' | 'trip';

export type Authority = 'advisory' | 'enforced';

/* ------------------------------------------------------------------ */
/* Pack limits                                                         */
/*                                                                     */
/* These mirror the backend constants in backend/config.py. They are     */
/* duplicated rather than fetched at runtime because the dashboard       */
/* must be usable (and must not crash) before the backend answers --     */
/* and because an out-of-range value here used to be sent unvalidated    */
/* and rejected with 422, which the retry logic then hammered.           */
/* ------------------------------------------------------------------ */

/** Pack topology: 96 series cells, 1 parallel string. */
export const PACK_SERIES_CELLS = 96;
export const PACK_PARALLEL_STRINGS = 1;

/**
 * Maximum continuously addressable pack current, A.
 *
 * ~2C of the 9.7 Ah pack. In a series pack the pack current *is* the string
 * current, so this is also the cell ceiling. A request above it would make the
 * Arrhenius resistance extrapolate ~90x past its fitted range; the backend
 * rejects such a request with 422 rather than extrapolate.
 */
export const MAX_PACK_CURRENT_A = 20;

/** Default cruise pack current, A (~1.2C). */
export const DEFAULT_PACK_CURRENT_A = 12;

/** Coolant flow window the BMS can actually command, L/min. */
export const MIN_COOLANT_FLOW_LPM = 0.8;
export const MAX_COOLANT_FLOW_LPM = 14;

/**
 * Part-load cruise flow, L/min.
 *
 * The *baseline* the anticipatory planner is scored against. At 6 L/min the
 * cold plate holds the pack ~11 K above the 42 degC ceiling with nothing left
 * for the controller to anticipate; the claim only has content against a
 * part-load baseline with a pre-warmed pack.
 */
export const BASELINE_CRUISE_FLOW_LPM = 2.0;

/** Pack temperature at scenario start, degC -- post-DC-fast-charge. */
export const PRE_FAST_CHARGE_TEMP_C = 38.0;

/* ------------------------------------------------------------------ */
/* Catalogue                                                          */
/* ------------------------------------------------------------------ */

export interface CoolantSpec {
  density_kg_m3: number;
  cp_j_kgk: number;
  k_w_mk: number;
  mu_pa_s: number;
  prandtl: number;
}

export interface ChannelSpec {
  n_channels: number;
  channel_width_mm: number;
  channel_depth_mm: number;
  length_mm: number;
  width_mm: number;
}

export interface ChemistryInfo {
  key: ChemistryKey;
  label: string;
  cathodes: string;
  anode: string;
  cp_j_kgk: number;
  thermal_runaway_onset_c: number;
  internal_resistance_reference_ohm: number;
  activation_energy_j_mol: number;
  nominal_capacity_ah: number;
  energy_density_wh_kg: number;
  thermal_diffusivity_m2_s: number;
  coolant: CoolantSpec;
  channels: ChannelSpec;
  notes: string;
  recommended_max_cell_temp_c: number;
  default_flow_setpoint_lpm: number;
}

export interface CatalogueResponse {
  chemistries: ChemistryInfo[];
  default: ChemistryKey;
  max_flow_lpm: number;
  min_flow_lpm: number;
  critical_temperature_c: number;
  inference_latency_budget_ms: number;
  physics_constants: Record<string, number>;
}

/* ------------------------------------------------------------------ */
/* Model introspection                                                 */
/* ------------------------------------------------------------------ */

export interface ModelInfo {
  architecture: string;
  total_parameters: number;
  conditioning_dim: number;
  hidden_dim: number;
  n_hidden_layers: number;
  fourier_features: number;
  checkpoint_path: string;
  checkpoint_present: boolean;
  device: string;
  dtype: string;
  gradient_penalty: Record<string, number>;
  trained_chemistries: ChemistryKey[];
  latency_budget_ms: number;
  notes: string;
}

/* ------------------------------------------------------------------ */
/* PINN field                                                          */
/* ------------------------------------------------------------------ */

export interface ResidualReport {
  pde_energy: number;
  pde_momentum: number;
  pde_continuity: number;
  bc_interface_flux: number;
  bc_wall_no_slip: number;
  bc_inlet: number;
  bc_outlet: number;
  total_physics: number;
  adaptive_weights: Record<string, number>;
  is_converged: boolean;
  nonfinite_guard: boolean;
}

export interface FieldResponse {
  chemistry: ChemistryKey;
  grid: [number, number, number];
  axes: {
    x_m: number[];
    y_m: number[];
    z_m: number[];
  };
  temperature_c: number[][][];
  max_temp_c: number;
  min_temp_c: number;
  mean_temp_c: number;
  cell_outlet_c: number;
  coolant_outlet_c: number;
  heat_rejected_w: number;
  thermal_runaway_margin_c: number;
  inference_latency_ms: number;
  within_latency_budget: boolean;
  residuals: ResidualReport;
  coefficients: Record<string, number> | null;
  timestamp: string;
}

/* ------------------------------------------------------------------ */
/* Telemetry                                                           */
/* ------------------------------------------------------------------ */

export interface TelemetryFrame {
  t_s: number;
  chemistry: ChemistryKey;
  cell_temps_c: number[];
  max_cell_temp_c: number;
  mean_cell_temp_c: number;
  coolant_flow_lpm: number;
  commanded_flow_lpm: number;
  coolant_inlet_c: number;
  coolant_outlet_c: number;
  ambient_c: number;
  road_surface_c: number;
  joule_heat_w: number;
  heat_rejected_w: number;
  soc: number;
  state_of_charge_pct: number;
  thermal_runaway_margin_c: number;
  risk_index: number;
  lat: number;
  lon: number;
  speed_mps: number;
  anticipatory_active: boolean;
  trip_state: TripState;
}

/* ------------------------------------------------------------------ */
/* Route / weather                                                     */
/* ------------------------------------------------------------------ */

export interface GeoPoint {
  lat: number;
  lon: number;
  t_s: number;
  speed_mps: number;
}

export interface WeatherSample {
  t_s: number;
  ambient_c: number;
  relative_humidity?: number;
  solar_w_m2: number;
  wind_mps: number;
  road_surface_c: number;
}

export interface HeatwaveZone {
  zone_id: string;
  t_enter_s: number;
  t_exit_s: number;
  peak_ambient_c: number;
  risk_index: number;
  predicted_peak_cell_c: number;
  predicted_margin_c: number;
  required_flow_lpm: number;
  dominant_mode: string;
}

export interface ActuationEvent {
  t_s: number;
  command_lpm: number;
  delta_lpm: number;
  lead_time_s: number;
  trigger: string;
  confidence: number;
  authority: Authority;
}

export interface SimulationResponse {
  chemistry: ChemistryKey;
  frames: TelemetryFrame[];
  heatwave_zones: HeatwaveZone[];
  actuation: ActuationEvent[];
  peak_cell_temp_c: number;
  min_thermal_runaway_margin_c: number;
  total_energy_wh: number;
  cooling_energy_kwh: number;
  coolant_used_l: number;
  inference_latency_p95_ms: number;
  within_latency_budget: boolean;
  simulation_walltime_s: number;
  timestamp: string;
  advisory?: string;
}

/* ------------------------------------------------------------------ */
/* NASA calibration                                                    */
/* ------------------------------------------------------------------ */

export interface NasaCalibration {
  dataset_root: string;
  dataset_present: boolean;
  synthetic_fallback: boolean;
  cells: Array<{
    cell_id: string;
    cycles_available: number;
    n_operating_points: number;
    r_int_ref_ohm: number;
    r_int_std_ohm: number;
    r_int_25c_ohm: number;
    ea_j_mol: number;
    rmse_voltage: number;
    capacity_fade_pct: number;
    eol_capacity_pct: number;
  }>;
  aggregate: Record<string, number>;
  coolant_heat_capacity_w_k: number;
  notes: string;
  timestamp: string;
}

/* ------------------------------------------------------------------ */
/* Control + health                                                    */
/* ------------------------------------------------------------------ */

export interface FlowCommandResponse {
  accepted: boolean;
  applied_flow_lpm: number;
  mode: string;
  ramp_rate_lpm_per_s: number;
  estimated_settle_s: number;
  heat_rejection_w: number;
  trip_state: TripState;
  message: string;
  timestamp: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error';
  version: string;
  uptime_s: number;
  device: string;
  torch_version: string;
  numpy_version: string;
  checkpoint_present: boolean;
  nasa_dataset_present: boolean;
  chemistry_defaults: Record<string, number>;
  timestamp: string;
  detail: Record<string, unknown>;
}

/* ------------------------------------------------------------------ */
/* Bootstrap                                                           */
/* ------------------------------------------------------------------ */

export interface BootstrapResponse {
  chemistry: ChemistryKey;
  catalogue: CatalogueResponse;
  model: ModelInfo;
  scenario: {
    chemistry: ChemistryKey;
    duration_s: number;
    dt_s: number;
    route: GeoPoint[];
    weather: WeatherSample[];
    description: string;
  };
  field: FieldResponse;
  safety: {
    comfort_ceiling_c: number;
    margin_floor_c: number;
    exotherm_onset_c: number;
    ambient_tau_s: number;
    ramp_rate_lpm_per_s: number;
    tiers: TripState[];
  };
  bootstrap_latency_ms: number;
  timestamp: string;
}

export interface ComparisonResponse {
  nmc: ComparisonEntry;
  lfp: ComparisonEntry;
  delta: {
    peak_cell_temp_c: number;
    min_exotherm_margin_c: number;
    peak_commanded_flow_lpm: number;
    total_ramp_lpm: number;
    interpretation: string;
  };
  timestamp: string;
}

export interface ComparisonEntry {
  peak_cell_temp_c: number;
  comfort_ceiling_c: number;
  ceiling_respected: boolean;
  exotherm_onset_c: number;
  min_exotherm_margin_c: number;
  margin_floor_c: number;
  margin_respected: boolean;
  specific_heat_j_kgk: number;
  r_int_at_40c_ohm: number;
  arrhenius_slope_kj_mol: number;
  mean_commanded_flow_lpm: number;
  peak_commanded_flow_lpm: number;
  total_ramp_lpm: number;
  n_zones: number;
  advisory: string;
}

/* ------------------------------------------------------------------ */
/* WebSocket envelope                                                  */
/* ------------------------------------------------------------------ */

export interface TelemetryWsMessage {
  type: 'hello' | 'telemetry' | 'pong' | 'ack' | 'error';
  data?: TelemetryFrame;
  chemistry?: ChemistryKey;
  tick_s?: number;
  exotherm_onset_c?: number;
  t_s?: number;
  action?: string;
  flow_lpm?: number;
  detail?: string;
  timestamp?: string;
}