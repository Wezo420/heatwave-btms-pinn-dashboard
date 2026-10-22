/**
 * Typed client for the FastAPI backend.
 *
 * All requests go through `/api/...` on the *same origin*; `next.config.mjs`
 * rewrites those to `BACKEND_URL` in development and to the origin in
 * production. Keeping the base path relative means there is nothing to
 * reconfigure between local and deployed environments.
 *
 * Error handling: every helper throws `ApiError` carrying the backend's
 * `traceback_id`, which is what the support flow (and the auto-heal monitor's
 * log correlation) keys on.
 */

import type {
  ActuationEvent,
  BootstrapResponse,
  CatalogueResponse,
  ChemistryKey,
  ComparisonResponse,
  FieldResponse,
  FlowCommandResponse,
  HealthResponse,
  HeatwaveZone,
  ModelInfo,
  NasaCalibration,
  OperatingMode,
  SimulationResponse,
  TelemetryFrame,
  TripState,
} from './types';
import {
  BASELINE_CRUISE_FLOW_LPM,
  DEFAULT_PACK_CURRENT_A,
} from './types';

export const API_BASE = '/api';

/** Error thrown by every helper; preserves the backend traceback id. */
export class ApiError extends Error {
  readonly status: number;
  readonly tracebackId: string | null;
  readonly detail: unknown;

  constructor(message: string, status: number, tracebackId: string | null, detail: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.tracebackId = tracebackId;
    this.detail = detail;
  }
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
  timeoutMs?: number;
}

/** Core fetch wrapper: JSON in, JSON out, typed errors, hard timeout. */
async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, query, timeoutMs = 20_000, headers, ...rest } = options;

  const url = new URL(`${API_BASE}${path}`, typeof window === 'undefined' ? 'http://localhost' : window.location.origin);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url.toString(), {
      ...rest,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(headers ?? {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });

    const text = await response.text();
    let payload: unknown = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = text;
      }
    }

    if (!response.ok) {
      const p = payload as Record<string, unknown> | null;
      const message =
        (p?.detail as string) ??
        (typeof p?.error === 'string' ? p.error : null) ??
        `Request failed with ${response.status}`;
      throw new ApiError(
        message,
        response.status,
        (p?.traceback_id as string) ?? null,
        p?.detail ?? payload,
      );
    }

    return payload as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if ((err as Error).name === 'AbortError') {
      throw new ApiError(`Request timed out after ${timeoutMs} ms`, 408, null, null);
    }
    throw new ApiError(
      `Network error contacting ${url.pathname}: ${(err as Error).message}`,
      0,
      null,
      null,
    );
  } finally {
    clearTimeout(timer);
  }
}

/* ------------------------------------------------------------------ */
/* Meta                                                               */
/* ------------------------------------------------------------------ */

export const getHealth = () => request<HealthResponse>('/health', { timeoutMs: 5_000 });

export const getCatalogue = () => request<CatalogueResponse>('/catalog/chemistries');

export const getModelInfo = () => request<ModelInfo>('/model/info', { timeoutMs: 10_000 });

/* ------------------------------------------------------------------ */
/* Prediction                                                         */
/* ------------------------------------------------------------------ */

export interface FieldParams {
  chemistry: ChemistryKey;
  nx?: number;
  ny?: number;
  nz?: number;
  coolant_inlet_c?: number;
  coolant_flow_lpm?: number;
  ambient_c?: number;
  pack_current_a?: number;
  state_of_charge?: number;
}

export const getFieldPreview = (params: FieldParams) =>
  request<FieldResponse>('/predict/field/preview', {
    query: {
      chemistry: params.chemistry,
      nx: params.nx ?? 18,
      ny: params.ny ?? 12,
      nz: params.nz ?? 8,
      coolant_inlet_c: params.coolant_inlet_c ?? 26,
      coolant_flow_lpm: params.coolant_flow_lpm ?? BASELINE_CRUISE_FLOW_LPM,
      ambient_c: params.ambient_c ?? 34,
      pack_current_a: params.pack_current_a ?? DEFAULT_PACK_CURRENT_A,
      state_of_charge: params.state_of_charge ?? 0.5,
    },
  });

export interface FieldRequestBody {
  chemistry: ChemistryKey;
  coolant_inlet_c: number;
  coolant_flow_lpm: number;
  ambient_c: number;
  pack_current_a: number;
  state_of_charge: number;
  heat_watts?: number;
  mode?: OperatingMode;
}

export const postField = (grid: [number, number, number], op: FieldRequestBody) =>
  request<FieldResponse>('/predict/field', {
    method: 'POST',
    body: {
      operating_point: op,
      grid,
      include_coefficients: true,
    },
    timeoutMs: 30_000,
  });

export const postSingle = (
  point: [number, number, number],
  op: FieldRequestBody,
) =>
  request<{
    chemistry: ChemistryKey;
    point_m: number[];
    temperature_c: number;
    uncertainty_k: number;
    dT_dt_c_per_s: number;
    local_heat_flux_w_m2: number;
    timestamp: string;
  }>('/predict/single', {
    method: 'POST',
    body: { operating_point: op, point },
  });

export const getResiduals = (params: FieldParams) =>
  request<{
    chemistry: ChemistryKey;
    residuals: Record<string, number>;
    adaptive_weights: Record<string, number>;
    inference_latency_ms: number;
    within_latency_budget: boolean;
    exotherm_onset_c: number;
    margin_c: number;
    gradient_penalty_note: string;
    timestamp: string;
  }>('/predict/residuals', {
    query: {
      chemistry: params.chemistry,
      coolant_flow_lpm: params.coolant_flow_lpm ?? BASELINE_CRUISE_FLOW_LPM,
      coolant_inlet_c: params.coolant_inlet_c ?? 26,
      ambient_c: params.ambient_c ?? 34,
      pack_current_a: params.pack_current_a ?? DEFAULT_PACK_CURRENT_A,
      state_of_charge: params.state_of_charge ?? 0.5,
    },
  });

/* ------------------------------------------------------------------ */
/* Heatwave                                                           */
/* ------------------------------------------------------------------ */

export interface ForecastPayload {
  route: Array<{ lat: number; lon: number; t_s: number; speed_mps: number }>;
  weather: Array<{
    t_s: number;
    ambient_c: number;
    relative_humidity: number;
    solar_w_m2: number;
    wind_mps: number;
    road_surface_c: number;
  }>;
  chemistry: ChemistryKey;
  pack_heat_w: number;
  state_of_charge: number;
}

export const postForecast = (payload: ForecastPayload) =>
  request<{
    zones: HeatwaveZone[];
    schedule: ActuationEvent[];
    total_ramp_distance_lpm: number;
    max_predicted_cell_temp_c: number;
    advisory: string;
  }>('/heatwave/forecast', { method: 'POST', body: payload, timeoutMs: 30_000 });

export const getExplain = (chemistry: ChemistryKey, index = 0) =>
  request<{
    chemistry: ChemistryKey;
    exotherm_onset_c: number;
    decision: Record<string, number | string | null>;
    advisory: string;
    decision_log: Record<string, number>;
    zones: Array<Record<string, unknown>>;
    timestamp: string;
  }>('/heatwave/explain', {
    query: { chemistry, index },
    timeoutMs: 20_000,
  });

export const getAmbientSeries = (chemistry: ChemistryKey) =>
  request<{
    chemistry: ChemistryKey;
    t_s: number[];
    raw_ambient_c: number[];
    effective_ambient_c: number[];
    solar_overshoot_k: number[];
    pack_ambient_c: number[];
    comfort_ceiling_c: number;
    exotherm_onset_c: number;
    note: string;
  }>('/heatwave/ambient', { query: { chemistry, n: 120, duration_s: 300 } });

/* ------------------------------------------------------------------ */
/* Simulation                                                         */
/* ------------------------------------------------------------------ */

export const getScenario = (chemistry: ChemistryKey, durationS = 240, n = 120) =>
  request<{
    chemistry: ChemistryKey;
    duration_s: number;
    dt_s: number;
    route: Array<{ lat: number; lon: number; t_s: number; speed_mps: number }>;
    weather: Array<{
      t_s: number;
      ambient_c: number;
      relative_humidity: number;
      solar_w_m2: number;
      wind_mps: number;
      road_surface_c: number;
    }>;
    description: string;
  }>('/simulate/scenario', {
    query: { chemistry, duration_s: durationS, n },
    timeoutMs: 15_000,
  });

export interface TrajectoryPayload {
  chemistry: ChemistryKey;
  duration_s: number;
  dt_s: number;
  route: Array<{ lat: number; lon: number; t_s: number; speed_mps: number }>;
  weather: Array<{
    t_s: number;
    ambient_c: number;
    relative_humidity: number;
    solar_w_m2: number;
    wind_mps: number;
    road_surface_c: number;
  }>;
  base_current_a: number;
  base_flow_lpm: number;
  initial_cell_temp_c: number;
  initial_soc: number;
  enable_anticipatory: boolean;
  enable_learned_field: boolean;
  seed?: number;
}

export const postTrajectory = (payload: TrajectoryPayload) =>
  request<SimulationResponse>('/simulate/trajectory', {
    method: 'POST',
    body: payload,
    timeoutMs: 60_000,
  });

export const postLiveTick = (chemistry: ChemistryKey, dtS = 0.5, reset = false) =>
  request<TelemetryFrame>('/simulate/live', {
    method: 'POST',
    query: { chemistry, dt_s: dtS, reset },
  });

export const postSeedRun = (
  chemistry: ChemistryKey,
  steps = 120,
  dtS = 1,
  seed = 1337,
  usePinn = true,
) =>
  request<{
    chemistry: ChemistryKey;
    frames: TelemetryFrame[];
    summary: Record<string, number>;
    timestamp: string;
  }>('/demo/seed-run', {
    method: 'POST',
    query: { chemistry, steps, dt_s: dtS, seed, use_pinn: usePinn },
    timeoutMs: 45_000,
  });

/* ------------------------------------------------------------------ */
/* Control                                                            */
/* ------------------------------------------------------------------ */

export const setFlow = (
  chemistry: ChemistryKey,
  flowLpm: number,
  mode: 'manual' | 'auto' | 'anticipatory' | 'failsafe' = 'auto',
  operatorId = 'dashboard',
) =>
  request<FlowCommandResponse>('/control/flow', {
    method: 'POST',
    body: { chemistry, flow_lpm: flowLpm, mode, operator_id: operatorId },
  });

export const getControlState = () =>
  request<{
    chemistries: Record<
      string,
      {
        applied_flow_lpm: number;
        mode: string;
        reynolds: number;
        h_cold_w_m2k: number;
        heat_rejection_capacity_w: number;
        trip_state: string;
        trip_reason: string;
        derate_factor: number;
        last_operator: string;
      }
    >;
    recent_commands: Array<Record<string, unknown>>;
    loop_limits: Record<string, number>;
    timestamp: string;
  }>('/control/state');

export const resetSafety = (chemistry: ChemistryKey) =>
  request<{
    chemistry: ChemistryKey;
    /** Supervisor tier before this reset. */
    previous_state: TripState;
    /** Supervisor tier after the reset -- always `'ok'` on a latched-trip clear. */
    trip_state: TripState;
    reason: string;
    precondition: string;
    timestamp: string;
  }>('/control/safety/reset', { method: 'POST', query: { chemistry } });

/* ------------------------------------------------------------------ */
/* NASA                                                               */
/* ------------------------------------------------------------------ */

export const getCalibration = () => request<NasaCalibration>('/nasa/calibration');

export const recalibrate = () =>
  request<NasaCalibration>('/nasa/calibrate', { method: 'POST', timeoutMs: 60_000 });

export const getDatasetStats = () =>
  request<{
    describe: Record<string, unknown>;
    chemistry_inferred: string;
    window_len: number;
    stride: number;
    stats: Record<string, { mean: number; std: number }>;
    first_cycles: Array<Record<string, unknown>>;
  }>('/nasa/dataset/stats');

export const getNasaProbe = () =>
  request<{
    root: string;
    root_exists: boolean;
    root_is_dir: boolean;
    n_mat_files: number;
    files: Array<Record<string, unknown>>;
  }>('/nasa/probe');

/* ------------------------------------------------------------------ */
/* Demo / study                                                       */
/* ------------------------------------------------------------------ */

export const getBootstrap = (chemistry: ChemistryKey) =>
  request<BootstrapResponse>('/demo/bootstrap', {
    query: { chemistry },
    timeoutMs: 45_000,
  });

export const getComparison = () =>
  request<ComparisonResponse>('/demo/comparison', { timeoutMs: 45_000 });

export const getAblation = () =>
  request<Record<string, unknown>>('/demo/ablation', { timeoutMs: 15_000 });

/* ------------------------------------------------------------------ */
/* WebSocket                                                          */
/* ------------------------------------------------------------------ */

/**
 * Resolve the WebSocket URL for the telemetry stream.
 *
 * The socket connects **directly** to the backend rather than through a
 * Next.js rewrite. Next.js rewrites proxy HTTP and validate that `destination`
 * begins with `/`, `http://` or `https://`; a `ws://` destination is rejected
 * at build time with "Invalid rewrite found", so the `/ws` route in
 * `next.config.mjs` never worked and `next build` failed outright. Rewrites
 * cannot carry an `Upgrade: websocket` header in any case -- a
 * browser-initiated upgrade request is not proxied by the Next dev server.
 *
 * Connecting directly is safe: the backend enables CORS (see
 * `backend/main.py`), and WebSocket handshakes are not subject to the preflight
 * rules that would otherwise block a cross-origin upgrade.
 */
export function telemetryWsUrl(chemistry: ChemistryKey): string {
  if (typeof window === 'undefined') return '';
  const backend =
    process.env.NEXT_PUBLIC_BACKEND_WS_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'ws://127.0.0.1:8000';
  const base = backend.replace(/\/$/, '').replace(/^http/, 'ws');
  return `${base}/api/simulate/ws/telemetry?chemistry=${chemistry}`;
}

/** Commands accepted by the telemetry socket. */
export type WsCommand =
  | { action: 'ping' }
  | { action: 'reset'; initial_temp_c?: number; initial_soc?: number }
  | { action: 'chemistry'; chemistry: ChemistryKey }
  | { action: 'set_flow'; flow_lpm: number };