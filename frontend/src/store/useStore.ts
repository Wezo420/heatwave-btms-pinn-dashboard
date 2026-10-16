/**
 * Global dashboard state (Zustand).
 *
 * Design decisions worth stating, because they are the ones that make this
 * dashboard *research-grade* rather than a demo:
 *
 * 1. **Advisory vs enforced are separate state, never merged.**  `advisorySchedule`
 *    holds what the planner *recommends*; `frames` holds what the controller
 *    *commanded*.  Merging them would let a predicted ramp be mistaken for an
 *    applied one -- the single most dangerous conflation in a predictive BMS UI.
 *
 * 2. **Chemistry is a first-class state field, not a derived value.**  Every
 *    chart, threshold and physics constant is re-derived from it, so switching
 *    NMC <-> LFP updates every visual simultaneously and none of them can drift
 *    out of sync with the chemistry.
 *
 * 3. **History is bounded and pre-aggregated.**  We keep at most `maxFrames`
 *    frames and downsample to a fixed bucket size for Recharts, so a 20-minute
 *    session does not degrade the render loop.
 */

import { create } from 'zustand';
import type {
  ActuationEvent,
  BootstrapResponse,
  CatalogueResponse,
  ChemistryKey,
  ComparisonResponse,
  FieldResponse,
  HeatwaveZone,
  ModelInfo,
  NasaCalibration,
  SimulationResponse,
  TelemetryFrame,
  TripState,
} from '@/lib/types';

const MAX_FRAMES = 900;

export interface OperatingInputs {
  ambientC: number;
  packCurrentA: number;
  stateOfCharge: number;
  coolantInletC: number;
  baseFlowLpm: number;
  mode: 'inference' | 'warmup' | 'cooldown' | 'fastcharge' | 'load';
}

export interface DashboardState {
  /* -- catalogue / model ------------------------------------------- */
  chemistry: ChemistryKey;
  catalogue: CatalogueResponse | null;
  modelInfo: ModelInfo | null;
  bootstrap: BootstrapResponse | null;

  /* -- live telemetry ----------------------------------------------- */
  connected: boolean;
  frames: TelemetryFrame[];
  latest: TelemetryFrame | null;
  latencyEmaMs: number;
  latencyP95Ms: number;
  serverTimeMs: number | null;

  /* -- predictions -------------------------------------------------- */
  field: FieldResponse | null;
  fieldLoading: boolean;
  fieldError: string | null;

  /* -- anticipatory heatwave ---------------------------------------- */
  anticipatoryEnabled: boolean;
  advisorySchedule: ActuationEvent[];
  advisoryZones: HeatwaveZone[];
  enforcedActuation: ActuationEvent[];
  enforcedZones: HeatwaveZone[];
  trajectory: SimulationResponse | null;
  trajectoryLoading: boolean;

  /* -- NASA --------------------------------------------------------- */
  calibration: NasaCalibration | null;

  /* -- study -------------------------------------------------------- */
  comparison: ComparisonResponse | null;

  /* -- controls ----------------------------------------------------- */
  inputs: OperatingInputs;
  tripState: TripState;
  alertBanner: { title: string; body: string; severity: 'info' | 'warn' | 'critical' } | null;

  /* -- actions ------------------------------------------------------ */
  setChemistry: (key: ChemistryKey) => void;
  setBootstrap: (data: BootstrapResponse) => void;
  setCatalogue: (data: CatalogueResponse) => void;
  setModelInfo: (data: ModelInfo) => void;
  setConnected: (connected: boolean) => void;
  pushFrame: (frame: TelemetryFrame, serverMs?: number) => void;
  setField: (field: FieldResponse | null) => void;
  setFieldLoading: (loading: boolean) => void;
  setFieldError: (error: string | null) => void;
  setAnticipatoryEnabled: (enabled: boolean) => void;
  setAdvisory: (schedule: ActuationEvent[], zones: HeatwaveZone[]) => void;
  setTrajectory: (data: SimulationResponse | null) => void;
  setTrajectoryLoading: (loading: boolean) => void;
  setCalibration: (data: NasaCalibration) => void;
  setComparison: (data: ComparisonResponse | null) => void;
  setInput: <K extends keyof OperatingInputs>(key: K, value: OperatingInputs[K]) => void;
  setTripState: (state: TripState) => void;
  setAlert: (alert: DashboardState['alertBanner']) => void;
  reset: () => void;
}

const DEFAULT_INPUTS: OperatingInputs = {
  ambientC: 38,
  packCurrentA: 180,
  stateOfCharge: 0.65,
  coolantInletC: 26,
  baseFlowLpm: 6,
  mode: 'inference',
};

export const useStore = create<DashboardState>((set, get) => ({
  chemistry: 'nmc',
  catalogue: null,
  modelInfo: null,
  bootstrap: null,

  connected: false,
  frames: [],
  latest: null,
  latencyEmaMs: 0,
  latencyP95Ms: 0,
  serverTimeMs: null,

  field: null,
  fieldLoading: false,
  fieldError: null,

  anticipatoryEnabled: true,
  advisorySchedule: [],
  advisoryZones: [],
  enforcedActuation: [],
  enforcedZones: [],
  trajectory: null,
  trajectoryLoading: false,

  calibration: null,
  comparison: null,

  inputs: { ...DEFAULT_INPUTS },
  tripState: 'ok',
  alertBanner: null,

  setChemistry: (key) => set({ chemistry: key, trajectory: null }),

  setBootstrap: (data) =>
    set({
      chemistry: data.chemistry,
      catalogue: data.catalogue,
      modelInfo: data.model,
      field: data.field,
    }),

  setCatalogue: (catalogue) => set({ catalogue }),
  setModelInfo: (modelInfo) => set({ modelInfo }),

  setConnected: (connected) => set({ connected }),

  pushFrame: (frame, serverMs) => {
    const state = get();
    const frames = [...state.frames, frame];
    const trimmed = frames.length > MAX_FRAMES ? frames.slice(-MAX_FRAMES) : frames;

    // Latency = the wall-clock gap between consecutive frames. The backend
    // ticks at 0.5 s, so the transport overhead is (gap - 500 ms); we keep the
    // raw gap and label it "frame interval" elsewhere to avoid subtracting a
    // constant that drifts with load.
    const prev = state.latest;
    let intervalMs = state.latencyEmaMs;
    if (prev) {
      const gap = (frame.t_s - prev.t_s) * 1000;
      if (Number.isFinite(gap) && gap > 0) {
        intervalMs = state.latencyEmaMs ? 0.85 * state.latencyEmaMs + 0.15 * gap : gap;
      }
    }

    const trip: TripState = frame.trip_state;
    const alert =
      trip === 'trip'
        ? {
            title: 'Thermal fail-safe TRIPPED',
            body: `Cell temperature reached ${frame.max_cell_temp_c.toFixed(1)} degC. Coolant forced to maximum. Operator reset required.`,
            severity: 'critical' as const,
          }
        : trip === 'derate'
          ? {
              title: 'Pack current derated',
              body: `Exotherm margin fell to ${frame.thermal_runaway_margin_c.toFixed(1)} K. Pack current reduced to 70%.`,
              severity: 'warn' as const,
            }
          : null;

    set({
      frames: trimmed,
      latest: frame,
      latencyEmaMs: intervalMs,
      latencyP95Ms: computeP95(trimmed.map((f) => (f.t_s - trimmed[0].t_s) * 1000)),
      serverTimeMs: serverMs ?? state.serverTimeMs,
      tripState: trip,
      alertBanner: alert ?? state.alertBanner,
    });
  },

  setField: (field) => set({ field, fieldError: null }),
  setFieldLoading: (fieldLoading) => set({ fieldLoading }),
  setFieldError: (fieldError) => set({ fieldError }),

  setAnticipatoryEnabled: (anticipatoryEnabled) => set({ anticipatoryEnabled }),

  setAdvisory: (advisorySchedule, advisoryZones) =>
    set({ advisorySchedule, advisoryZones }),

  setTrajectory: (trajectory) =>
    set({
      trajectory,
      enforcedActuation: trajectory?.actuation ?? [],
      enforcedZones: trajectory?.heatwave_zones ?? [],
    }),

  setTrajectoryLoading: (trajectoryLoading) => set({ trajectoryLoading }),
  setCalibration: (calibration) => set({ calibration }),
  setComparison: (comparison) => set({ comparison }),

  setInput: (key, value) =>
    set({ inputs: { ...get().inputs, [key]: value } as OperatingInputs }),

  setTripState: (tripState) => set({ tripState }),
  setAlert: (alertBanner) => set({ alertBanner }),

  reset: () =>
    set({
      frames: [],
      latest: null,
      trajectory: null,
      tripState: 'ok',
      alertBanner: null,
      inputs: { ...DEFAULT_INPUTS },
    }),
}));

function computeP95(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(Math.floor(0.95 * (sorted.length - 1)), sorted.length - 1);
  return sorted[idx];
}

/* ------------------------------------------------------------------ */
/* Derived selectors                                                  */
/* ------------------------------------------------------------------ */

/** Chemistry metadata for the current selection, or a safe fallback. */
export function selectChemSpec(state: DashboardState) {
  const found = state.catalogue?.chemistries.find((c) => c.key === state.chemistry);
  return (
    found ?? {
      key: state.chemistry,
      label: state.chemistry.toUpperCase(),
      cathodes: '--',
      anode: '--',
      cp_j_kgk: state.chemistry === 'nmc' ? 1040 : 1150,
      thermal_runaway_onset_c: state.chemistry === 'nmc' ? 210 : 270,
      internal_resistance_reference_ohm: state.chemistry === 'nmc' ? 0.0135 : 0.0182,
      activation_energy_j_mol: state.chemistry === 'nmc' ? 23500 : 19800,
      nominal_capacity_ah: state.chemistry === 'nmc' ? 4.85 : 5.2,
      energy_density_wh_kg: state.chemistry === 'nmc' ? 255 : 165,
      thermal_diffusivity_m2_s: 0,
      coolant: { density_kg_m3: 1051, cp_j_kgk: 3400, k_w_mk: 0.505, mu_pa_s: 0.00205, prandtl: 13.8 },
      channels: { n_channels: 24, channel_width_mm: 2.2, channel_depth_mm: 3.5, length_mm: 360, width_mm: 230 },
      notes: 'Catalogue not loaded; using documented defaults.',
      recommended_max_cell_temp_c: state.chemistry === 'nmc' ? 42 : 47,
      default_flow_setpoint_lpm: 6,
    }
  );
}

/** Comfort ceiling for the active chemistry. */
export function selectComfortCeiling(state: DashboardState): number {
  return state.bootstrap?.safety.comfort_ceiling_c ?? selectChemSpec(state).recommended_max_cell_temp_c;
}

/** Thermal-runaway onset for the active chemistry (210 / 270 degC). */
export function selectRunawayOnset(state: DashboardState): number {
  return selectChemSpec(state).thermal_runaway_onset_c;
}