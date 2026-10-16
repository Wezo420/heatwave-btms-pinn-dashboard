'use client';

/**
 * `Dashboard` -- the composition root for the client dashboard.
 *
 * Data-flow ownership (this is the part worth reading):
 *
 * ```
 *   useDashboardData  ── REST ──▶ catalogue, model info, PINN field,
 *                                advisory forecast, enforced rollout
 *   useTelemetry      ── WS   ──▶ live frames (2 Hz)
 *                                       │
 *                                       ▼
 *                                 Zustand store
 *                                       │
 *        ┌──────────────┬───────────────┼───────────────┬─────────────┐
 *        ▼              ▼               ▼               ▼             ▼
 *   StatTiles   TelemetryChart   CellTempChart   ThermalFieldMap  AlertOverlay
 *```
 *
 * The store is the single source of truth; every panel is a pure projection of
 * it. No panel fetches independently, which is what guarantees the chemistry
 * toggle updates all thresholds simultaneously (a per-panel fetch would race and
 * produce a display where the chart says NMC and the ceiling says LFP).
 */

import { useCallback, useEffect } from 'react';
import { AlertCircle, Loader2, RefreshCw } from 'lucide-react';
import Header from './Header';
import ChemistryToggle from './ChemistryToggle';
import StatTiles from './StatTiles';
import TelemetryChart from './TelemetryChart';
import CellTemperatureChart from './CellTemperatureChart';
import ThermalFieldMap from './ThermalFieldMap';
import HeatwaveAlertOverlay from './HeatwaveAlertOverlay';
import ControlPanel from './ControlPanel';
import NasaCalibrationCard from './NasaCalibrationCard';
import ChemistryComparisonCard from './ChemistryComparisonCard';
import ActuationTimeline from './ActuationTimeline';
import ResidualPanel from './ResidualPanel';
import { useDashboardData } from '@/hooks/useDashboardData';
import { useTelemetry } from '@/hooks/useTelemetry';
import type { TelemetryStatus } from '@/hooks/useTelemetry';
import { useStore, selectChemSpec } from '@/store/useStore';
import { cn, fmt } from '@/lib/utils';

export default function Dashboard() {
  const catalogue = useStore((s) => s.catalogue);
  const modelInfo = useStore((s) => s.modelInfo);
  const chemistry = useStore((s) => s.chemistry);
  const setChemistry = useStore((s) => s.setChemistry);
  const setConnected = useStore((s) => s.setConnected);
  const pushFrame = useStore((s) => s.pushFrame);
  const field = useStore((s) => s.field);
  const spec = useStore(selectChemSpec);

  const {
    bootstrapping,
    runningTrajectory,
    error,
    runTrajectory,
    refetchComparison,
  } = useDashboardData();

  /* -- live telemetry ------------------------------------------------ */
  const onFrame = useCallback(
    (frame: Parameters<typeof pushFrame>[0]) => pushFrame(frame),
    [pushFrame],
  );
  // The callback receives a TelemetryStatus ('idle' | 'connecting' | 'open' |
  // 'closed' | 'error'), not a boolean. Deriving the parameter type from
  // `setConnected` -- a Zustand boolean setter -- typed it as `boolean`, which
  // made `status === 'open'` a comparison between `boolean` and `string` that
  // TypeScript correctly rejects: the comparison can never be true as typed.
  const onStatus = useCallback(
    (status: TelemetryStatus) => setConnected(status === 'open'),
    [setConnected],
  );
  const { status: wsStatus, send } = useTelemetry({
    chemistry,
    enabled: true,
    onFrame,
    onStatusChange: onStatus,
  });

  // When the chemistry changes, tell the hub to rebuild its per-chemistry
  // simulator so the stream reflects the toggle immediately.
  useEffect(() => {
    send({ action: 'chemistry', chemistry });
  }, [chemistry, send]);

  // Latency readout: prefer the engine's own server-side measurement when the
  // field query has returned, since that excludes the client render cost.
  const latencyP95 = useStore((s) => s.latencyP95Ms);

  return (
    <div className="flex min-h-screen flex-col">
      <Header wsStatus={wsStatus} latencyP95={latencyP95} />

      {error && (
        <div className="border-b border-rose-800/60 bg-rose-950/40 px-4 py-2">
          <div className="mx-auto flex max-w-[1600px] items-start gap-2 text-xs text-rose-200">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <div>
              <strong>Backend error:</strong> {error}
              <p className="mt-0.5 text-[10px] text-rose-200/70">
                The backend returns a <code>traceback_id</code> with every failure; quote it
                when checking <code>logs/backend.log</code> or the auto-heal report in{' '}
                <code>logs/auto_heal.log</code>.
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-4">
        {/* -- Row 1: chemistry selector + chemistry facts ---------------- */}
        <section aria-labelledby="chem-heading" className="mb-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 id="chem-heading" className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Cell chemistry (drives model conditioning, thresholds and control)
              </h2>
              <ChemistryToggle
                chemistries={catalogue?.chemistries ?? []}
                value={chemistry}
                onChange={setChemistry}
                disabled={bootstrapping}
              />
            </div>

            <div className="panel min-w-[280px] flex-1 px-3 py-2">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                {spec.key.toUpperCase()} material summary
              </div>
              <dl className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                <Fact label="Cathode" value={spec.cathodes} />
                <Fact label="Anode" value={spec.anode} />
                <Fact label="Specific heat" value={`${fmt(spec.cp_j_kgk, 0)} J/(kg·K)`} />
                <Fact label="Thermal diffusivity" value={`${fmt(spec.thermal_diffusivity_m2_s * 1e6, 3)} mm²/s`} />
                <Fact label="Energy density" value={`${fmt(spec.energy_density_wh_kg, 0)} Wh/kg`} />
                <Fact label="Nominal capacity" value={`${fmt(spec.nominal_capacity_ah, 2)} Ah`} />
                <Fact label="Exotherm onset" value={`${fmt(spec.thermal_runaway_onset_c, 0)} °C`} accent="text-rose-300" />
                <Fact
                  label="Coolant"
                  value={`${fmt(spec.coolant.cp_j_kgk, 0)} J/(kg·K), k=${fmt(spec.coolant.k_w_mk, 3)}`}
                />
              </dl>
              <p className="mt-1.5 border-t border-slate-800 pt-1.5 text-[10px] leading-snug text-slate-500">
                {spec.notes}
              </p>
            </div>
          </div>
        </section>

        {/* -- Row 2: KPIs ------------------------------------------------ */}
        <StatTiles className="mb-4" />

        {/* -- Row 3: charts --------------------------------------------- */}
        <div className="grid gap-4 xl:grid-cols-3">
          <div className="xl:col-span-2">
            <TelemetryChart className="mb-4" />
            <div className="grid gap-4 md:grid-cols-2">
              <CellTemperatureChart />
              <ThermalFieldMap field={field} />
            </div>
          </div>

          <aside className="space-y-4">
            <ControlPanel onRunRoute={() => void runTrajectory()} running={runningTrajectory} />
            <NasaCalibrationCard />
            <ChemistryComparisonCard onLoad={refetchComparison} loading={runningTrajectory} />
          </aside>
        </div>

        {/* -- Row 4: schedule + residual audit --------------------------- */}
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          <ActuationTimeline />
          <ResidualPanel />
        </div>

        {/* -- Footer provenance ----------------------------------------- */}
        <footer className="mt-4 rounded-xl border border-slate-800/60 bg-slate-900/30 px-4 py-3 text-[10px] leading-relaxed text-slate-500">
          <p>
            <strong className="text-slate-400">Method.</strong> The temperature field is a
            dual-chemistry Physics-Informed Neural Network: a {modelInfo?.n_hidden_layers ?? 5}
            -layer SIREN trunk with random Fourier features and FiLM chemistry conditioning
            (conditioning vector dimension {modelInfo?.conditioning_dim ?? 12}), trained on a
            weighted sum of an energy-equation residual (anisotropic conduction with an Arrhenius
            Joule source), a reduced Navier-Stokes momentum residual from a stream-function
            velocity ansatz, conjugate interface conditions, and labelled telemetry. Loss terms
            are rebalanced each step by a gradient-norm penalty (EMA decay{' '}
            {modelInfo?.gradient_penalty.ema_decay ?? 0.999}), which is what keeps the PDE and
            boundary terms from starving one another.
          </p>
          <p className="mt-1.5">
            <strong className="text-slate-400">Provenance.</strong> Internal resistance is
            calibrated from the NASA PCoE run-to-failure discharge cycles via a weighted
            least-squares fit of ln R(T, SOC); the synthetic fallback is flagged explicitly in the
            calibration card. Anticipatory control is a receding-horizon feasibility sweep over
            the reduced 12-cell model, rate-limited to {fmt(0.22, 2)} L/min/s, with a three-tier
            fail-safe ({'ok'} &rarr; {'derate'} &rarr; {'trip'}) latching below a
            chemistry-specific margin floor. Predicted (advisory) and commanded (enforced)
            schedules are shown separately throughout and are never merged.
          </p>
        </footer>
      </main>

      {/* -- Anticipatory overlay ---------------------------------------- */}
      <HeatwaveAlertOverlay />
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Fact({ label, value, accent = 'text-slate-200' }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="shrink-0 text-slate-500">{label}</dt>
      <dd className={cn('truncate text-right font-mono font-medium', accent)} title={value}>
        {value}
      </dd>
    </div>
  );
}

/* ------------------------------------------------------------------ */

/**
 * `DashboardLoader` -- shown while the bootstrap request is in flight.
 *
 * A skeleton rather than a spinner: the panel geometry is already known from the
 * layout, so showing it immediately avoids a large reflow when the data lands,
 * and it makes the ~1-3 s backend warm-up (torch import + network construction)
 * legible as "loading the model" rather than "the page is broken".
 */
export function DashboardSkeleton() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <Loader2 className="mx-auto h-8 w-8 animate-spin text-sky-400" aria-hidden />
        <p className="mt-3 text-sm text-slate-400">
          Loading PINN weights and chemistry catalogue...
        </p>
        <p className="mt-1 text-xs text-slate-600">
          First request initialises torch and runs a CUDA probe; 1-3 s is expected.
        </p>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 inline-flex items-center gap-1.5 rounded border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
        >
          <RefreshCw className="h-3 w-3" aria-hidden />
          Retry
        </button>
      </div>
    </div>
  );
}