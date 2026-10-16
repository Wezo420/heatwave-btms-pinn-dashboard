'use client';

/**
 * `NasaCalibrationCard` -- provenance for the internal-resistance model.
 *
 * This card exists to make one specific claim checkable: that the ``I^2 R``
 * term driving every thermal result is calibrated against the NASA PCoE
 * run-to-failure dataset rather than assumed. Two design points matter:
 *
 * 1. **The synthetic-fallback banner is unmissable.** When the real ``.mat``
 *    files are absent the loader falls back to a labelled synthetic generator.
 *    That fallback is legitimate for a demo but must never be mistaken for the
 *    real dataset, so when it is active the card switches to a warning frame
 *    and states the remedy.
 * 2. **Error bars are drawn for E_a and R_0.** The standard errors come out of
 *    the weighted least-squares fit and are shown as explicit uncertainty, not
 *    folded into a single point estimate. ``E_a`` is the quantity that governs
 *    how fast ``R`` -- and therefore every Joule-heat prediction -- grows with
 *    temperature, so its uncertainty is a genuine safety-relevant number.
 */

import { Database, FileWarning, RefreshCw } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { cn, fmt, fmtParams } from '@/lib/utils';
import * as api from '@/lib/api';
import { useCallback, useState } from 'react';

interface Props {
  className?: string;
  onRecalibrate?: () => void;
}

export default function NasaCalibrationCard({ className, onRecalibrate }: Props) {
  const calibration = useStore((s) => s.calibration);
  const modelInfo = useStore((s) => s.modelInfo);
  const setCalibration = useStore((s) => s.setCalibration);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const recalibrate = useCallback(async () => {
    setBusy(true);
    setNote(null);
    try {
      const data = await api.recalibrate();
      setCalibration(data);
      onRecalibrate?.();
      setNote('Recalibration complete.');
    } catch (err) {
      setNote(
        err instanceof api.ApiError ? `${err.message} [${err.tracebackId ?? '-'}]` : String(err),
      );
    } finally {
      setBusy(false);
    }
  }, [setCalibration, onRecalibrate]);

  if (!calibration) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 text-sm text-slate-500',
          className,
        )}
        style={{ height: 180 }}
      >
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" aria-hidden />
        Loading NASA calibration report...
      </div>
    );
  }

  const agg = calibration.aggregate;
  const eaErr = agg.ea_stderr_j_mol ?? 0;
  const r0Err = agg.r0_stderr_ohm ?? 0;

  return (
    <div
      className={cn(
        'rounded-xl border bg-slate-900/40 p-3',
        calibration.synthetic_fallback ? 'border-amber-500/40 bg-amber-950/10' : 'border-slate-800',
        className,
      )}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-200">
          <Database className="h-4 w-4 text-sky-400" aria-hidden />
          NASA battery dataset calibration
        </h3>
        <button
          onClick={recalibrate}
          disabled={busy}
          title="Refit R_int from the .mat files on disk"
          className="rounded border border-slate-700 px-1.5 py-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-50"
        >
          <RefreshCw className={cn('h-3 w-3', busy && 'animate-spin')} aria-hidden />
        </button>
      </div>

      {calibration.synthetic_fallback && (
        <div className="mb-2 flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-950/30 p-2">
          <FileWarning className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" aria-hidden />
          <p className="text-[10px] leading-snug text-amber-200/90">
            <strong className="text-amber-300">Synthetic fallback active.</strong> No NASA{' '}
            <code>.mat</code> files were found under{' '}
            <code className="rounded bg-black/30 px-1">{calibration.dataset_root}</code>. The
            loader generated a deterministic run-to-failure dataset matching the published
            NASA statistics. Place the real <code>B00xx.mat</code> files there (or set{' '}
            <code>NASA_BATTERY_DATA_ROOT</code>) and press the refresh button to refit.
          </p>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
        <Metric
          label="R_0 at 25 degC"
          value={`${fmt((agg.r0_ohm ?? 0) * 1000, 2)} +/- ${fmt(r0Err * 1000, 2)}`}
          unit="mOhm"
        />
        <Metric
          label="Activation energy"
          value={`${fmt((agg.ea_j_mol ?? 0) / 1000, 1)} +/- ${fmt(eaErr / 1000, 1)}`}
          unit="kJ/mol"
          accent="text-violet-300"
        />
        <Metric label="RMSE in ln(R)" value={fmt(agg.rmse_log_r, 4)} unit="" accent="text-emerald-300" />
        <Metric label="Fit bins" value={fmt(agg.n_bins, 0)} unit="" />
        <Metric
          label="Loop m-dot cp"
          value={fmt(calibration.coolant_heat_capacity_w_k, 0)}
          unit="W/K"
          accent="text-coolant"
        />
        <Metric
          label="PINN parameters"
          value={modelInfo ? fmtParams(modelInfo.total_parameters) : '--'}
          unit=""
        />
      </dl>

      <div className="mt-2.5 border-t border-slate-800 pt-2">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          Per-cell capacity fade
        </div>
        <div className="max-h-24 overflow-y-auto">
          <table className="w-full text-[10px]">
            <thead className="sticky top-0 bg-slate-900/95">
              <tr className="text-slate-500">
                <th className="py-0.5 text-left font-medium">Cell</th>
                <th className="py-0.5 text-right font-medium">Cycles</th>
                <th className="py-0.5 text-right font-medium">Fade %</th>
                <th className="py-0.5 text-right font-medium">EOL %</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {calibration.cells.map((c) => (
                <tr key={c.cell_id} className="border-t border-slate-800/60">
                  <td className="py-0.5 text-slate-300">{c.cell_id}</td>
                  <td className="py-0.5 text-right text-slate-400">{c.cycles_available}</td>
                  <td className="py-0.5 text-right text-amber-300">{fmt(c.capacity_fade_pct, 1)}</td>
                  <td className="py-0.5 text-right text-slate-400">{fmt(c.eol_capacity_pct, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="mt-2 border-t border-slate-800 pt-1.5 text-[10px] leading-snug text-slate-600">
        {calibration.notes}
      </p>

      {note && <p className="mt-1 text-[10px] text-slate-400">{note}</p>}

      <details className="mt-2">
        <summary className="cursor-pointer text-[10px] font-medium text-slate-500 hover:text-slate-300">
          Regression form
        </summary>
        <pre className="mt-1 overflow-x-auto rounded border border-slate-800 bg-slate-950/70 p-2 font-mono text-[9px] leading-relaxed text-slate-400">
{`ln R(T, z) = ln R0
          - (Ea/Rg)(1/T - 1/298.15)
          + c2 z^2 + c1 z

c2 = ${fmt(agg.soc_quad, 4)}
c1 = ${fmt(agg.soc_lin, 4)}
Rg  = 8.314462618 J/(mol K)`}
        </pre>
      </details>
    </div>
  );
}

function Metric({
  label,
  value,
  unit,
  accent = 'text-slate-200',
}: {
  label: string;
  value: string;
  unit: string;
  accent?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-1">
      <dt className="truncate text-slate-500">{label}</dt>
      <dd className={cn('shrink-0 font-mono font-semibold', accent)}>
        {value}
        {unit && <span className="ml-0.5 text-[9px] font-normal text-slate-500">{unit}</span>}
      </dd>
    </div>
  );
}