'use client';

/**
 * `ThermalFieldMap` -- 3-D PINN field visualisation.
 *
 * Renders the (nx x ny x nz) temperature tensor as a stack of 2-D cross-sections
 * along x, coloured on the shared thermal ramp. A 3-D surface would be prettier
 * and less informative: the physically meaningful features of this problem --
 * the inlet-to-outlet gradient and the channel-to-cell gradient -- are planar,
 * so slices show them without perspective distortion.
 *
 * Cross-section selector: the user picks the x-slice. We default to the
 * *inlet-most* slice (i = 1), because the worst case in a microchannel pack is
 * at the coolant entry where the coolant is coldest and the Joule load is
 * unchanged. Showing the mid-pack slice by default would hide the constraint
 * that binds.
 *
 * Rendering: each cell is an absolutely-positioned div inside a CSS grid, sized
 * to a fixed pixel pitch. This keeps 18x12 = 216 nodes well under the layout
 * budget and avoids canvas, which would be inaccessible to screen readers and
 * untestable without a DOM shim.
 */

import { useMemo, useState } from 'react';
import { Layers, Thermometer } from 'lucide-react';
import { useStore, selectChemSpec } from '@/store/useStore';
import { cn, fmt, thermalColor } from '@/lib/utils';
import type { FieldResponse } from '@/lib/types';

interface Props {
  field: FieldResponse | null;
  className?: string;
  /** Pixel pitch per grid node. */
  pitch?: number;
}

export default function ThermalFieldMap({ field, className, pitch = 13 }: Props) {
  const [sliceIndex, setSliceIndex] = useState(1);
  const spec = useStore(selectChemSpec);
  const loading = useStore((s) => s.fieldLoading);
  const error = useStore((s) => s.fieldError);

  /* -- colour domain -------------------------------------------------- */
  const domain = useMemo<[number, number]>(() => {
    if (!field) return [20, 60];
    const lo = Math.floor(field.min_temp_c / 5) * 5;
    const hi = Math.ceil(Math.max(field.max_temp_c, lo + 20) / 5) * 5;
    return [lo, hi];
  }, [field]);

  // The no-field early return must declare *every* key the populated branch
  // returns. Omitting `zs` made TypeScript infer a union whose `zs` is
  // optional, and `slice.zs[k]` on line 163 then failed to compile even though
  // it is unreachable while `field` is null. Declaring the shape explicitly
  // once, and returning the same shape from both branches, removes the union.
  //
  // Hoisted to module scope so it is a stable reference: a fresh object literal
  // inside the component would be a new dependency on every render and would
  // defeat the memo entirely.
  const slice = useMemo((): {
    grid: number[][];
    xs: number[];
    ys: number[];
    zs: number[];
  } => {
    if (!field) return { grid: [], xs: [], ys: [], zs: [] };
    const [nx, ny] = field.grid;
    const i = Math.min(Math.max(sliceIndex, 0), nx - 1);
    const grid: number[][] = [];
    for (let j = 0; j < ny; j += 1) {
      const row: number[] = [];
      for (let k = 0; k < field.grid[2]; k += 1) {
        row.push(field.temperature_c[i]?.[j]?.[k] ?? Number.NaN);
      }
      grid.push(row);
    }
    return { grid, xs: field.axes.x_m, ys: field.axes.y_m, zs: field.axes.z_m };
  }, [field, sliceIndex]);

  if (error) {
    return (
      <div className={cn('rounded-xl border border-rose-800/60 bg-rose-950/30 p-4 text-sm text-rose-300', className)}>
        <strong>Field evaluation failed.</strong>
        <p className="mt-1 text-xs text-rose-200/80">{error}</p>
        <p className="mt-2 text-[11px] text-slate-500">
          The auto-heal monitor watches for this class of failure; check
          <code className="mx-1 rounded bg-slate-800 px-1">logs/auto_heal.log</code>
          for a healing hypothesis.
        </p>
      </div>
    );
  }

  if (!field) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 text-sm text-slate-500',
          className,
        )}
        style={{ height: 260 }}
      >
        Evaluating PINN field...
      </div>
    );
  }

  const [nx, ny, nz] = field.grid;

  return (
    <div className={cn('rounded-xl border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-200">
          <Layers className="h-4 w-4 text-sky-400" aria-hidden />
          3-D PINN temperature field
          {loading && (
            <span className="ml-1 h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" aria-label="computing" />
          )}
        </h3>
        <div className="flex items-center gap-2 text-[11px] font-mono text-slate-500">
          <span>
            {nx}&times;{ny}&times;{nz} grid
          </span>
          <span className="text-slate-700">|</span>
          <span>{fmt(field.inference_latency_ms, 1)} ms</span>
        </div>
      </div>

      <div className="flex gap-3">
        {/* Slice stack selector */}
        <div className="flex flex-col gap-0.5" role="radiogroup" aria-label="Streamwise slice">
          {Array.from({ length: nx }, (_, i) => {
            // inlet at the bottom of the stack, matching the x axis increasing rightward
            const label = ((nx - 1 - i) * (slice.xs[1] - slice.xs[0])).toFixed(0);
            const isActive = i === sliceIndex;
            return (
              <button
                key={i}
                role="radio"
                aria-checked={isActive}
                onClick={() => setSliceIndex(i)}
                className={cn(
                  'h-2.5 w-9 rounded-sm font-mono text-[8px] leading-none transition-colors',
                  isActive
                    ? 'bg-sky-400 text-slate-950'
                    : 'bg-slate-800 text-slate-500 hover:bg-slate-700',
                )}
                title={`x = ${label} mm`}
              >
                {nx - 1 - i === 0 ? 'IN' : nx - 1 - i}
              </button>
            );
          })}
        </div>

        {/* Slice grid */}
        <div className="flex-1">
          <div
            className="grid gap-[1px] rounded border border-slate-800 bg-slate-900/60 p-1"
            style={{
              gridTemplateColumns: `repeat(${ny}, ${pitch}px)`,
              gridAutoRows: `${pitch}px`,
              width: 'fit-content',
            }}
            role="img"
            aria-label={`Temperature cross-section at x = ${sliceIndex} of ${nx}. Max ${fmt(
              field.max_temp_c,
              1,
            )} degrees Celsius, min ${fmt(field.min_temp_c, 1)} degrees Celsius.`}
          >
            {slice.grid.flatMap((row, j) =>
              row.map((t, k) => (
                <div
                  key={`${j}-${k}`}
                  className="rounded-[1px] transition-colors duration-150"
                  style={{ background: Number.isFinite(t) ? thermalColor(t, domain) : '#334155' }}
                  title={`y = ${fmt(slice.ys[j] * 1000, 0)} mm, z = ${fmt(slice.zs[k] * 1000, 1)} mm: ${fmt(t, 2)} degC`}
                />
              )),
            )}
          </div>

          <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500">
            <span>
              y (across pack) &rarr; {ny} columns, {ny > 1 ? fmt((slice.ys[1] - slice.ys[0]) * 1000, 1) : '--'} mm pitch
            </span>
            <span>
              z (through thickness) &darr; {nz} rows, {fmt(COLD_PLATE_T, 1)} mm
            </span>
          </div>
        </div>
      </div>

      {/* Colour legend */}
      <div className="mt-3">
        <div className="flex items-center justify-between text-[10px] text-slate-500">
          <span className="flex items-center gap-1">
            <Thermometer className="h-3 w-3" aria-hidden />
            {fmt(domain[0], 0)}&deg;C
          </span>
          <span>
            Comfort ceiling {fmt(spec.recommended_max_cell_temp_c, 0)}&deg;C &middot; onset{' '}
            {fmt(spec.thermal_runaway_onset_c, 0)}&deg;C
          </span>
          <span>{fmt(domain[1], 0)}&deg;C</span>
        </div>
        <div
          className="mt-1 h-2 rounded-full"
          style={{
            background: `linear-gradient(90deg, ${Array.from({ length: 12 }, (_, i) =>
              thermalColor(domain[0] + ((domain[1] - domain[0]) * i) / 11, domain),
            ).join(', ')})`,
          }}
          role="img"
          aria-label={`Colour scale from ${domain[0]} to ${domain[1]} degrees Celsius`}
        />
      </div>

      {/* Physics diagnostics */}
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-800 pt-2 text-[11px] sm:grid-cols-3">
        <Stat label="Max cell" value={`${fmt(field.max_temp_c, 2)} C`} accent="text-rose-300" />
        <Stat label="Mean cell" value={`${fmt(field.mean_temp_c, 2)} C`} accent="text-slate-200" />
        <Stat label="Outlet cell" value={`${fmt(field.cell_outlet_c, 2)} C`} accent="text-orange-300" />
        <Stat label="Coolant out" value={`${fmt(field.coolant_outlet_c, 2)} C`} accent="text-sky-300" />
        <Stat label="Q rejected" value={`${fmt(field.heat_rejected_w, 1)} W`} accent="text-emerald-300" />
        <Stat
          label="Runaway margin"
          value={`${fmt(field.thermal_runaway_margin_c, 1)} K`}
          accent={
            field.thermal_runaway_margin_c < 45 ? 'text-rose-300' : 'text-emerald-300'
          }
        />
      </dl>

      {/* Residual audit */}
      <div className="mt-2 rounded border border-slate-800 bg-slate-950/50 px-2.5 py-1.5">
        <div className="flex items-center justify-between text-[10px]">
          <span className="font-semibold uppercase tracking-wider text-slate-500">
            Physics residual audit
          </span>
          <span
            className={cn(
              'rounded px-1.5 py-0.5 font-mono font-bold',
              field.residuals.is_converged
                ? 'bg-emerald-500/20 text-emerald-300'
                : 'bg-amber-500/20 text-amber-300',
            )}
          >
            {field.residuals.is_converged ? 'CONVERGED' : 'NOT CONVERGED'}
          </span>
        </div>
        <div className="mt-1 grid grid-cols-3 gap-x-2 font-mono text-[10px] sm:grid-cols-4">
          <ResidualCell label="PDE" value={field.residuals.pde_energy} />
          <ResidualCell label="BC" value={field.residuals.bc_inlet + field.residuals.bc_outlet} />
          <ResidualCell label="iface" value={field.residuals.bc_interface_flux} />
          <ResidualCell label="mom" value={field.residuals.pde_momentum} />
        </div>
        <p className="mt-1 text-[10px] leading-snug text-slate-600">
          These are the squared residuals of the governing equations at the returned
          field, not a training loss. A low value means the displayed temperature
          distribution actually satisfies conjugate heat transfer.
        </p>
      </div>
    </div>
  );
}

const COLD_PLATE_T = 12;

function Stat({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="flex items-baseline justify-between gap-1">
      <dt className="text-slate-500">{label}</dt>
      <dd className={cn('font-mono font-semibold', accent)}>{value}</dd>
    </div>
  );
}

function ResidualCell({ label, value }: { label: string; value: number }) {
  const finite = Number.isFinite(value);
  const mag = finite ? Math.abs(value) : Infinity;
  const tone =
    mag < 1e-2 ? 'text-emerald-400' : mag < 1 ? 'text-amber-400' : 'text-rose-400';
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-slate-600">{label}</span>
      <span className={cn('font-mono', tone)}>
        {finite ? (mag < 1e-3 ? mag.toExponential(1) : mag.toFixed(3)) : 'NaN'}
      </span>
    </div>
  );
}