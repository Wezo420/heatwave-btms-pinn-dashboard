'use client';

/**
 * `ChemistryComparisonCard` -- the NMC/LFP comparative stability study.
 *
 * Renders the side-by-side numbers from `GET /api/demo/comparison`, which runs
 * the *same* route and the *same* heat load through both chemistries. The card
 * is the visual anchor for the model card's "Metrics & Benchmarks" section, so
 * it is deliberately presented as an experiment report rather than a marketing
 * comparison: the pass/fail state against each chemistry's own comfort ceiling
 * and margin floor is shown explicitly, and the interpretation string comes
 * verbatim from the backend rather than being re-worded in the UI.
 */

import { useEffect } from 'react';
import { ArrowRight, GitCompare, Loader2 } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { cn, fmt } from '@/lib/utils';
import type { ComparisonEntry } from '@/lib/types';

interface Props {
  className?: string;
  onLoad: () => void;
  loading?: boolean;
}

export default function ChemistryComparisonCard({ className, onLoad, loading }: Props) {
  const comparison = useStore((s) => s.comparison);

  useEffect(() => {
    if (!comparison) onLoad();
  }, [comparison, onLoad]);

  if (loading && !comparison) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 text-sm text-slate-500',
          className,
        )}
        style={{ height: 160 }}
      >
        <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
        Running comparative stability study...
      </div>
    );
  }

  if (!comparison) {
    return (
      <button
        onClick={onLoad}
        className={cn(
          'flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-slate-700 bg-slate-900/40 py-8 text-sm text-slate-400 hover:border-slate-500 hover:text-slate-200',
          className,
        )}
      >
        <GitCompare className="h-4 w-4" aria-hidden />
        Run NMC vs LFP comparative stability study
      </button>
    );
  }

  const { nmc, lfp, delta } = comparison;

  return (
    <div className={cn('rounded-xl border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-200">
          <GitCompare className="h-4 w-4 text-violet-400" aria-hidden />
          NMC vs LFP -- identical route, identical heat load
        </h3>
        <span className="font-mono text-[10px] text-slate-600">deterministic (seed 1337)</span>
      </div>

      <div className="grid grid-cols-[1fr_auto_1fr] items-stretch gap-2">
        <Column entry={nmc} chemistry="NMC" />
        <div className="flex items-center">
          <ArrowRight className="h-4 w-4 text-slate-600" aria-hidden />
        </div>
        <Column entry={lfp} chemistry="LFP" />
      </div>

      <div className="mt-2.5 grid grid-cols-3 gap-2 border-t border-slate-800 pt-2 text-center">
        <Delta
          label="Peak cell (LFP - NMC)"
          value={`${delta.peak_cell_temp_c >= 0 ? '+' : ''}${fmt(delta.peak_cell_temp_c, 2)} K`}
          good={delta.peak_cell_temp_c <= 0}
        />
        <Delta
          label="Exotherm margin"
          value={`${delta.min_exotherm_margin_c >= 0 ? '+' : ''}${fmt(delta.min_exotherm_margin_c, 1)} K`}
          good={delta.min_exotherm_margin_c >= 0}
        />
        <Delta
          label="Peak coolant flow"
          value={`${delta.peak_commanded_flow_lpm >= 0 ? '+' : ''}${fmt(delta.peak_commanded_flow_lpm, 2)}`}
          good={delta.peak_commanded_flow_lpm <= 0}
          unit="L/min"
        />
      </div>

      <p className="mt-2 rounded border border-slate-800 bg-slate-950/50 px-2.5 py-2 text-[11px] leading-snug text-slate-400">
        {delta.interpretation}
      </p>

      <div className="mt-2 space-y-1">
        <PassRow
          label="Comfort ceiling respected"
          nmcOk={nmc.ceiling_respected}
          lfpOk={lfp.ceiling_respected}
        />
        <PassRow
          label="Exotherm margin floor respected"
          nmcOk={nmc.margin_respected}
          lfpOk={lfp.margin_respected}
        />
      </div>
    </div>
  );
}

function Column({ entry, chemistry }: { entry: ComparisonEntry; chemistry: string }) {
  const accent = chemistry === 'NMC' ? 'text-nmc' : 'text-lfp';
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2.5">
      <div className={cn('text-xs font-bold tracking-wide', accent)}>{chemistry}</div>
      <dl className="mt-1.5 space-y-0.5 text-[10px]">
        <Line label="Peak cell" value={`${fmt(entry.peak_cell_temp_c, 2)} C`} />
        <Line label="Ceiling" value={`${fmt(entry.comfort_ceiling_c, 0)} C`} />
        <Line
          label="Exotherm margin"
          value={`${fmt(entry.min_exotherm_margin_c, 1)} K`}
          accent={entry.margin_respected ? 'text-emerald-300' : 'text-rose-300'}
        />
        <Line label="Cp" value={`${fmt(entry.specific_heat_j_kgk, 0)} J/kgK`} />
        <Line label="R at 40 C" value={`${fmt(entry.r_int_at_40c_ohm * 1000, 2)} mOhm`} />
        <Line label="Ea" value={`${fmt(entry.arrhenius_slope_kj_mol, 1)} kJ/mol`} />
        <Line label="Peak flow" value={`${fmt(entry.peak_commanded_flow_lpm, 2)} L/min`} />
        <Line label="Zones" value={fmt(entry.n_zones, 0)} />
      </dl>
    </div>
  );
}

function Line({ label, value, accent = 'text-slate-200' }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-1">
      <dt className="truncate text-slate-500">{label}</dt>
      <dd className={cn('shrink-0 font-mono font-semibold', accent)}>{value}</dd>
    </div>
  );
}

function Delta({ label, value, good, unit = 'K' }: { label: string; value: string; good: boolean; unit?: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950/40 px-1 py-1.5">
      <div className="text-[9px] leading-tight text-slate-500">{label}</div>
      <div className={cn('font-mono text-sm font-bold', good ? 'text-emerald-300' : 'text-amber-300')}>
        {value}
        {unit !== 'K' && <span className="ml-0.5 text-[9px] font-normal text-slate-500">{unit}</span>}
      </div>
    </div>
  );
}

function PassRow({ label, nmcOk, lfpOk }: { label: string; nmcOk: boolean; lfpOk: boolean }) {
  const badge = (ok: boolean) => (
    <span
      className={cn(
        'rounded px-1.5 py-0.5 text-[9px] font-bold',
        ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300',
      )}
    >
      {ok ? 'PASS' : 'FAIL'}
    </span>
  );
  return (
    <div className="flex items-center justify-between rounded border border-slate-800 bg-slate-950/40 px-2 py-1">
      <span className="text-[10px] text-slate-500">{label}</span>
      <span className="flex items-center gap-1.5">
        <span className="text-[9px] text-slate-600">NMC</span>
        {badge(nmcOk)}
        <span className="text-[9px] text-slate-600">LFP</span>
        {badge(lfpOk)}
      </span>
    </div>
  );
}