'use client';

/**
 * `StatTiles` -- the KPI row.
 *
 * Each tile is a "claim + evidence" pair: the big number is the claim, and the
 * sub-line names the measured quantity it came from plus the threshold it is
 * being judged against. A dashboard that shows only the number invites the
 * reader to supply their own notion of what "good" means; naming the threshold
 * removes that ambiguity, which matters here because the thresholds are
 * chemistry-dependent and change with the toggle.
 */

import { Activity, Gauge, Thermometer, Waves, Zap } from 'lucide-react';
import { useStore, selectChemSpec, selectComfortCeiling } from '@/store/useStore';
import { cn, fmt, fmtMs } from '@/lib/utils';

interface Props {
  className?: string;
}

export default function StatTiles({ className }: Props) {
  const latest = useStore((s) => s.latest);
  const field = useStore((s) => s.field);
  const modelInfo = useStore((s) => s.modelInfo);
  const latencyEma = useStore((s) => s.latencyEmaMs);
  const tripState = useStore((s) => s.tripState);
  const spec = useStore(selectChemSpec);
  const ceiling = useStore(selectComfortCeiling);

  const maxTemp = latest?.max_cell_temp_c ?? field?.max_temp_c ?? 0;
  const margin = spec.thermal_runaway_onset_c - maxTemp;
  const flow = latest?.coolant_flow_lpm ?? 0;
  const joule = latest?.joule_heat_w ?? 0;
  const rejected = latest?.heat_rejected_w ?? field?.heat_rejected_w ?? 0;
  const inferenceMs = field?.inference_latency_ms ?? latencyEma;
  const budget = modelInfo?.latency_budget_ms ?? 800;

  return (
    <div className={cn('grid grid-cols-2 gap-2.5 lg:grid-cols-5', className)}>
      <Tile
        icon={<Thermometer className="h-4 w-4" />}
        label="Max cell temp"
        value={`${fmt(maxTemp, 1)}`}
        unit="degC"
        sub={`ceiling ${fmt(ceiling, 0)} degC | onset ${fmt(spec.thermal_runaway_onset_c, 0)} degC`}
        tone={
          maxTemp > ceiling
            ? 'critical'
            : maxTemp > ceiling - 5
              ? 'warn'
              : 'ok'
        }
        badge={tripState !== 'ok' ? tripState.toUpperCase() : undefined}
      />

      <Tile
        icon={<Activity className="h-4 w-4" />}
        label="Exotherm margin"
        value={`${fmt(margin, 1)}`}
        unit="K"
        sub={`${spec.key.toUpperCase()} adiabatic onset ${fmt(spec.thermal_runaway_onset_c, 0)} degC`}
        tone={margin < 45 ? 'critical' : margin < 80 ? 'warn' : 'ok'}
      />

      <Tile
        icon={<Waves className="h-4 w-4" />}
        label="Coolant flow"
        value={`${fmt(flow, 2)}`}
        unit="L/min"
        sub={
          latest && latest.commanded_flow_lpm > 6.05
            ? `ramping +${fmt(latest.commanded_flow_lpm - 6, 2)} over set-point`
            : 'nominal set-point 6.00 L/min'
        }
        tone={latest?.anticipatory_active ? 'accent' : 'neutral'}
        badge={latest?.anticipatory_active ? 'ANTICIPATORY' : undefined}
      />

      <Tile
        icon={<Zap className="h-4 w-4" />}
        label="Joule heat"
        value={`${fmt(joule, 0)}`}
        unit="W"
        sub={`rejected ${fmt(rejected, 0)} W | ${fmt(spec.cp_j_kgk, 0)} J/kgK`}
        tone="neutral"
      />

      <Tile
        icon={<Gauge className="h-4 w-4" />}
        label="Inference latency"
        value={inferenceMs ? fmtMs(inferenceMs) : '--'}
        unit=""
        sub={`budget ${fmt(budget, 0)} ms | ${modelInfo?.device ?? 'cpu'}`}
        tone={inferenceMs <= budget ? 'ok' : 'critical'}
        badge={inferenceMs <= budget ? 'WITHIN BUDGET' : 'OVER BUDGET'}
      />
    </div>
  );
}

type Tone = 'ok' | 'warn' | 'critical' | 'accent' | 'neutral';

const TONE_STYLE: Record<Tone, { value: string; ring: string }> = {
  ok: { value: 'text-emerald-300', ring: 'border-emerald-500/25 hover:border-emerald-500/40' },
  warn: { value: 'text-amber-300', ring: 'border-amber-500/30 hover:border-amber-500/50' },
  critical: {
    value: 'text-rose-300',
    ring: 'border-rose-500/40 hover:border-rose-500/60 bg-rose-950/20',
  },
  accent: { value: 'text-amber-300', ring: 'border-amber-400/35 hover:border-amber-400/55' },
  neutral: { value: 'text-slate-100', ring: 'border-slate-700/60 hover:border-slate-600' },
};

function Tile({
  icon,
  label,
  value,
  unit,
  sub,
  tone,
  badge,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit: string;
  sub: string;
  tone: Tone;
  badge?: string;
}) {
  const style = TONE_STYLE[tone];
  return (
    <div
      className={cn(
        'relative rounded-xl border bg-slate-900/50 p-3 transition-colors',
        style.ring,
      )}
    >
      <div className="flex items-center justify-between gap-1">
        <span className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
          {icon}
          {label}
        </span>
        {badge && (
          <span
            className={cn(
              'rounded px-1 py-0.5 text-[8px] font-bold tracking-wider',
              tone === 'critical'
                ? 'bg-rose-500/25 text-rose-200'
                : tone === 'accent' || tone === 'warn'
                  ? 'bg-amber-500/25 text-amber-200'
                  : 'bg-emerald-500/20 text-emerald-200',
            )}
          >
            {badge}
          </span>
        )}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span className={cn('font-mono text-2xl font-bold leading-none', style.value)}>{value}</span>
        {unit && <span className="text-xs text-slate-500">{unit}</span>}
      </div>
      <p className="mt-1 truncate text-[10px] text-slate-500" title={sub}>
        {sub}
      </p>
    </div>
  );
}