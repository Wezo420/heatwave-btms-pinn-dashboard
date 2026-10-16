'use client';

/**
 * `CellTemperatureChart` -- per-cell temperature distribution.
 *
 * Shows all 12 cells at the current instant as a horizontal bar chart plus a
 * stacked "lateral spread" trace. Two reasons for this view existing alongside
 * the line chart:
 *
 * 1. **Lateral spread is a fault signature.** In a healthy pack, cells within
 *    1-2 K of each other. A growing spread indicates a blocked or fouled
 *    channel, which raises local ``R`` and eventually a local hot spot -- the
 *    dominant real-world BTMS failure mode, and invisible on a max-only trace.
 *
 * 2. **It is the chemistry-dependent comparison surface.** Switching NMC <-> LFP
 *    visibly shifts every bar, because the PINN's FiLM conditioning re-evaluates
 *    the field. That is the "one model, two chemistries" claim made concrete.
 */

import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, Cell as CellFill, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useStore, selectChemSpec } from '@/store/useStore';
import { cn, fmt, thermalColor } from '@/lib/utils';

interface Props {
  className?: string;
  height?: number;
}

export default function CellTemperatureChart({ className, height = 220 }: Props) {
  const latest = useStore((s) => s.latest);
  const spec = useStore(selectChemSpec);

  const data = useMemo(() => {
    if (!latest) return [];
    return latest.cell_temps_c.map((t, i) => ({
      cell: `C${String(i + 1).padStart(2, '0')}`,
      index: i + 1,
      temp: t,
      dev: t - latest.mean_cell_temp_c,
    }));
  }, [latest]);

  const domain = useMemo<[number, number]>(() => {
    if (!latest) return [20, 60];
    const temps = latest.cell_temps_c;
    const lo = Math.floor((Math.min(...temps) - 2) / 5) * 5;
    const hi = Math.ceil((Math.max(...temps) + 2) / 5) * 5;
    return [Math.max(lo, 0), Math.max(hi, lo + 10)];
  }, [latest]);

  if (data.length === 0) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 text-sm text-slate-500',
          className,
        )}
        style={{ height }}
      >
        Waiting for per-cell data...
      </div>
    );
  }

  const spread = latest ? latest.max_cell_temp_c - Math.min(...latest.cell_temps_c) : 0;
  const spreadTone = spread > 4 ? 'text-rose-300' : spread > 2 ? 'text-amber-300' : 'text-emerald-300';

  return (
    <div className={cn('rounded-xl border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Per-cell temperature</h3>
        <div className="text-right">
          <div className={cn('font-mono text-sm font-bold', spreadTone)}>{fmt(spread, 2)} K</div>
          <div className="text-[10px] text-slate-500">lateral spread</div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -18 }} barCategoryGap="18%">
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="cell"
            tick={{ fontSize: 9, fill: '#64748b' }}
            axisLine={{ stroke: '#334155' }}
            tickLine={false}
            interval={0}
          />
          <YAxis
            domain={domain}
            tick={{ fontSize: 9, fill: '#94a3b8' }}
            axisLine={{ stroke: '#334155' }}
            tickLine={false}
            width={44}
          />
          <ReferenceLine
            y={spec.recommended_max_cell_temp_c}
            stroke="#f97316"
            strokeDasharray="5 3"
            label={{
              value: 'ceiling',
              position: 'right',
              fill: '#f97316',
              fontSize: 9,
            }}
          />
          <ReferenceLine
            y={latest?.mean_cell_temp_c ?? 0}
            stroke="#64748b"
            strokeWidth={1}
            strokeDasharray="2 2"
          />
          <Tooltip
            cursor={{ fill: 'rgba(148,163,184,0.06)' }}
            isAnimationActive={false}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as { cell: string; temp: number; dev: number };
              return (
                <div className="rounded border border-slate-700 bg-slate-950/95 px-2 py-1 text-[11px] shadow-lg">
                  <div className="font-semibold text-slate-200">{row.cell}</div>
                  <div className="font-mono text-coolant">{fmt(row.temp, 3)} degC</div>
                  <div
                    className={cn(
                      'font-mono',
                      row.dev > 0 ? 'text-rose-300' : 'text-sky-300',
                    )}
                  >
                    {row.dev >= 0 ? '+' : ''}
                    {fmt(row.dev, 3)} K vs mean
                  </div>
                </div>
              );
            }}
          />
          <Bar dataKey="temp" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <CellFill key={d.cell} fill={thermalColor(d.temp, domain)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="mt-1 text-[10px] leading-snug text-slate-600">
        Lateral spread above 4 K is treated as a fouled-channel fault signature by the
        fail-safe supervisor: a partially blocked channel raises local resistance without
        changing the pack average, so it is only detectable from this view.
      </p>
    </div>
  );
}