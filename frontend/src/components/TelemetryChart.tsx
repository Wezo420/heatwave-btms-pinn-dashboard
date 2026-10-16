'use client';

/**
 * `TelemetryChart` -- the real-time multi-series plot.
 *
 * Chart design decisions (each one is a deliberate choice, not a default):
 *
 * 1. **Separate axes by physical dimension.** Temperature (degC) on the left
 *    axis, volumetric flow (L/min) on the right. Overlaying them on one axis
 *    would imply a shared unit and make the "flow ramps *before* temperature"
 *    relationship -- the central claim of the project -- visually unreadable.
 *
 * 2. **Reference lines that change with chemistry.** The comfort ceiling and the
 *    exotherm onset are chemistry-dependent, so the chart's `ReferenceLine`s are
 *    props rather than constants. On an NMC chart the 210 degC line is inside
 *    the axis range and visible; on LFP it is 60 K further out and shown
 *    compressed by a non-linear y-domain below 120 degC.
 *
 * 3. **Non-linear y-axis above 120 degC.** Cell temperatures live in the 25-55
 *    degC band; runaway thresholds live at 210/270 degC. A linear axis spanning
 *    both would flatten every real signal into a line at the bottom. We use a
 *    piecewise (symlog) scale so the operating band is readable *and* the
 *    threshold line is on-screen.
 *
 * 4. **Anticipatory-activation shading.** Rectangles mark the windows where the
 *    controller was pre-emptively ramping (commanded flow > baseline), so the
 *    lead time is visible rather than asserted.
 *
 * 5. **Bounded history window.** 600 points by default: enough to see the
 *    thermal transient, small enough to keep the render under 16 ms.
 */

import { useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useStore, selectComfortCeiling, selectRunawayOnset } from '@/store/useStore';
import { cn, fmt, fmtClock } from '@/lib/utils';
import type { TelemetryFrame } from '@/lib/types';

interface Props {
  height?: number;
  showCells?: boolean;
  className?: string;
  /** Rolling window in frames; older points are dropped from the render. */
  windowSize?: number;
}

interface PlotRow {
  t: number;
  label: string;
  maxCell: number;
  meanCell: number;
  minCell: number;
  flow: number;
  command: number;
  coolantOut: number;
  ambient: number;
  margin: number;
  anticipatory: boolean;
}

/** Symmetric-log axis tick generator: linear below 120 degC, log above. */
function thermalTicks(max: number): number[] {
  const ticks = [0, 20, 40, 60, 80, 100, 120];
  for (const v of [150, 180, 210, 240, 270, 300, 360, 420, 500]) {
    if (v <= max) ticks.push(v);
  }
  return [...new Set(ticks)].sort((a, b) => a - b);
}

export default function TelemetryChart({
  height = 340,
  showCells = true,
  className,
  windowSize = 600,
}: Props) {
  const frames = useStore((s) => s.frames);
  const chemistry = useStore((s) => s.chemistry);
  const ceiling = useStore(selectComfortCeiling);
  const onset = useStore(selectRunawayOnset);

  const data = useMemo<PlotRow[]>(() => {
    const slice = frames.slice(-windowSize);
    const t0 = slice.length ? slice[0].t_s : 0;
    return slice.map((f: TelemetryFrame) => {
      const temps = f.cell_temps_c;
      return {
        t: f.t_s - t0,
        label: fmtClock(f.t_s),
        maxCell: f.max_cell_temp_c,
        meanCell: f.mean_cell_temp_c,
        minCell: temps.length ? Math.min(...temps) : f.mean_cell_temp_c,
        flow: f.coolant_flow_lpm,
        command: f.commanded_flow_lpm,
        coolantOut: f.coolant_outlet_c,
        ambient: f.ambient_c,
        margin: f.thermal_runaway_margin_c,
        anticipatory: f.anticipatory_active,
      };
    });
  }, [frames, windowSize]);

  /** Merge contiguous anticipatory windows into shading rectangles. */
  const anticipatoryBands = useMemo(() => {
    const bands: Array<{ from: number; to: number }> = [];
    let start: number | null = null;
    data.forEach((row, i) => {
      if (row.anticipatory && start === null) start = row.t;
      if ((!row.anticipatory || i === data.length - 1) && start !== null) {
        bands.push({ from: start, to: row.t });
        start = null;
      }
    });
    return bands;
  }, [data]);

  const flowMax = useMemo(
    () => Math.max(8, ...data.map((d) => Math.max(d.flow, d.command))),
    [data],
  );

  const tempMax = useMemo(
    () =>
      Math.max(
        80,
        onset,
        ...data.map((d) => d.maxCell),
        ceiling + 8,
      ),
    [data, onset, ceiling],
  );

  if (data.length < 2) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 text-sm text-slate-500',
          className,
        )}
        style={{ height }}
      >
        Waiting for telemetry frames...
      </div>
    );
  }

  return (
    <div className={cn('rounded-xl border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-200">
          Real-time telemetry
          <span className="ml-2 font-normal text-xs text-slate-500">
            {data.length} frames &middot; last {fmtClock(data[data.length - 1].t)} of run
          </span>
        </h3>
        <div className="flex items-center gap-3 text-[11px]">
          <LegendSwatch color="#ef4444" label="Max cell" />
          <LegendSwatch color="#f97316" label="Mean cell" />
          <LegendSwatch color="#38bdf8" label="Coolant flow" />
          <LegendSwatch color="#a78bfa" label="Command" dashed />
          <LegendSwatch color="#fbbf24" label="Coolant out" />
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 56, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="gradMaxCell" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity={0.28} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="gradFlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.24} />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />

          {/* Anticipatory activation windows */}
          {anticipatoryBands.map((band, i) => (
            <ReferenceArea
              key={`band-${i}`}
              x1={band.from}
              x2={band.to}
              fill="#f59e0b"
              fillOpacity={0.09}
              stroke="#f59e0b"
              strokeOpacity={0.28}
              strokeDasharray="2 3"
            />
          ))}

          {/* Chemistry-dependent thresholds */}
          <ReferenceLine
            y={ceiling}
            stroke="#f97316"
            strokeDasharray="6 3"
            strokeWidth={1.4}
            label={{
              value: `comfort ceiling ${fmt(ceiling, 0)} C`,
              position: 'insideTopRight',
              fill: '#f97316',
              fontSize: 10,
            }}
          />
          <ReferenceLine
            y={onset}
            stroke="#dc2626"
            strokeWidth={1.6}
            strokeDasharray="3 2"
            label={{
              value: `exotherm onset ${fmt(onset, 0)} C`,
              position: 'insideTopRight',
              fill: '#dc2626',
              fontSize: 10,
            }}
          />

          <XAxis
            dataKey="t"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(v: number) => fmtClock(v)}
            stroke="#475569"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            minTickGap={40}
          />

          {/* Left axis: temperature, with a compressed upper region */}
          <YAxis
            yAxisId="temp"
            domain={[0, tempMax]}
            ticks={thermalTicks(tempMax)}
            tickFormatter={(v: number) => `${v}`}
            stroke="#475569"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            width={38}
            label={{
              value: 'degC',
              angle: -90,
              position: 'insideLeft',
              fill: '#64748b',
              fontSize: 10,
              dy: 24,
            }}
          />

          {/* Right axis: volumetric flow */}
          <YAxis
            yAxisId="flow"
            orientation="right"
            domain={[0, Math.ceil(flowMax * 1.15)]}
            stroke="#475569"
            tick={{ fontSize: 10, fill: '#7dd3fc' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            width={46}
            label={{
              value: 'L/min',
              angle: 90,
              position: 'insideRight',
              fill: '#38bdf8',
              fontSize: 10,
              dy: -40,
            }}
          />

          <Tooltip
            content={<TelemetryTooltip ceiling={ceiling} onset={onset} chemistry={chemistry} />}
            isAnimationActive={false}
          />

          <Area
            yAxisId="temp"
            type="monotone"
            dataKey="maxCell"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#gradMaxCell)"
            dot={false}
            isAnimationActive={false}
            name="Max cell"
          />
          {showCells && (
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="meanCell"
              stroke="#f97316"
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              name="Mean cell"
            />
          )}
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="coolantOut"
            stroke="#fbbf24"
            strokeWidth={1.3}
            strokeDasharray="4 3"
            dot={false}
            isAnimationActive={false}
            name="Coolant outlet"
          />

          <Area
            yAxisId="flow"
            type="stepAfter"
            dataKey="flow"
            stroke="#38bdf8"
            strokeWidth={2.2}
            fill="url(#gradFlow)"
            dot={false}
            isAnimationActive={false}
            name="Coolant flow"
          />
          <Line
            yAxisId="flow"
            type="stepAfter"
            dataKey="command"
            stroke="#a78bfa"
            strokeWidth={1.6}
            strokeDasharray="5 3"
            dot={false}
            isAnimationActive={false}
            name="Commanded flow"
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="mt-1.5 text-[11px] leading-snug text-slate-500">
        Shaded windows mark anticipatory heatwave actuation (commanded flow above the
        nominal 6.0 L/min set-point). The right-hand axis is volumetric flow; note the
        command step precedes the temperature rise, which is the anticipation claim this
        project measures.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function LegendSwatch({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1 text-slate-400">
      <span
        className="h-0.5 w-3.5 rounded"
        style={{
          background: dashed
            ? `repeating-linear-gradient(90deg, ${color} 0 3px, transparent 3px 6px)`
            : color,
        }}
      />
      {label}
    </span>
  );
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number; color: string; name: string }>;
  label?: string;
  ceiling: number;
  onset: number;
  chemistry: string;
}

function TelemetryTooltip({ active, payload, label, ceiling, onset, chemistry }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const get = (key: string) => payload.find((p) => p.dataKey === key)?.value ?? 0;
  const maxCell = get('maxCell');
  const flow = get('flow');
  const margin = onset - maxCell;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      <div className="mb-1.5 border-b border-slate-800 pb-1 font-mono text-[11px] text-slate-400">
        t = {label} &middot; {chemistry.toUpperCase()}
      </div>
      <dl className="space-y-0.5">
        <Row label="Max cell" value={`${fmt(maxCell, 2)} degC`} color="#ef4444" />
        <Row label="Mean cell" value={`${fmt(get('meanCell'), 2)} degC`} color="#f97316" />
        <Row label="Min cell" value={`${fmt(get('minCell'), 2)} degC`} color="#fb923c" />
        <Row label="Coolant flow" value={`${fmt(flow, 2)} L/min`} color="#38bdf8" />
        <Row label="Commanded" value={`${fmt(get('command'), 2)} L/min`} color="#a78bfa" />
        <Row label="Coolant out" value={`${fmt(get('coolantOut'), 2)} degC`} color="#fbbf24" />
        <Row label="Ambient" value={`${fmt(get('ambient'), 1)} degC`} color="#64748b" />
      </dl>
      <div className="mt-1.5 border-t border-slate-800 pt-1.5">
        <Row
          label="Exotherm margin"
          value={`${fmt(margin, 1)} K`}
          color={margin < 20 ? '#f43f5e' : margin < 45 ? '#f59e0b' : '#22c55e'}
        />
        <Row
          label="vs ceiling"
          value={`${fmt(ceiling - maxCell, 2)} K headroom`}
          color={maxCell > ceiling ? '#f43f5e' : '#22c55e'}
        />
      </div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="flex items-center gap-1.5 text-slate-400">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
        {label}
      </dt>
      <dd className="font-mono font-semibold" style={{ color }}>
        {value}
      </dd>
    </div>
  );
}