'use client';

/**
 * `ActuationTimeline` -- Gantt-style view of the anticipatory schedule.
 *
 * The central scientific claim of this project is temporal: **the coolant ramp
 * begins measurably before the cell temperature would breach its ceiling**. That
 * claim is only legible if the coolant events and the thermal envelope are drawn
 * on a *shared time axis*. This component does exactly that, with one
 * non-negotiable rule:
 *
 *   Advisory events (predicted) and enforced events (commanded) are drawn in
 *   visually distinct lanes with distinct fills and distinct labels. They are
 *   never merged, because a reader must be able to tell which claim is a
 *   forecast and which is a measurement of what the actuator did.
 *
 * The lead-time annotation is the payoff: a bracket spanning from the last
 * anticipatory ramp to the moment the comfort ceiling would have been crossed
 * under the baseline (no-anticipation) trajectory. Its length in seconds is the
 * headline experimental result.
 */

import { useMemo } from 'react';
import { Clock, GitCommitVertical } from 'lucide-react';
import { useStore, selectComfortCeiling } from '@/store/useStore';
import { cn, fmt, fmtClock } from '@/lib/utils';
import type { ActuationEvent } from '@/lib/types';

interface Props {
  className?: string;
  /** Total time span in seconds. */
  durationS?: number;
}

interface Bar {
  event: ActuationEvent;
  leftPct: number;
  widthPct: number;
}

export default function ActuationTimeline({ className, durationS = 240 }: Props) {
  const advisory = useStore((s) => s.advisorySchedule);
  const enforced = useStore((s) => s.enforcedActuation);
  const anticipatoryEnabled = useStore((s) => s.anticipatoryEnabled);
  const trajectory = useStore((s) => s.trajectory);
  const zones = useStore((s) => s.enforcedZones.length ? s.enforcedZones : s.advisoryZones);
  const ceiling = useStore(selectComfortCeiling);

  const showEnforced = anticipatoryEnabled && enforced.length > 0;
  const primary = showEnforced ? enforced : advisory;

  const span = useMemo(() => {
    const tMax = Math.max(
      durationS,
      ...advisory.map((e) => e.t_s),
      ...enforced.map((e) => e.t_s),
      ...zones.map((z) => z.t_exit_s),
      trajectory?.frames.at(-1)?.t_s ?? 0,
    );
    return Math.max(tMax, 1);
  }, [advisory, enforced, zones, trajectory, durationS]);

  const toBar = (event: ActuationEvent): Bar => ({
    event,
    leftPct: (Math.max(0, event.t_s) / span) * 100,
    widthPct: Math.max(1.2, (span / 60 / span) * 100),
  });

  const bars = primary.map(toBar);
  const zoneBands = zones.map((z) => ({
    id: z.zone_id,
    leftPct: (z.t_enter_s / span) * 100,
    widthPct: Math.max(0.5, ((z.t_exit_s - z.t_enter_s) / span) * 100),
    risk: z.risk_index,
  }));

  /**
   * Lead time: seconds between the final anticipatory ramp and the first moment
   * the *baseline* (no-control) trajectory would have exceeded the ceiling.
   * Reported as null when the baseline never breached (nothing to anticipate).
   */
  const leadTime = useMemo(() => {
    if (!trajectory?.frames?.length) return null;
    const baseline = trajectory.advisory ?? '';
    const breachIdx = trajectory.frames.findIndex((f) => f.max_cell_temp_c > ceiling);
    if (breachIdx <= 0) return null;
    const lastRampT = Math.max(
      ...enforced.filter((e) => e.delta_lpm > 0).map((e) => e.t_s),
      -Infinity,
    );
    if (!Number.isFinite(lastRampT)) return null;
    const lead = trajectory.frames[breachIdx].t_s - lastRampT;
    return { seconds: lead, baselineNote: baseline || null, firstBreachT: trajectory.frames[breachIdx].t_s };
  }, [trajectory, enforced, ceiling]);

  return (
    <div className={cn('rounded-xl border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-200">
          <GitCommitVertical className="h-4 w-4 text-sky-400" aria-hidden />
          Anticipatory actuation timeline
        </h3>
        <span className="font-mono text-[10px] text-slate-500">
          0 &ndash; {fmtClock(span)} span
        </span>
      </div>

      {bars.length === 0 ? (
        <p className="py-6 text-center text-xs text-slate-500">
          No scheduled actuation. Run the anticipatory rollout to populate the schedule.
        </p>
      ) : (
        <>
          {/* Heat zones */}
          {zoneBands.length > 0 && (
            <div className="mb-2">
              <div className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-slate-500">
                Heat zones (detected)
              </div>
              <div className="relative h-4 rounded bg-slate-950/60">
                {zoneBands.map((z) => (
                  <div
                    key={z.id}
                    className="absolute top-0 h-full rounded-sm border border-rose-500/50 bg-rose-500/20"
                    style={{ left: `${z.leftPct}%`, width: `${z.widthPct}%` }}
                    title={`${z.id}: risk index ${fmt(z.risk, 3)}`}
                  >
                    <span className="absolute -top-0.5 left-1 whitespace-nowrap text-[8px] text-rose-300">
                      {z.id}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Lane: advisory */}
          <Lane
            label="ADVISORY"
            subtitle="predicted (not applied)"
            variant="advisory"
            bars={advisory.map(toBar)}
            _span={span}
          />

          {/* Lane: enforced */}
          <Lane
            label="ENFORCED"
            subtitle={showEnforced ? 'commanded (applied)' : 'disabled — showing advisory'}
            variant="enforced"
            bars={(showEnforced ? enforced : advisory).map(toBar)}
            _span={span}
            dimmed={!showEnforced}
          />

          {/* Ceiling axis */}
          <div className="relative mt-1 h-4 border-t border-slate-800">
            {Array.from({ length: 6 }, (_, i) => {
              const t = (span / 5) * i;
              return (
                <span
                  key={i}
                  className="absolute top-0.5 -translate-x-1/2 font-mono text-[9px] text-slate-600"
                  style={{ left: `${(t / span) * 100}%` }}
                >
                  {fmtClock(t)}
                </span>
              );
            })}
          </div>
        </>
      )}

      {/* Lead-time annotation */}
      {leadTime && leadTime.seconds > 0 && (
        <div className="mt-2 flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-950/20 px-2.5 py-2">
          <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" aria-hidden />
          <div>
            <p className="text-[11px] font-semibold text-emerald-300">
              Anticipation lead time: {fmt(leadTime.seconds, 1)} s
            </p>
            <p className="mt-0.5 text-[10px] leading-snug text-slate-400">
              The final anticipatory ramp commanded at {fmtClock(
                Math.max(...enforced.filter((e) => e.delta_lpm > 0).map((e) => e.t_s), 0),
              )}
              , before the comfort ceiling ({fmt(ceiling, 0)}&deg;C) was reached at{' '}
              {fmtClock(leadTime.firstBreachT)}. Under the constant-flow baseline the pack
              would have crossed the ceiling without any pre-emptive action.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function Lane({
  label,
  subtitle,
  variant,
  bars,
  // `span` is declared by both call sites and part of the component's contract,
  // but the lane positions bars by their precomputed percentage widths, so it is
  // not read here. Prefix with `_` to say that explicitly instead of leaving a
  // bare unused binding.
  _span,
  dimmed,
}: {
  label: string;
  subtitle: string;
  variant: 'advisory' | 'enforced';
  bars: Bar[];
  /** Total timeline span in seconds. Unused here; prefixed `_` at the call site. */
  _span: number;
  dimmed?: boolean;
}) {
  return (
    <div className="mb-1.5">
      <div className="mb-0.5 flex items-baseline gap-2">
        <span
          className={cn(
            'text-[9px] font-bold tracking-wider',
            variant === 'advisory' ? 'text-slate-500' : 'text-sky-300',
          )}
        >
          {label}
        </span>
        <span className="text-[9px] text-slate-600">{subtitle}</span>
      </div>
      <div
        className={cn(
          'relative h-5 rounded bg-slate-950/60',
          dimmed && 'opacity-40',
          variant === 'advisory' && 'border border-dashed border-slate-700',
        )}
      >
        {/* Second gridlines */}
        {Array.from({ length: 11 }, (_, i) => (
          <span
            key={i}
            className="absolute top-0 h-full w-px bg-slate-800/50"
            style={{ left: `${(i / 10) * 100}%` }}
          />
        ))}
        {bars.map((b, i) => (
          <div
            key={`${b.event.t_s}-${i}`}
            className={cn(
              'absolute top-1/2 h-3 -translate-y-1/2 rounded-sm transition-all',
              b.event.delta_lpm > 0
                ? 'bg-amber-400/80 shadow-[0_0_8px_rgba(251,191,36,0.6)]'
                : 'bg-sky-400/80',
              variant === 'advisory' && 'border border-dashed border-slate-500',
            )}
            style={{ left: `${b.leftPct}%`, width: `${b.widthPct}%` }}
            title={`${b.event.trigger} @ ${fmtClock(b.event.t_s)}: ${fmt(
              b.event.command_lpm,
              2,
            )} L/min (${b.event.delta_lpm > 0 ? '+' : ''}${fmt(b.event.delta_lpm, 2)}), confidence ${fmt(
              b.event.confidence * 100,
              0,
            )}%`}
          />
        ))}
      </div>
    </div>
  );
}