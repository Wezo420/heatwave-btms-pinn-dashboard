'use client';

/**
 * `HeatwaveAlertOverlay` -- the predictive cooling actuation display.
 *
 * This component answers three questions simultaneously and must not blur them:
 *
 *   1. **Is the vehicle approaching a heat zone?**  -> zone countdown / banner
 *   2. **What is the controller doing about it, and how early?**  -> lead time
 *   3. **What did we *predict* vs what did we *do*?**  -> advisory vs enforced
 *
 * Questions 2 and 3 are rendered in visually distinct treatments on purpose.
 * The advisory schedule (a prediction) is drawn as a translucent dashed band;
 * the enforced actuation (an applied command) is drawn as a solid, higher-
 * contrast band. Collapsing them would imply a certainty the prediction does not
 * have -- a failure mode documented in the model card's safety section.
 *
 * The overlay is `position: fixed` and non-blocking (`pointer-events-none` with
 * interactive children) so it never prevents the operator from using the rest
 * of the dashboard during an alert.
 */

import { useMemo } from 'react';
import { AlertTriangle, Brain, Route, Shield, Thermometer, Zap } from 'lucide-react';
import { useStore, selectChemSpec } from '@/store/useStore';
import { cn, fmt, fmtClock } from '@/lib/utils';
import type { ActuationEvent, HeatwaveZone } from '@/lib/types';

type Severity = 'nominal' | 'watch' | 'advisory' | 'critical';

const SEVERITY_STYLE: Record<Severity, { ring: string; bg: string; text: string; label: string }> = {
  nominal: {
    ring: 'border-emerald-500/40',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-300',
    label: 'NOMINAL',
  },
  watch: {
    ring: 'border-sky-500/40',
    bg: 'bg-sky-500/10',
    text: 'text-sky-300',
    label: 'WATCH',
  },
  advisory: {
    ring: 'border-amber-400/50',
    bg: 'bg-amber-400/10',
    text: 'text-amber-300',
    label: 'ANTICIPATORY ACTUATION',
  },
  critical: {
    ring: 'border-rose-500/60',
    bg: 'bg-rose-500/15',
    text: 'text-rose-300',
    label: 'THERMAL FAIL-SAFE',
  },
};

interface Props {
  /** Compact single-line variant for the header strip. */
  compact?: boolean;
  className?: string;
}

export default function HeatwaveAlertOverlay({ compact = false, className }: Props) {
  const chemistry = useStore((s) => s.chemistry);
  const latest = useStore((s) => s.latest);
  const tripState = useStore((s) => s.tripState);
  const anticipatoryEnabled = useStore((s) => s.anticipatoryEnabled);
  const advisoryZones = useStore((s) => s.advisoryZones);
  const advisorySchedule = useStore((s) => s.advisorySchedule);
  const enforcedZones = useStore((s) => s.enforcedZones);
  const enforcedActuation = useStore((s) => s.enforcedActuation);
  const spec = useStore(selectChemSpec);

  // No local 1 Hz clock. An earlier version kept `now` via setInterval purely to
// force a re-render, but nothing read its value -- the overlay's visible state
// all comes from the store, which the telemetry socket already updates at 2 Hz.
// The interval was therefore a second timer running for no reason. If a
// wall-clock "time since breach" readout is ever added, derive it from the
// newest frame's `t_s` rather than reintroducing an unsynchronised clock.

  /* -- severity ----------------------------------------------------- */
  const severity: Severity = useMemo(() => {
    if (tripState === 'trip') return 'critical';
    if (tripState === 'derate') return 'advisory';
    if (latest?.anticipatory_active) return 'advisory';
    if (latest && latest.risk_index > 0.55) return 'watch';
    return 'nominal';
  }, [tripState, latest]);

  /* -- active / upcoming zone --------------------------------------- */
  const zoneStatus = useMemo(() => {
    const t = latest?.t_s ?? 0;
    const allZones: HeatwaveZone[] = [...enforcedZones, ...advisoryZones];
    if (allZones.length === 0) return { active: null, upcoming: null, countdownS: null };

    const active = allZones.find((z) => t >= z.t_enter_s && t <= z.t_exit_s) ?? null;
    const upcoming = allZones.find((z) => z.t_enter_s > t) ?? null;
    const countdownS = upcoming ? Math.max(0, upcoming.t_enter_s - t) : null;
    return { active, upcoming, countdownS };
  }, [latest, enforcedZones, advisoryZones]);

  /* -- lead-time metric --------------------------------------------- */
  const leadTime = useMemo(() => {
    const events = anticipatoryEnabled ? enforcedActuation : advisorySchedule;
    if (events.length === 0) return null;
    const rising = events.filter((e) => e.delta_lpm > 0);
    if (rising.length === 0) return null;
    const latestRise = rising[rising.length - 1];
    const zone = zoneStatus.active ?? zoneStatus.upcoming;
    if (!zone) return { seconds: latestRise.lead_time_s, zone: null };
    const lead = zone.t_enter_s - latestRise.t_s;
    return { seconds: Math.max(0, lead), zone: zone.zone_id };
  }, [enforcedActuation, advisorySchedule, anticipatoryEnabled, zoneStatus]);

  const style = SEVERITY_STYLE[severity];
  const latestRise = useMemo(() => {
    const events = anticipatoryEnabled ? enforcedActuation : advisorySchedule;
    return events.filter((e) => e.delta_lpm > 0).slice(-1)[0] ?? null;
  }, [enforcedActuation, advisorySchedule, anticipatoryEnabled]);

  if (compact) {
    return (
      <div
        className={cn(
          'pointer-events-none fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold backdrop-blur-md transition-colors',
          style.ring,
          style.bg,
          style.text,
          className,
        )}
        role="status"
        aria-live="polite"
      >
        <span
          className={cn(
            'h-2 w-2 rounded-full',
            severity === 'critical' ? 'animate-pulse bg-rose-400' : 'bg-current',
          )}
        />
        {style.label}
        {latest && (
          <span className="font-mono font-normal text-slate-300">
            {fmt(latest.max_cell_temp_c, 1)}&deg;C &middot; {fmt(latest.coolant_flow_lpm, 1)} L/min
          </span>
        )}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center p-4',
        className,
      )}
    >
      <div
        className={cn(
          'pointer-events-auto w-full max-w-4xl animate-slide-up rounded-2xl border backdrop-blur-xl',
          'bg-slate-950/85 shadow-2xl',
          style.ring,
          severity === 'critical' && 'animate-pulse-ring',
        )}
        role="alert"
        aria-live="assertive"
      >
        {/* Header strip */}
        <div className={cn('flex items-center justify-between gap-3 rounded-t-2xl px-4 py-2.5', style.bg)}>
          <div className="flex items-center gap-2.5">
            {severity === 'critical' ? (
              <Shield className="h-5 w-5 text-rose-400" aria-hidden />
            ) : severity === 'advisory' ? (
              <Zap className="h-5 w-5 text-amber-400" aria-hidden />
            ) : (
              <Thermometer className="h-5 w-5 text-emerald-400" aria-hidden />
            )}
            <span className={cn('text-sm font-bold tracking-wide', style.text)}>
              {style.label}
            </span>
            <span className="text-[11px] text-slate-400">
              {chemistry.toUpperCase()} &middot; onset {fmt(spec.thermal_runaway_onset_c, 0)}&deg;C
            </span>
          </div>
          <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
            <span className={cn('h-1.5 w-1.5 rounded-full', latest?.anticipatory_active ? 'animate-pulse bg-amber-400' : 'bg-slate-600')} />
            WS {latest ? 'streaming' : 'idle'}
          </div>
        </div>

        <div className="grid gap-4 p-4 md:grid-cols-4">
          {/* 1. Predicted zone */}
          <Metric
            icon={<Route className="h-3.5 w-3.5" />}
            label={zoneStatus.active ? 'IN ZONE' : zoneStatus.upcoming ? 'ZONE IN' : 'ZONE SCAN'}
            value={
              zoneStatus.active
                ? zoneStatus.active.zone_id
                : zoneStatus.countdownS !== null
                  ? fmtClock(zoneStatus.countdownS)
                  : 'CLEAR'
            }
            sub={
              zoneStatus.active
                ? `peak ambient ${fmt(zoneStatus.active.peak_ambient_c, 1)} degC`
                : zoneStatus.upcoming
                  ? `${zoneStatus.upcoming.zone_id} @ ${fmt(zoneStatus.upcoming.peak_ambient_c, 0)} degC`
                  : 'no heat zones on route'
            }
            accent={zoneStatus.active ? 'text-amber-300' : 'text-slate-200'}
          />

          {/* 2. Actuation */}
          <Metric
            icon={<Zap className="h-3.5 w-3.5" />}
            label="COOLANT COMMAND"
            value={`${fmt(latest?.coolant_flow_lpm ?? 0, 2)} L/min`}
            sub={
              latest && latest.commanded_flow_lpm > 6.05
                ? `ramp +${fmt(latest.commanded_flow_lpm - 6, 2)} L/min over set-point`
                : 'holding nominal 6.00 L/min'
            }
            accent="text-coolant"
          />

          {/* 3. Lead time */}
          <Metric
            icon={<Brain className="h-3.5 w-3.5" />}
            label="ANTICIPATION LEAD"
            value={leadTime ? `${fmt(leadTime.seconds, 0)} s` : '--'}
            sub={
              latestRise
                ? `trigger: ${latestRise.trigger} (conf ${fmt(latestRise.confidence * 100, 0)}%)`
                : 'no anticipatory event yet'
            }
            accent="text-violet-300"
          />

          {/* 4. Margin */}
          <Metric
            icon={<Thermometer className="h-3.5 w-3.5" />}
            label="EXOTHERM MARGIN"
            value={`${fmt(latest?.thermal_runaway_margin_c ?? 0, 1)} K`}
            sub={`max cell ${fmt(latest?.max_cell_temp_c ?? 0, 2)} degC`}
            accent={
              (latest?.thermal_runaway_margin_c ?? 999) < 45 ? 'text-rose-300' : 'text-emerald-300'
            }
          />
        </div>

        {/* Advisory vs enforced bands */}
        <div className="grid gap-3 border-t border-slate-800/80 px-4 py-3 md:grid-cols-2">
          <Band
            variant="advisory"
            title="ADVISORY (predicted)"
            events={advisorySchedule}
            active={latest?.anticipatory_active ?? false}
            caption="Receding-horizon forecast. Not applied to the actuator."
          />
          <Band
            variant="enforced"
            title="ENFORCED (commanded)"
            events={anticipatoryEnabled ? enforcedActuation : advisorySchedule}
            active={anticipatoryEnabled && (latest?.anticipatory_active ?? false)}
            caption={
              anticipatoryEnabled
                ? 'Applied to the coolant pump through the rate limiter.'
                : 'Anticipation disabled for the A/B study -- advisory shown instead.'
            }
          />
        </div>

        {severity !== 'nominal' && zoneStatus.active && (
          <div className="flex items-start gap-2 border-t border-slate-800/80 px-4 py-2.5 text-[11px] leading-snug text-amber-200/90">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            <p>
              <strong>{zoneStatus.active.zone_id}</strong> dominant load:{' '}
              <em>{zoneStatus.active.dominant_mode}</em>. Required flow{' '}
              {fmt(zoneStatus.active.required_flow_lpm, 1)} L/min; projected peak{' '}
              {fmt(zoneStatus.active.predicted_peak_cell_c, 1)}&deg;C leaves{' '}
              {fmt(zoneStatus.active.predicted_margin_c, 1)} K of margin below the{' '}
              {fmt(spec.thermal_runaway_onset_c, 0)}&deg;C exotherm onset.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Metric({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  accent: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
        {icon}
        {label}
      </div>
      <div className={cn('mt-1 font-mono text-lg font-bold leading-none', accent)}>{value}</div>
      <div className="mt-1 truncate text-[10px] text-slate-500" title={sub}>
        {sub}
      </div>
    </div>
  );
}

/**
 * `Band` -- advisory vs enforced schedule visualisation.
 *
 * The visual distinction is load-bearing: advisory uses a 1 px dashed border at
 * 60% opacity; enforced uses a 2 px solid border at full opacity with a solid
 * fill. A reader must be able to tell which one they are looking at without
 * reading the label.
 */
function Band({
  variant,
  title,
  events,
  active,
  caption,
}: {
  variant: 'advisory' | 'enforced';
  title: string;
  events: ActuationEvent[];
  active: boolean;
  caption: string;
}) {
  const advisory = variant === 'advisory';
  const shown = events.slice(-6);

  return (
    <div
      className={cn(
        'rounded-lg border px-3 py-2',
        advisory
          ? 'border-dashed border-slate-600/60 bg-slate-900/40 opacity-90'
          : 'border-solid border-amber-500/40 bg-amber-500/5',
        active && !advisory && 'shadow-glow',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span
          className={cn(
            'text-[10px] font-bold tracking-wider',
            advisory ? 'text-slate-400' : 'text-amber-300',
          )}
        >
          {title}
        </span>
        <span className="text-[10px] font-mono text-slate-500">
          {events.length} event{events.length === 1 ? '' : 's'}
        </span>
      </div>

      {shown.length === 0 ? (
        <p className="mt-1.5 text-[11px] text-slate-600">no scheduled events</p>
      ) : (
        <ul className="mt-1.5 space-y-0.5">
          {shown.map((e, i) => (
            <li key={`${e.t_s}-${i}`} className="flex items-baseline justify-between gap-2 text-[11px]">
              <span className="font-mono text-slate-400">{fmtClock(e.t_s)}</span>
              <span
                className={cn(
                  'font-mono font-semibold',
                  e.delta_lpm > 0 ? 'text-amber-300' : 'text-sky-300',
                )}
              >
                {e.delta_lpm > 0 ? '+' : ''}
                {fmt(e.delta_lpm, 2)} &rarr; {fmt(e.command_lpm, 1)}
              </span>
              <span className="truncate text-[10px] text-slate-600" title={e.trigger}>
                {e.trigger.replace('anticipatory:', '')}
              </span>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-1.5 text-[10px] leading-snug text-slate-600">{caption}</p>
    </div>
  );
}