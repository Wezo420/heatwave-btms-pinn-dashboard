'use client';

/**
 * `ControlPanel` -- operator inputs and the coolant actuator.
 *
 * The manual-flow control is intentionally *not* a plain slider that writes
 * straight to the actuator. Two properties make it safe to use in a demo of a
 * safety-adjacent system:
 *
 *  1. **Rate-limited locally before it is sent.** The slider shows the *requested*
 *     set-point; the value actually sent is the previous command advanced by
 *     `ramp_rate * dt`. This mirrors the backend's own rate limiter, so the UI
 *     cannot command a step the pump physically cannot make.
 *  2. **Fail-safe precedence is explicit.** If a trip is latched, the slider is
 *     disabled and a "reset required" affordance appears. There is no way to
 *     command a lower flow while a trip is active.
 *
 * The anticipation master switch is the A/B control for the project's central
 * claim: turning it off and re-running the same seeded rollout gives a
 * measured A/B difference rather than an assertion.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, RotateCcw, ShieldAlert, Sliders, Zap } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { clamp, cn, fmt } from '@/lib/utils';
import * as api from '@/lib/api';

interface Props {
  onRunRoute: () => void;
  running: boolean;
  className?: string;
}

export default function ControlPanel({ onRunRoute, running, className }: Props) {
  const chemistry = useStore((s) => s.chemistry);
  const inputs = useStore((s) => s.inputs);
  const setInput = useStore((s) => s.setInput);
  const catalogue = useStore((s) => s.catalogue);
  const tripState = useStore((s) => s.tripState);
  const anticipatoryEnabled = useStore((s) => s.anticipatoryEnabled);
  const setAnticipatoryEnabled = useStore((s) => s.setAnticipatoryEnabled);
  const setTripState = useStore((s) => s.setTripState);
  const setAlert = useStore((s) => s.setAlert);

  const maxFlow = catalogue?.max_flow_lpm ?? 14;
  const minFlow = catalogue?.min_flow_lpm ?? 0.8;
  const rampRate = 0.22; // L/min per second (matches backend RAMP_RATE_LPM_PER_S)

  const spec = useMemo(
    () => catalogue?.chemistries.find((c) => c.key === chemistry),
    [catalogue, chemistry],
  );

  const [requestedFlow, setRequestedFlow] = useState<number>(inputs.baseFlowLpm);
  const [appliedFlow, setAppliedFlow] = useState<number>(inputs.baseFlowLpm);
  const [commandMsg, setCommandMsg] = useState<string>('');
  const [manualMode, setManualMode] = useState(false);
  const lastTickRef = useRef<number>(Date.now());

  // Sync the slider when the store's set-point changes from elsewhere.
  useEffect(() => {
    if (!manualMode) {
      setRequestedFlow(inputs.baseFlowLpm);
      setAppliedFlow(inputs.baseFlowLpm);
    }
  }, [inputs.baseFlowLpm, manualMode]);

  // Local rate limiter: advance `applied` toward `requested` at `rampRate`.
  useEffect(() => {
    if (!manualMode) return;
    const id = setInterval(() => {
      const now = Date.now();
      const dt = Math.min((now - lastTickRef.current) / 1000, 1.0);
      lastTickRef.current = now;
      setAppliedFlow((prev) => {
        const target = clamp(requestedFlow, minFlow, maxFlow);
        const step = rampRate * dt;
        const next = prev + clamp(target - prev, -step, step);
        return clamp(next, minFlow, maxFlow);
      });
    }, 200);
    return () => clearInterval(id);
  }, [manualMode, requestedFlow, minFlow, maxFlow]);

  // Push the rate-limited command to the actuator.
  useEffect(() => {
    if (!manualMode) return;
    let cancelled = false;
    api
      .setFlow(chemistry, appliedFlow, 'manual')
      .then((resp) => {
        if (cancelled) return;
        setCommandMsg(resp.message);
        setTripState(resp.trip_state);
      })
      .catch((err) => {
        if (cancelled) return;
        setCommandMsg(
          err instanceof api.ApiError ? `${err.message} [${err.tracebackId ?? '-'}]` : String(err),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [manualMode, appliedFlow, chemistry, setTripState]);

  const onReset = useCallback(async () => {
    try {
      const resp = await api.resetSafety(chemistry);
      setTripState(resp.trip_state);
      setAlert({ title: 'Fail-safe reset', body: resp.precondition, severity: 'info' });
    } catch (err) {
      setCommandMsg(String(err));
    }
  }, [chemistry, setAlert, setTripState]);

  const tripped = tripState === 'trip';

  return (
    <div
      className={cn(
        'rounded-xl border border-slate-800 bg-slate-900/50 p-4',
        tripped && 'border-rose-500/50 bg-rose-950/10',
        className,
      )}
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-200">
          <Sliders className="h-4 w-4 text-slate-400" aria-hidden />
          Operating point &amp; actuator
        </h3>
        <span
          className={cn(
            'rounded px-1.5 py-0.5 font-mono text-[10px] font-bold',
            tripped ? 'bg-rose-500/25 text-rose-200' : 'bg-emerald-500/20 text-emerald-200',
          )}
        >
          {tripState.toUpperCase()}
        </span>
      </div>

      {tripped && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-950/40 p-2.5">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-hidden />
          <div className="flex-1">
            <p className="text-xs font-semibold text-rose-200">Fail-safe latched</p>
            <p className="mt-0.5 text-[11px] leading-snug text-rose-200/80">
              The actuator is pinned to maximum flow. Manual commands are rejected until an
              operator reset. In production this reset is interlocked with a physical
              pack-temperature check.
            </p>
          </div>
          <button
            onClick={onReset}
            className="rounded border border-rose-400/40 px-2 py-1 text-[10px] font-semibold text-rose-200 hover:bg-rose-500/20"
          >
            Reset
          </button>
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2">
        <Slider
          label="Ambient temperature"
          unit="degC"
          value={inputs.ambientC}
          min={5}
          max={60}
          step={0.5}
          onChange={(v) => setInput('ambientC', v)}
          hint="Mock weather service reading"
        />
        <Slider
          label="Pack current"
          unit="A"
          value={inputs.packCurrentA}
          min={0}
          max={600}
          step={5}
          onChange={(v) => setInput('packCurrentA', v)}
          hint={`Joule source I^2 R, R at 25 degC = ${fmt(
            (spec?.internal_resistance_reference_ohm ?? 0.0135) * 1000,
            2,
          )} mohm`}
        />
        <Slider
          label="State of charge"
          unit=""
          value={inputs.stateOfCharge}
          min={0.05}
          max={0.95}
          step={0.01}
          format={(v) => `${fmt(v * 100, 0)}%`}
          onChange={(v) => setInput('stateOfCharge', v)}
          hint="Coulomb-counted from the seed value"
        />
        <Slider
          label="Coolant inlet"
          unit="degC"
          value={inputs.coolantInletC}
          min={10}
          max={45}
          step={0.5}
          onChange={(v) => setInput('coolantInletC', v)}
          hint="Chiller / radiator outlet"
        />
      </div>

      {/* Coolant actuator */}
      <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/40 p-3">
        <div className="flex items-center justify-between">
          <label
            htmlFor="flow-slider"
            className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-400"
          >
            <Zap className="h-3.5 w-3.5 text-coolant" aria-hidden />
            Coolant mass flow
          </label>
          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-sm font-bold text-coolant">
              {fmt(manualMode ? appliedFlow : inputs.baseFlowLpm, 2)}
            </span>
            <span className="text-[10px] text-slate-500">L/min</span>
            {manualMode && requestedFlow !== appliedFlow && (
              <span className="text-[10px] text-amber-400">
                &rarr; {fmt(requestedFlow, 2)} pending
              </span>
            )}
          </div>
        </div>

        <input
          id="flow-slider"
          type="range"
          min={minFlow}
          max={maxFlow}
          step={0.1}
          value={manualMode ? requestedFlow : inputs.baseFlowLpm}
          disabled={tripped}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (!manualMode) {
              setManualMode(true);
              setAppliedFlow(inputs.baseFlowLpm);
            }
            setRequestedFlow(v);
          }}
          className={cn(
            'mt-2 h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-sky-400',
            'disabled:cursor-not-allowed disabled:opacity-40',
          )}
        />
        <div className="mt-1 flex justify-between font-mono text-[9px] text-slate-600">
          <span>{fmt(minFlow, 1)}</span>
          <span className="text-slate-500">ramp-limited to {rampRate} L/min/s</span>
          <span>{fmt(maxFlow, 1)}</span>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button
            onClick={() => {
              setManualMode(false);
              setCommandMsg('');
            }}
            disabled={!manualMode}
            className="rounded border border-slate-700 px-2 py-1 text-[10px] font-semibold text-slate-300 hover:bg-slate-800 disabled:opacity-40"
          >
            <RotateCcw className="mr-1 inline h-3 w-3" aria-hidden />
            Return to auto
          </button>
          {manualMode && (
            <button
              onClick={() => {
                setRequestedFlow(appliedFlow);
                setCommandMsg('Ramp target cleared.');
              }}
              className="rounded border border-slate-700 px-2 py-1 text-[10px] font-semibold text-slate-300 hover:bg-slate-800"
            >
              Hold current
            </button>
          )}
        </div>

        {commandMsg && (
          <p className="mt-2 rounded border border-slate-800 bg-slate-900/60 px-2 py-1 text-[10px] leading-snug text-slate-400">
            {commandMsg}
          </p>
        )}
      </div>

      {/* Anticipation A/B */}
      <div className="mt-3 flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2.5">
        <div>
          <label
            htmlFor="anticipatory-toggle"
            className="block text-[11px] font-semibold uppercase tracking-wider text-slate-300"
          >
            Anticipatory heatwave control
          </label>
          <p className="mt-0.5 text-[10px] leading-snug text-slate-500">
            Off = constant {fmt(inputs.baseFlowLpm, 1)} L/min baseline (the A/B arm of the
            study). Re-run the route to measure the peak-temperature difference.
          </p>
        </div>
        <button
          id="anticipatory-toggle"
          role="switch"
          aria-checked={anticipatoryEnabled}
          onClick={() => setAnticipatoryEnabled(!anticipatoryEnabled)}
          className={cn(
            'relative ml-3 h-6 w-11 shrink-0 rounded-full transition-colors',
            anticipatoryEnabled ? 'bg-amber-500' : 'bg-slate-700',
          )}
        >
          <span
            className={cn(
              'absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform',
              anticipatoryEnabled ? 'translate-x-[1.375rem]' : 'translate-x-0.5',
            )}
          />
        </button>
      </div>

      <button
        onClick={onRunRoute}
        disabled={running}
        className={cn(
          'mt-3 w-full rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors',
          running
            ? 'cursor-wait bg-slate-800 text-slate-500'
            : 'bg-sky-600 text-white hover:bg-sky-500',
        )}
      >
        <Play className="mr-1.5 inline h-4 w-4" aria-hidden />
        {running ? 'Integrating 240 s route...' : 'Run anticipatory rollout'}
      </button>
    </div>
  );
}

function Slider({
  label,
  unit,
  value,
  min,
  max,
  step,
  onChange,
  hint,
  format,
}: {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  hint?: string;
  format?: (value: number) => string;
}) {
  const id = `slider-${label.replace(/\s+/g, '-').toLowerCase()}`;
  const display = format ? format(value) : `${value.toFixed(step < 1 ? 1 : 0)}${unit ? ` ${unit}` : ''}`;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-[11px] font-medium text-slate-400">
          {label}
        </label>
        <span className="font-mono text-xs font-semibold text-slate-200">{display}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1.5 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-sky-400"
      />
      <div className="mt-0.5 flex justify-between font-mono text-[9px] text-slate-600">
        <span>{min}</span>
        {hint && (
          <span className="max-w-[60%] truncate text-center" title={hint}>
            {hint}
          </span>
        )}
        <span>{max}</span>
      </div>
    </div>
  );
}