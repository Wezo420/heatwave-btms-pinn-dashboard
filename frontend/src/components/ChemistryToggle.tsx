'use client';

/**
 * `ChemistryToggle` -- the NMC / LPF selector.
 *
 * This is the primary control of the whole project, so it does more than flip a
 * boolean: it shows, for each chemistry, the two numbers that actually change
 * the physics (thermal-runaway onset and specific heat capacity) alongside the
 * flow set-point the controller will adopt. Switching chemistry is a *model*
 * switch, not a display preference, and the UI makes that legible.
 *
 * Accessibility: implemented as a `radiogroup` with real `role="radio"` buttons
 * and arrow-key navigation, so it is operable from the keyboard and announced
 * correctly by a screen reader.
 */

import { useCallback, useRef } from 'react';
import { Check } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { cn, fmt } from '@/lib/utils';
import type { ChemistryInfo, ChemistryKey } from '@/lib/types';

interface Props {
  chemistries: ChemistryInfo[];
  value: ChemistryKey;
  onChange: (key: ChemistryKey) => void;
  disabled?: boolean;
}

export default function ChemistryToggle({ chemistries, value, onChange, disabled }: Props) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent, index: number) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const nextIndex =
        e.key === 'ArrowRight'
          ? (index + 1) % chemistries.length
          : (index - 1 + chemistries.length) % chemistries.length;
      const next = chemistries[nextIndex];
      if (!next) return;
      onChange(next.key);
      refs.current[nextIndex]?.focus();
    },
    [chemistries, onChange],
  );

  if (chemistries.length === 0) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <span className="inline-block h-4 w-24 animate-pulse rounded bg-slate-700" />
        Loading chemistry catalogue...
      </div>
    );
  }

  return (
    <div
      role="radiogroup"
      aria-label="Battery chemistry"
      className="inline-flex flex-col gap-1.5 rounded-xl border border-slate-700/60 bg-slate-900/60 p-1.5 backdrop-blur sm:flex-row sm:gap-2"
    >
      {chemistries.map((chem, index) => {
        const active = chem.key === value;
        const accent = chem.key === 'nmc' ? 'nmc' : 'lfp';
        return (
          <button
            key={chem.key}
            ref={(el) => {
              refs.current[index] = el;
            }}
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            disabled={disabled}
            onClick={() => onChange(chem.key)}
            onKeyDown={(e) => onKeyDown(e, index)}
            className={cn(
              'group relative min-w-[168px] rounded-lg px-3.5 py-2.5 text-left transition-all duration-200',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950',
              'disabled:cursor-not-allowed disabled:opacity-50',
              active
                ? cn(
                    accent === 'nmc'
                      ? 'bg-nmc-bg shadow-lg shadow-violet-900/30 ring-1 ring-nmc/50'
                      : 'bg-lfp-bg shadow-lg shadow-teal-900/30 ring-1 ring-lfp/50',
                  )
                : 'bg-slate-800/40 hover:bg-slate-800/70 ring-1 ring-transparent hover:ring-slate-600/60',
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div
                  className={cn(
                    'text-sm font-semibold tracking-tight',
                    active ? (accent === 'nmc' ? 'text-nmc' : 'text-lfp') : 'text-slate-200',
                  )}
                >
                  {chem.key.toUpperCase()}
                </div>
                <div className="mt-0.5 text-[11px] leading-tight text-slate-400">
                  {chem.key === 'nmc' ? 'Nickel Manganese Cobalt' : 'Lithium Iron Phosphate'}
                </div>
              </div>
              {active && (
                <Check
                  className={cn('h-4 w-4 shrink-0', accent === 'nmc' ? 'text-nmc' : 'text-lfp')}
                  aria-hidden
                />
              )}
            </div>

            <dl className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[10px]">
              <div>
                <dt className="text-slate-500">Runaway</dt>
                <dd className="font-mono font-semibold text-rose-300">
                  {fmt(chem.thermal_runaway_onset_c, 0)}&deg;C
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Cp</dt>
                <dd className="font-mono font-semibold text-slate-200">
                  {fmt(chem.cp_j_kgk, 0)} J/kgK
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">R&#8320; (25&deg;C)</dt>
                <dd className="font-mono font-semibold text-slate-200">
                  {fmt(chem.internal_resistance_reference_ohm * 1000, 2)} m&Omega;
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Set-point</dt>
                <dd className="font-mono font-semibold text-coolant">
                  {fmt(chem.default_flow_setpoint_lpm, 1)} L/min
                </dd>
              </div>
            </dl>

            {active && (
              <p className="mt-2 border-t border-slate-700/60 pt-1.5 text-[10px] leading-snug text-slate-400">
                {chem.notes}
              </p>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** Compact variant used in the header. */
export function ChemistryChip({ chemistry }: { chemistry: ChemistryKey }) {
  const spec = useStore((s) => s.catalogue?.chemistries.find((c) => c.key === s.chemistry));
  const isNmc = chemistry === 'nmc';
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold',
        isNmc ? 'border-nmc/40 bg-nmc-bg text-nmc' : 'border-lfp/40 bg-lfp-bg text-lfp',
      )}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', isNmc ? 'bg-nmc' : 'bg-lfp')} />
      {chemistry.toUpperCase()}
      {spec && (
        <span className="font-mono text-[10px] font-normal text-slate-400">
          {fmt(spec.cp_j_kgk, 0)} J/kgK
        </span>
      )}
    </span>
  );
}