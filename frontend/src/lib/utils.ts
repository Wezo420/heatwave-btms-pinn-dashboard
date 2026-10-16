import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Merge Tailwind class names, letting later classes win conflicts. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a number with fixed decimals and an optional SI-ish suffix. */
export function fmt(value: number | null | undefined, digits = 2, suffix = ''): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '--';
  return `${value.toFixed(digits)}${suffix}`;
}

/** Format millisecond latency with an appropriate precision. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '--';
  if (ms < 1) return `${(ms * 1000).toFixed(0)} us`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** Format a coulomb/energy quantity in Wh or kWh depending on magnitude. */
export function fmtEnergy(wh: number): string {
  if (!Number.isFinite(wh)) return '--';
  return Math.abs(wh) >= 1000 ? `${(wh / 1000).toFixed(2)} kWh` : `${wh.toFixed(0)} Wh`;
}

/** Format seconds as m:ss. */
export function fmtClock(seconds: number): string {
  if (!Number.isFinite(seconds)) return '00:00';
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${m.toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;
}

/** Clamp a value into [lo, hi]. */
export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

/** Linear interpolation between two stops. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Map a temperature to a colour on the shared thermal ramp.
 *
 * The same ramp is used by the heat-map cells, the KPI sparkline and the
 * legend, so a colour always carries the same meaning. `domain` is
 * `[tMin, tMax]` in degC and is clamped, so a runaway outlier saturates at the
 * end of the ramp instead of wrapping or returning undefined.
 */
const THERMAL_STOPS: Array<[number, string]> = [
  [0.0, '#0b1d3a'],
  [0.15, '#0e4a6e'],
  [0.3, '#0e7490'],
  [0.45, '#22c55e'],
  [0.6, '#eab308'],
  [0.75, '#f97316'],
  [0.87, '#ef4444'],
  [1.0, '#dc2626'],
];

export function thermalColor(tempC: number, domain: [number, number]): string {
  const [lo, hi] = domain;
  const span = Math.max(hi - lo, 1e-6);
  const t = clamp((tempC - lo) / span, 0, 1);
  for (let i = 1; i < THERMAL_STOPS.length; i += 1) {
    const [t1, c1] = THERMAL_STOPS[i];
    const [t0, c0] = THERMAL_STOPS[i - 1];
    if (t <= t1) {
      return mixHex(c0, c1, (t - t0) / Math.max(t1 - t0, 1e-6));
    }
  }
  return THERMAL_STOPS[THERMAL_STOPS.length - 1][1];
}

function mixHex(a: string, b: string, t: number): string {
  const pa = hexToRgb(a);
  const pb = hexToRgb(b);
  const r = Math.round(lerp(pa[0], pb[0], t));
  const g = Math.round(lerp(pa[1], pb[1], t));
  const bl = Math.round(lerp(pa[2], pb[2], t));
  return `rgb(${r},${g},${bl})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

/** Accent colour for a chemistry key. */
export function chemistryColor(key: 'nmc' | 'lfp'): string {
  return key === 'nmc' ? '#a78bfa' : '#5eead4';
}

/** Tailwind classes for the chemistry chip. */
export function chemistryClasses(key: 'nmc' | 'lfp'): string {
  return key === 'nmc'
    ? 'bg-nmc-bg text-nmc border-nmc/40'
    : 'bg-lfp-bg text-lfp border-lfp/40';
}

/**
 * Exponential moving average.
 *
 * Used for the displayed latency and flow-rate readouts: the raw per-frame
 * values are noisy enough that showing them raw makes the dashboard look
 * unstable even when the loop is fine.
 */
export function ema(prev: number, next: number, alpha: number): number {
  return alpha * next + (1 - alpha) * prev;
}

/** Percentile of a numeric array (linear interpolation). */
export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = clamp((sorted.length - 1) * p, 0, sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return lerp(sorted[lo], sorted[hi], idx - lo);
}

/** Round to a given number of decimals (avoids float noise in the UI). */
export function round(value: number, digits = 2): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

/** Truncate a long string with an ellipsis. */
export function truncate(text: string, max = 48): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}\u2026`;
}

/** Format a large parameter count (e.g. 214_553 -> "214.6 k"). */
export function fmtParams(n: number): string {
  if (!Number.isFinite(n)) return '--';
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} k`;
  return n.toString();
}