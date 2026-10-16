'use client';

/**
 * `ResidualPanel` -- the physics-residual audit table.
 *
 * Exists as its own component (rather than being folded into the heat map)
 * because it answers a *methodological* question -- "is this field actually a
 * solution of the governing equations?" -- and that question deserves to be
 * answerable without touching any slider. It also renders the adaptive loss
 * weights from the gradient-penalty balancer, which is the mechanism the model
 * card describes for keeping the PDE and boundary terms balanced.
 */

import { useCallback, useEffect, useState } from 'react';
import { FlaskConical, RefreshCw } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { cn, fmt } from '@/lib/utils';
import * as api from '@/lib/api';
import type { FieldParams } from '@/lib/api';

interface ResidualSnapshot {
  chemistry: string;
  residuals: Record<string, number>;
  adaptive_weights: Record<string, number>;
  inference_latency_ms: number;
  within_latency_budget: boolean;
  exotherm_onset_c: number;
  margin_c: number;
  gradient_penalty_note: string;
  timestamp: string;
}

export default function ResidualPanel({ className }: { className?: string }) {
  const chemistry = useStore((s) => s.chemistry);
  const inputs = useStore((s) => s.inputs);
  const [snap, setSnap] = useState<ResidualSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [manualNonce, setManualNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    const params: FieldParams = {
      chemistry,
      coolant_flow_lpm: inputs.baseFlowLpm,
      coolant_inlet_c: inputs.coolantInletC,
      ambient_c: inputs.ambientC,
      pack_current_a: inputs.packCurrentA,
      state_of_charge: inputs.stateOfCharge,
    };
    api
      .getResiduals(params)
      .then((data) => {
        if (cancelled) return;
        setSnap(data as unknown as ResidualSnapshot);
        setErr(null);
      })
      .catch((e: unknown) => {
        if (!cancelled) setErr(e instanceof api.ApiError ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
    // `manualNonce` forces a re-probe when the refresh button is pressed.
  }, [
    chemistry,
    inputs.baseFlowLpm,
    inputs.coolantInletC,
    inputs.ambientC,
    inputs.packCurrentA,
    inputs.stateOfCharge,
    manualNonce,
  ]);

  const probe = useCallback(() => setManualNonce((n) => n + 1), []);

  if (err) {
    return (
      <div className={cn('rounded-xl border border-rose-800/60 bg-rose-950/20 p-3 text-xs text-rose-300', className)}>
        Residual probe failed: {err}
      </div>
    );
  }

  if (!snap) return null;

  const rows: Array<[string, number, string]> = [
    ['Solid energy (PDE)', snap.residuals.pde_energy, 'rho cp dT/dt - div(k grad T) - I^2R/V'],
    ['Momentum (NS)', snap.residuals.pde_momentum, 'mu lap(u) - dp/dx'],
    ['Continuity', snap.residuals.pde_continuity, 'div(u) = 0'],
    ['Inlet Dirichlet', snap.residuals.bc_inlet, 'T(0) = T_coolant_in'],
    ['Outlet convective', snap.residuals.bc_outlet, '-k dT/dn = h (T - T_out)'],
    ['Shell convection', snap.residuals.bc_wall, '-k dT/dn = h_air (T - T_amb) - q_rad'],
    ['Interface flux', snap.residuals.bc_interface_flux, 'k_s dT_s/dn = k_f dT_f/dn'],
    ['TOTAL', snap.residuals.total_physics, 'sum of all terms'],
  ];

  const weights = snap.adaptive_weights ?? {};

  return (
    <div className={cn('rounded-xl border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-200">
          <FlaskConical className="h-4 w-4 text-violet-400" aria-hidden />
          Physics residual audit
        </h3>
        <button
          onClick={probe}
          className="rounded border border-slate-700 px-1.5 py-1 text-slate-400 hover:bg-slate-800"
          title="Re-probe"
        >
          <RefreshCw className={cn('h-3 w-3', busy && 'animate-spin')} aria-hidden />
        </button>
      </div>

      <table className="w-full text-[10px]">
        <thead>
          <tr className="border-b border-slate-800 text-slate-500">
            <th className="py-1 text-left font-medium">Term</th>
            <th className="py-1 text-right font-medium">MSE</th>
            <th className="hidden py-1 pl-3 text-left font-medium sm:table-cell">Definition</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.map(([label, value, def]) => {
            const mag = Number.isFinite(value) ? Math.abs(value) : Infinity;
            const tone =
              mag < 1e-3 ? 'text-emerald-400' : mag < 1 ? 'text-amber-400' : 'text-rose-400';
            return (
              <tr key={label} className="border-b border-slate-800/40 last:border-0">
                <td className="py-0.5 font-sans text-slate-300">{label}</td>
                <td className={cn('py-0.5 text-right', tone)}>
                  {Number.isFinite(value)
                    ? mag < 1e-3 && mag > 0
                      ? value.toExponential(2)
                      : mag.toFixed(5)
                    : 'NaN'}
                </td>
                <td className="hidden py-0.5 pl-3 font-sans text-slate-600 sm:table-cell">{def}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {Object.keys(weights).length > 0 && (
        <div className="mt-2 rounded border border-slate-800 bg-slate-950/50 px-2 py-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Adaptive loss weights (gradient-norm penalty)
          </div>
          <div className="mt-1 flex flex-wrap gap-2">
            {Object.entries(weights).map(([k, v]) => (
              <span key={k} className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[9px] text-slate-300">
                lambda_{k} = {fmt(v as number, 4)}
              </span>
            ))}
          </div>
          <p className="mt-1 text-[9px] leading-snug text-slate-600">{snap.gradient_penalty_note}</p>
        </div>
      )}

      <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500">
        <span>
          margin to onset: <span className="font-mono text-slate-300">{fmt(snap.margin_c, 1)} K</span>
        </span>
        <span
          className={cn(
            'rounded px-1.5 py-0.5 font-mono font-bold',
            snap.within_latency_budget ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300',
          )}
        >
          {fmt(snap.inference_latency_ms, 1)} ms
        </span>
      </div>
    </div>
  );
}

