'use client';

/**
 * `Header` -- project title, connection state and provenance badges.
 *
 * The header reports three independent health facts, each with its own colour:
 * API reachable, WebSocket streaming, and PINN checkpoint present. These are
 * separate because they fail separately -- a demo where the API is up but the
 * checkpoint is missing is running an untrained model, and the header says so
 * rather than implying the numbers are research-grade.
 */

import { useEffect, useState } from 'react';
import { Activity, Cpu, Radio, Wifi, WifiOff } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { cn, fmtParams, truncate } from '@/lib/utils';
import { ChemistryChip } from './ChemistryToggle';
import type { TelemetryStatus } from '@/hooks/useTelemetry';

interface Props {
  wsStatus: TelemetryStatus;
  latencyP95: number;
  className?: string;
}

export default function Header({ wsStatus, latencyP95, className }: Props) {
  const chemistry = useStore((s) => s.chemistry);
  const modelInfo = useStore((s) => s.modelInfo);
  const connected = useStore((s) => s.connected);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const res = await fetch('/api/health');
        if (alive) setApiUp(res.ok);
      } catch {
        if (alive) setApiUp(false);
      }
    };
    void check();
    const id = setInterval(check, 20_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <header
      className={cn(
        'sticky top-0 z-30 border-b border-slate-800 bg-slate-950/90 backdrop-blur-md',
        className,
      )}
    >
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <div className="relative rounded-lg bg-gradient-to-br from-sky-500 to-violet-600 p-1.5">
            <Cpu className="h-4 w-4 text-white" aria-hidden />
          </div>
          <div>
            <h1 className="text-sm font-bold leading-tight tracking-tight text-slate-100">
              Heatwave-Aware AI-Assisted Microchannel BTMS
            </h1>
            <p className="text-[10px] leading-tight text-slate-500">
              Dual-chemistry PINN temperature field + anticipatory coolant control for EVs
            </p>
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <ChemistryChip chemistry={chemistry} />

          <Badge
            ok={apiUp === true}
            warn={apiUp === null}
            icon={apiUp ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            label={apiUp === true ? 'API OK' : apiUp === null ? 'API ...' : 'API DOWN'}
          />

          <Badge
            ok={wsStatus === 'open'}
            warn={wsStatus === 'connecting' || wsStatus === 'stale'}
            icon={<Radio className="h-3 w-3" />}
            label={`WS ${wsStatus.toUpperCase()}`}
            pulse={wsStatus === 'open'}
          />

          {connected && (
            <Badge
              ok
              icon={<Activity className="h-3 w-3" />}
              label={`P95 ${fmtParams(Math.round(latencyP95))} ms`}
            />
          )}

          <Badge
            ok={modelInfo?.checkpoint_present ?? false}
            warn={!modelInfo?.checkpoint_present}
            icon={<Cpu className="h-3 w-3" />}
            label={
              modelInfo?.checkpoint_present
                ? `${modelInfo.n_hidden_layers}x${modelInfo.hidden_dim} SIREN`
                : 'UNTRAINED CKPT'
            }
            title={
              modelInfo
                ? truncate(
                    `${modelInfo.architecture} | ${modelInfo.total_parameters} params on ${modelInfo.device} (${modelInfo.dtype})`,
                    120,
                  )
                : 'model info unavailable'
            }
          />

          {modelInfo && (
            <span
              className="hidden font-mono text-[10px] text-slate-600 sm:inline"
              title={`Total trainable parameters: ${modelInfo.total_parameters}`}
            >
              {fmtParams(modelInfo.total_parameters)} params &middot; {modelInfo.device}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

function Badge({
  ok,
  warn = false,
  pulse = false,
  icon,
  label,
  title,
}: {
  ok: boolean;
  warn?: boolean;
  pulse?: boolean;
  icon: React.ReactNode;
  label: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] font-semibold',
        ok
          ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
          : warn
            ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
            : 'border-rose-500/40 bg-rose-500/10 text-rose-300',
      )}
    >
      {pulse && <span className="relative flex h-1.5 w-1.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
      </span>}
      {!pulse && icon}
      {label}
    </span>
  );
}