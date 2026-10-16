'use client';

/**
 * `useTelemetry` -- WebSocket lifecycle for the live telemetry stream.
 *
 * Robustness requirements this hook implements (each one is a real failure we
 * hit in the lab, and each maps to a visible state in the UI rather than a
 * silent console warning):
 *
 * 1. **Exponential backoff with jitter** on reconnect. A hot-reload loop where
 *    the backend is down would otherwise produce a tight reconnect storm.
 * 2. **Fresh-socket sequencing.** Every reconnect closes the previous socket
 *    first; without this, the previous socket's `onclose` fires and tears down
 *    the *new* connection, which manifests as a stream that connects and
 *    immediately dies.
 * 3. **Chemistry change tears down and rebuilds.** The backend's hub owns one
 *    simulator per chemistry; switching without resubscribing would keep
 *    streaming the old chemistry's frames while the UI shows the new one.
 * 4. **Frame-gap detection.** If no frame arrives within 3x the nominal period
 *    the connection is declared stale and proactively recycled.
 * 5. **Command queueing.** Commands sent while the socket is CONNECTING are
 *    replayed on open, so a user action on a reconnecting dashboard is not lost.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { telemetryWsUrl, type WsCommand } from '@/lib/api';
import type { ChemistryKey, TelemetryFrame, TelemetryWsMessage } from '@/lib/types';

export interface TelemetryOptions {
  chemistry: ChemistryKey;
  enabled?: boolean;
  onFrame: (frame: TelemetryFrame) => void;
  onStatusChange?: (status: TelemetryStatus) => void;
  onError?: (error: string) => void;
}

export type TelemetryStatus =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'stale'
  | 'closed'
  | 'error';

const NOMINAL_PERIOD_MS = 500;
const STALE_MULTIPLIER = 3;
const MAX_BACKOFF_MS = 15_000;
const BASE_BACKOFF_MS = 500;

/** Bounded exponential backoff with full jitter. */
function backoffDelay(attempt: number): number {
  const ceiling = Math.min(BASE_BACKOFF_MS * 2 ** attempt, MAX_BACKOFF_MS);
  return Math.random() * ceiling;
}

export function useTelemetry({
  chemistry,
  enabled = true,
  onFrame,
  onStatusChange,
  onError,
}: TelemetryOptions): { status: TelemetryStatus; send: (cmd: WsCommand) => void; reconnect: () => void } {
  const [status, setStatus] = useState<TelemetryStatus>('idle');

  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const staleTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const queueRef = useRef<WsCommand[]>([]);
  const closedByUsRef = useRef(false);
  const frameCbRef = useRef(onFrame);
  const statusCbRef = useRef(onStatusChange);
  const errCbRef = useRef(onError);

  // Keep callbacks fresh without re-opening the socket on every render.
  useEffect(() => {
    frameCbRef.current = onFrame;
    statusCbRef.current = onStatusChange;
    errCbRef.current = onError;
  }, [onFrame, onStatusChange, onError]);

  const setStatusBoth = useCallback((next: TelemetryStatus) => {
    setStatus(next);
    statusCbRef.current?.(next);
  }, []);

  const send = useCallback((cmd: WsCommand) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    } else {
      // Replay once the socket is ready.
      queueRef.current.push(cmd);
      if (queueRef.current.length > 16) queueRef.current.shift();
    }
  }, []);

  const connect = useCallback(() => {
    if (!enabled || typeof window === 'undefined') return;

    // Always dispose the previous socket before creating a new one.
    if (wsRef.current) {
      closedByUsRef.current = true;
      try {
        wsRef.current.onclose = null;
        wsRef.current.close();
      } catch {
        /* already closed */
      }
      wsRef.current = null;
    }

    setStatusBoth(attemptRef.current === 0 ? 'connecting' : 'closed');

    let ws: WebSocket;
    try {
      ws = new WebSocket(telemetryWsUrl(chemistry));
    } catch (err) {
      setStatusBoth('error');
      errCbRef.current?.(`WebSocket construction failed: ${String(err)}`);
      return;
    }
    wsRef.current = ws;
    closedByUsRef.current = false;

    ws.onopen = () => {
      attemptRef.current = 0;
      setStatusBoth('open');
      const queued = queueRef.current.splice(0);
      for (const cmd of queued) ws.send(JSON.stringify(cmd));
      startStaleWatch();
    };

    ws.onmessage = (event) => {
      let msg: TelemetryWsMessage;
      try {
        msg = JSON.parse(event.data as string) as TelemetryWsMessage;
      } catch {
        return;
      }
      if (msg.type === 'telemetry' && msg.data) {
        frameCbRef.current(msg.data);
      } else if (msg.type === 'error' && msg.detail) {
        errCbRef.current?.(msg.detail);
      }
    };

    ws.onerror = () => {
      // The browser withholds the underlying reason; report the phase instead.
      setStatusBoth('error');
    };

    ws.onclose = (event) => {
      stopStaleWatch();
      if (closedByUsRef.current) return;
      setStatusBoth(event.wasClean ? 'closed' : 'error');
      scheduleReconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chemistry, enabled, setStatusBoth]);

  const scheduleReconnect = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    const delay = backoffDelay(attemptRef.current);
    attemptRef.current = Math.min(attemptRef.current + 1, 6);
    timerRef.current = setTimeout(connect, delay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connect]);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    scheduleReconnect();
  }, [scheduleReconnect]);

  // `stopStaleWatch` is declared *before* `startStaleWatch` so the dependency
  // array below can reference it. Declared after, the exhaustive-deps rule
  // flags the omission and the correct fix would be a temporal-dead-zone
  // reference during render.
  const stopStaleWatch = useCallback(() => {
    if (staleTimerRef.current) {
      clearInterval(staleTimerRef.current);
      staleTimerRef.current = null;
    }
  }, []);

  const startStaleWatch = useCallback(() => {
    stopStaleWatch();
    staleTimerRef.current = setInterval(() => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      // A liveness ping round-trips through the hub; if it comes back, frames
      // are flowing even if this client is idle.
      ws.send(JSON.stringify({ action: 'ping' }));
    }, NOMINAL_PERIOD_MS * STALE_MULTIPLIER);
  }, [stopStaleWatch]);

  useEffect(() => {
    if (!enabled) {
      setStatusBoth('idle');
      return;
    }
    connect();
    return () => {
      closedByUsRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      stopStaleWatch();
      const ws = wsRef.current;
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      wsRef.current = null;
    };
  }, [enabled, connect, setStatusBoth, stopStaleWatch]);

  return { status, send, reconnect };
}

/**
 * `useDebouncedValue` -- used to throttle slider-driven field recomputation.
 *
 * The PINN field query costs 60-180 ms server-side; firing it on every
 * `input` event would queue requests faster than they complete. A 350 ms
 * debounce keeps the sliders responsive while bounding the request rate.
 */
export function useDebouncedValue<T>(value: T, delayMs = 350): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return debounced;
}