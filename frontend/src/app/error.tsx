'use client';

import { useEffect } from 'react';
import { AlertOctagon, RotateCcw } from 'lucide-react';

/**
 * Route-level error boundary.
 *
 * `reset()` re-renders the segment without a full reload, which is the right
 * recovery for a transient fetch or WebSocket failure; `reload()` is offered for
 * the case where client state has been corrupted (the store is module-level, so
 * a hard reload is the only way to reset it).
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.error('[dashboard] unhandled error', error);
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="max-w-lg rounded-2xl border border-rose-800/60 bg-rose-950/20 p-6 text-center">
        <AlertOctagon className="mx-auto h-10 w-10 text-rose-400" aria-hidden />
        <h1 className="mt-3 text-lg font-bold text-rose-200">Dashboard crashed</h1>
        <p className="mt-2 text-sm text-rose-200/80">{error.message}</p>
        {error.digest && (
          <p className="mt-1 font-mono text-xs text-slate-500">digest: {error.digest}</p>
        )}
        <p className="mt-3 text-xs leading-snug text-slate-500">
          Check that the FastAPI backend is running on port 8000 (
          <code className="rounded bg-slate-800 px-1">uvicorn backend.main:app --port 8000</code>)
          and that <code className="rounded bg-slate-800 px-1">NEXT_PUBLIC_BACKEND_URL</code> points
          at it.
        </p>
        <div className="mt-4 flex justify-center gap-2">
          <button
            onClick={reset}
            className="inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-500"
          >
            <RotateCcw className="h-4 w-4" aria-hidden />
            Retry
          </button>
          <button
            onClick={() => window.location.reload()}
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-slate-800"
          >
            Hard reload
          </button>
        </div>
      </div>
    </div>
  );
}