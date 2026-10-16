/** App-router loading boundary (shown during the dynamic import of Dashboard). */
export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-sky-400" />
        <p className="mt-3 text-sm text-slate-400">Initialising dashboard...</p>
      </div>
    </div>
  );
}