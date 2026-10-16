import Link from 'next/link';

/** 404 page -- kept minimal; the app is a single route. */
export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="text-center">
        <p className="font-mono text-5xl font-bold text-slate-700">404</p>
        <h1 className="mt-2 text-lg font-semibold text-slate-300">Route not found</h1>
        <p className="mt-1 text-sm text-slate-500">
          This dashboard is a single-page application.
        </p>
        <Link
          href="/"
          className="mt-5 inline-block rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-500"
        >
          Back to dashboard
        </Link>
      </div>
    </div>
  );
}