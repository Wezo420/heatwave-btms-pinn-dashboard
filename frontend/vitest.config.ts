import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

/**
 * Vitest configuration for the dashboard.
 *
 * This file did not previously exist, which meant `npm test` (declared in
 * package.json) found no configuration and silently collected zero tests --
 * a script that looks like it verifies the frontend and does not.
 *
 * `jsdom` is the environment because the components under test are React
 * components that read from the Zustand store, mount `useEffect` hooks and
 * query the DOM. `happy-dom` would be marginally faster but its `WebSocket`
 * and `ResizeObserver` shims differ enough from a browser that the telemetry
 * hook would need environment-specific stubs.
 */
export default defineConfig({
  // Next.js 14 uses the automatic JSX runtime (no `import React` required), and
  // esbuild must match that in the test transform or every component fails
  // with "React is not defined" -- which is a transform artefact, not a bug in
  // the components.
  esbuild: {
    jsx: 'automatic',
    jsxImportSource: 'react',
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    setupFiles: [],
    // The dashboard renders on mount and immediately opens a WebSocket and
    // fires REST requests. jsdom has neither, so the tests that mount it must
    // stub those -- see `src/__render_probe.test.tsx` for the shape.
    testTimeout: 20_000,
    hookTimeout: 20_000,
  },
});
