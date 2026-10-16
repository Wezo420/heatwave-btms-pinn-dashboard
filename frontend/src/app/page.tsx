import dynamic from 'next/dynamic';
import { DashboardSkeleton } from '@/components/Dashboard';

/**
 * The dashboard is a fully client-side application: the WebSocket telemetry
 * stream, the Zustand store and the Recharts plots all require the browser. We
 * therefore `ssr: false` the whole tree and render a skeleton on the server.
 *
 * This is deliberate rather than incidental. The 3-D field tensor, the frame
 * history and the actuation schedule are all large client-side state; hydrating
 * them from an SSR payload would ship megabytes of JSON for content that only
 * exists after the first WebSocket frame anyway. The skeleton preserves layout
 * stability while the client takes over.
 */
const Dashboard = dynamic(() => import('@/components/Dashboard').then((m) => m.default), {
  ssr: false,
  loading: () => <DashboardSkeleton />,
});

export default function Page() {
  return <Dashboard />;
}