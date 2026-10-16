/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Rewrites keep the browser on a single origin for REST calls, which removes
  // the need for CORS preflights on every telemetry frame.
  //
  // There is deliberately **no** `/ws` rewrite. Next.js validates that a
  // rewrite `destination` begins with `/`, `http://` or `https://`, so a
  // `ws://` destination is rejected outright -- "Invalid rewrite found" -- and
  // `next build` fails. Rewrites cannot carry an `Upgrade: websocket` header
  // regardless, so the WebSocket telemetry stream connects directly to the
  // backend; see `telemetryWsUrl` in src/lib/api.ts.
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL ?? 'http://127.0.0.1:8000'}/api/:path*`,
      },
    ];
  },

  env: {
    NEXT_PUBLIC_BACKEND_URL:
      process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://127.0.0.1:8000',
    NEXT_PUBLIC_BACKEND_WS_URL:
      process.env.NEXT_PUBLIC_BACKEND_WS_URL ??
      process.env.NEXT_PUBLIC_BACKEND_URL ??
      'ws://127.0.0.1:8000',
  },

  // Recharts + d3-scale pull in optional Node builtins that must not be
  // bundled into the client runtime.
  webpack: (config) => {
    config.externals.push({
      canvas: false,
      fs: false,
      net: false,
      tls: false,
    });
    return config;
  },

  // The heat-map renders a 3-D tensor to JSON; allow a generous body.
  experimental: {
    serverActions: { bodySizeLimit: '4mb' },
  },

  eslint: {
    ignoreDuringBuilds: false,
  },
};

export default nextConfig;