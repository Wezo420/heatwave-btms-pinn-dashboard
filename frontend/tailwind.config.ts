import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: [
    './src/app/**/*.{ts,tsx}',
    './src/components/**/*.{ts,tsx}',
    './src/hooks/**/*.{ts,tsx}',
    './src/lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Thermal ramp: used identically by the heat-map, the KPI tiles and
        // the legend, so "orange" always means the same temperature.
        thermal: {
          0: '#0b1d3a',
          10: '#0e4a6e',
          20: '#0e7490',
          30: '#0891b2',
          40: '#22c55e',
          50: '#eab308',
          60: '#f97316',
          70: '#ef4444',
          80: '#dc2626',
          90: '#a21caf',
          100: '#f8fafc',
        },
        // Chemistry accent colours (NMC = violet, LFP = teal) -- chosen to be
        // distinguishable under deuteranopia and in greyscale print.
        nmc: {
          DEFAULT: '#a78bfa',
          dim: '#7c3aed',
          bg: '#1e1b4b',
        },
        lfp: {
          DEFAULT: '#5eead4',
          dim: '#0d9488',
          bg: '#042f2e',
        },
        coolant: '#38bdf8',
        alert: '#fb7185',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 24px -4px rgb(56 189 248 / 0.45)',
        'glow-alert': '0 0 32px -4px rgb(251 113 133 / 0.55)',
      },
      keyframes: {
        pulseRing: {
          '0%': { boxShadow: '0 0 0 0 rgba(251,113,133,0.55)' },
          '70%': { boxShadow: '0 0 0 14px rgba(251,113,133,0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(251,113,133,0)' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'pulse-ring': 'pulseRing 2.4s cubic-bezier(0.4,0,0.6,1) infinite',
        'slide-up': 'slideUp 320ms ease-out',
        shimmer: 'shimmer 2s infinite',
      },
    },
  },
  plugins: [],
};

export default config;