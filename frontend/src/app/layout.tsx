import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Heatwave-Aware AI-Assisted Microchannel BTMS | EV Battery Thermal Dashboard',
  description:
    'Dual-chemistry (NMC/LFP) physics-informed neural network for the 3-D temperature field of a liquid-cooled microchannel battery pack, with anticipatory coolant control ahead of heatwave entry.',
  applicationName: 'Heatwave-Aware Microchannel BTMS',
  authors: [{ name: 'GenAI Lab -- Semester 7 Mini Project' }],
  keywords: [
    'battery thermal management',
    'physics-informed neural network',
    'microchannel cold plate',
    'electric vehicle',
    'NMC',
    'LFP',
    'thermal runaway',
    'anticipatory control',
    'NASA battery dataset',
  ],
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: '#020617',
  width: 'device-width',
  initialScale: 1,
  colorScheme: 'dark',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} dark`}>
      <body className="min-h-screen font-sans">{children}</body>
    </html>
  );
}