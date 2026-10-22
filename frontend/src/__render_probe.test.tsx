import { describe, it, expect, beforeAll } from 'vitest';
import { render, act } from '@testing-library/react';

beforeAll(() => {
  class FakeWS { static OPEN=1; readyState=0; onopen:any=null; onmessage:any=null; onerror:any=null; onclose:any=null; send(){} close(){} }
  (globalThis as any).WebSocket = FakeWS;
  (globalThis as any).ResizeObserver = class { observe(){} unobserve(){} disconnect(){} };
});

const candidates = [
  'StatTiles', 'Header', 'ControlPanel', 'NasaCalibrationCard',
  'ChemistryComparisonCard', 'TelemetryChart', 'CellTemperatureChart',
  'ThermalFieldMap', 'ActuationTimeline', 'ResidualPanel', 'ChemistryToggle',
];

describe.each(candidates)('isolate %s', (name) => {
  it(`mounts alone without an update loop`, async () => {
    const mod = await import(`@/components/${name}`);
    const C = mod.default;
    const { container } = render(<C />);
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });
    expect(container.querySelector('*')).toBeTruthy();
  }, 15000);
});
