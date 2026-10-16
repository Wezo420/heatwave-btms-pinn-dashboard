'use client';

/**
 * `useDashboardData` -- orchestrates the REST side of the dashboard.
 *
 * Responsibilities:
 *  - bootstrap on mount and on chemistry change (single round-trip);
 *  - refetch the PINN field when the operating inputs settle;
 *  - fetch the anticipatory forecast (advisory) whenever inputs change;
 *  - fetch the enforced rollout on demand (the "Run route" button);
 *  - poll the NASA calibration report once at startup.
 *
 * Each fetch is generation-tagged so a slow response for chemistry NMC cannot
 * overwrite a fast response for LFP (a classic race in tabbed dashboards).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from '@/lib/api';
import { useDebouncedValue } from './useTelemetry';
import { useStore } from '@/store/useStore';
import type { ChemistryKey } from '@/lib/types';

export interface DashboardDataState {
  bootstrapping: boolean;
  runningTrajectory: boolean;
  error: string | null;
  runTrajectory: (overrides?: { enableAnticipatory?: boolean }) => Promise<void>;
  refetchField: () => Promise<void>;
  refetchCalibration: () => Promise<void>;
  refetchComparison: () => Promise<void>;
}

export function useDashboardData(): DashboardDataState {
  const chemistry = useStore((s) => s.chemistry);
  const inputs = useStore((s) => s.inputs);
  const anticipatoryEnabled = useStore((s) => s.anticipatoryEnabled);

  const [bootstrapping, setBootstrapping] = useState(true);
  const [runningTrajectory, setRunningTrajectory] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const genRef = useRef(0);
  const debouncedInputs = useDebouncedValue(inputs, 350);

  const setBootstrap = useStore((s) => s.setBootstrap);
  const setField = useStore((s) => s.setField);
  const setFieldLoading = useStore((s) => s.setFieldLoading);
  const setFieldError = useStore((s) => s.setFieldError);
  const setAdvisory = useStore((s) => s.setAdvisory);
  const setTrajectory = useStore((s) => s.setTrajectory);
  const setTrajectoryLoading = useStore((s) => s.setTrajectoryLoading);
  const setCalibration = useStore((s) => s.setCalibration);
  const setComparison = useStore((s) => s.setComparison);

  /* -- bootstrap ---------------------------------------------------- */
  useEffect(() => {
    const gen = ++genRef.current;
    setBootstrapping(true);
    setError(null);
    api
      .getBootstrap(chemistry)
      .then((data) => {
        if (gen !== genRef.current) return;
        setBootstrap(data);
      })
      .catch((err: unknown) => {
        if (gen !== genRef.current) return;
        const message =
          err instanceof api.ApiError
            ? `${err.message}${err.tracebackId ? ` [${err.tracebackId}]` : ''}`
            : String(err);
        setError(message);
      })
      .finally(() => {
        if (gen === genRef.current) setBootstrapping(false);
      });
  }, [chemistry, setBootstrap]);

  /* -- field preview ------------------------------------------------- */
  const refetchField = useCallback(async () => {
    const gen = ++genRef.current;
    setFieldLoading(true);
    try {
      const data = await api.getFieldPreview({
        chemistry,
        nx: 18,
        ny: 12,
        nz: 8,
        coolant_inlet_c: debouncedInputs.coolantInletC,
        coolant_flow_lpm: debouncedInputs.baseFlowLpm,
        ambient_c: debouncedInputs.ambientC,
        pack_current_a: debouncedInputs.packCurrentA,
        state_of_charge: debouncedInputs.stateOfCharge,
      });
      if (gen !== genRef.current) return;
      setField(data);
    } catch (err) {
      if (gen !== genRef.current) return;
      setFieldError(
        err instanceof api.ApiError ? err.message : `Field query failed: ${String(err)}`,
      );
    } finally {
      if (gen === genRef.current) setFieldLoading(false);
    }
  }, [
    chemistry,
    debouncedInputs.coolantInletC,
    debouncedInputs.baseFlowLpm,
    debouncedInputs.ambientC,
    debouncedInputs.packCurrentA,
    debouncedInputs.stateOfCharge,
    setField,
    setFieldLoading,
    setFieldError,
  ]);

  useEffect(() => {
    void refetchField();
  }, [refetchField]);

  /* -- advisory forecast -------------------------------------------- */
  useEffect(() => {
    const gen = ++genRef.current;
    let cancelled = false;

    (async () => {
      try {
        const scenario = await api.getScenario(chemistry, 240, 90);
        if (cancelled || gen !== genRef.current) return;
        const forecast = await api.postForecast({
          route: scenario.route,
          weather: scenario.weather,
          chemistry,
          pack_heat_w: debouncedInputs.packCurrentA * 0.02,
          state_of_charge: debouncedInputs.stateOfCharge,
        });
        if (cancelled || gen !== genRef.current) return;
        setAdvisory(forecast.schedule, forecast.zones);
      } catch {
        // Advisory is best-effort; the enforced path is what matters for safety.
        if (!cancelled) setAdvisory([], []);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    chemistry,
    debouncedInputs.packCurrentA,
    debouncedInputs.stateOfCharge,
    setAdvisory,
  ]);

  /* -- enforced rollout (on demand) --------------------------------- */
  const runTrajectory = useCallback(
    async (overrides?: { enableAnticipatory?: boolean }) => {
      setRunningTrajectory(true);
      setTrajectoryLoading(true);
      try {
        const scenario = await api.getScenario(chemistry, 240, 120);
        const data = await api.postTrajectory({
          chemistry,
          duration_s: scenario.duration_s,
          dt_s: scenario.dt_s,
          route: scenario.route,
          weather: scenario.weather,
          base_current_a: inputs.packCurrentA,
          base_flow_lpm: inputs.baseFlowLpm,
          initial_cell_temp_c: 34,
          initial_soc: inputs.stateOfCharge,
          enable_anticipatory: overrides?.enableAnticipatory ?? anticipatoryEnabled,
          enable_learned_field: true,
          seed: 1337,
        });
        setTrajectory(data);
      } catch (err) {
        setError(
          err instanceof api.ApiError
            ? `${err.message}${err.tracebackId ? ` [${err.tracebackId}]` : ''}`
            : String(err),
        );
      } finally {
        setRunningTrajectory(false);
        setTrajectoryLoading(false);
      }
    },
    [
      chemistry,
      inputs.packCurrentA,
      inputs.baseFlowLpm,
      inputs.stateOfCharge,
      anticipatoryEnabled,
      setTrajectory,
      setTrajectoryLoading,
    ],
  );

  /* -- NASA calibration + comparison -------------------------------- */
  const refetchCalibration = useCallback(async () => {
    try {
      const data = await api.getCalibration();
      setCalibration(data);
    } catch {
      /* the provenance card degrades to a placeholder */
    }
  }, [setCalibration]);

  const refetchComparison = useCallback(async () => {
    try {
      const data = await api.getComparison();
      setComparison(data);
    } catch {
      /* the comparison card degrades to a placeholder */
    }
  }, [setComparison]);

  useEffect(() => {
    void refetchCalibration();
  }, [refetchCalibration]);

  return {
    bootstrapping,
    runningTrajectory,
    error,
    runTrajectory,
    refetchField,
    refetchCalibration,
    refetchComparison,
  };
}

/** Convenience: current chemistry typed for API calls. */
export function useChemistry(): ChemistryKey {
  return useStore((s) => s.chemistry);
}