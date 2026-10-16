r"""
backend.routers.nasa
====================

``/api/nasa`` -- prognostics-dataset provenance and calibration report.

This router is what makes the NASA integration *inspectable*.  A claim in the
model card such as "``E_a = 24.1 +/- 1.8 kJ/mol``" is backed by a GET that
returns the fit coefficients, their standard errors, the goodness of fit in
``ln R``, the per-cell capacity-fade parameters and an explicit boolean saying
whether the real NASA ``.mat`` files were present or the synthetic fallback was
used.  A dashboard that cannot distinguish the two is lying to its user.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..config import NASA_DATA_ROOT, NMC_CELL_IDS
from ..models.nasa_dataset import (
    NASABatteryDataset,
    get_calibration,
    load_or_fit_calibration,
    prior_mean_resistance,
    synthetic_run_to_failure,
)
from ..models.nasa_dataset import _read_mat
from ..config import CHEMISTRY_REGISTRY, get_chemistry
from ..schemas import NacaCalibrationCell, NacaCalibrationResponse, utcnow_iso

router = APIRouter(prefix="/api/nasa", tags=["nasa-prognostics"])


def _cells_report(fit, dataset: NASABatteryDataset) -> List[NacaCalibrationCell]:
    """Per-cell records; for the global fit, cells share the pooled coefficients."""
    out: List[NacaCalibrationCell] = []
    for cid, fade in sorted(fit.cell_capacity.items()):
        out.append(
            NacaCalibrationCell(
                cell_id=cid,
                cycles_available=fade.n_cycles,
                n_operating_points=fit.n_bins,
                r_int_ref_ohm=round(fit.r_ref_25c, 6),
                r_int_std_ohm=round(fit.r_std_ohm, 6),
                r_int_25c_ohm=round(fit.predict(25.0, 0.5).item(), 6)
                if hasattr(fit.predict(25.0, 0.5), "item") else round(float(fit.predict(25.0, 0.5)), 6),
                ea_j_mol=round(fit.ea_j_mol, 1),
                rmse_voltage=round(fit.rmse_log_r, 5),
                capacity_fade_pct=round(fade.capacity_fade_pct, 3),
                eol_capacity_pct=round(fade.eol_capacity_pct, 3),
            )
        )
    if not out:
        out.append(
            NacaCalibrationCell(
                cell_id="POOLED", cycles_available=len(dataset.cycles),
                n_operating_points=fit.n_bins, r_int_ref_ohm=round(fit.r_ref_25c, 6),
                r_int_std_ohm=round(fit.r_std_ohm, 6), r_int_25c_ohm=0.0,
                ea_j_mol=round(fit.ea_j_mol, 1), rmse_voltage=round(fit.rmse_log_r, 5),
                capacity_fade_pct=0.0, eol_capacity_pct=100.0,
            )
        )
    return out


@router.get("/calibration", response_model=NacaCalibrationResponse,
            summary="Internal-resistance calibration report")
def calibration() -> NacaCalibrationResponse:
    """Return the cached (or freshly fitted) ``R_int(T, SOC)`` calibration."""
    root = Path(os.getenv("NASA_BATTERY_DATA_ROOT", str(NASA_DATA_ROOT)))
    present = root.exists() and any(root.glob("B*.mat"))
    fit = get_calibration()

    dataset_present = present
    synthetic = bool(fit.synthetic) or not present
    notes = (
        "Calibrated against the NASA PCoE run-to-failure discharge cycles; "
        f"root scanned: {root}. "
        + ("Real .mat files found. " if dataset_present else
           "No .mat files found -- the deterministic synthetic generator matching "
           "the published NASA statistics was used instead. "
           "Set NASA_BATTERY_DATA_ROOT to the BatteryData directory to use the real "
           "dataset; re-run this endpoint (or POST /api/nasa/calibrate) to refit. ")
        + f"Fit is {'usable' if fit.usable else 'NOT usable (insufficient bins)'}; "
          f"RMSE in ln(R) = {fit.rmse_log_r:.4f}."
    )

    try:
        dataset = NASABatteryDataset(root=root)
    except Exception:
        dataset = NASABatteryDataset(from_synthetic=True)

    return NacaCalibrationResponse(
        dataset_root=str(root),
        dataset_present=dataset_present,
        synthetic_fallback=synthetic,
        cells=_cells_report(fit, dataset),
        aggregate={
            "r0_ohm": round(fit.r0_ohm, 6),
            "r0_stderr_ohm": round(fit.r0_stderr_ohm, 6),
            "ea_j_mol": round(fit.ea_j_mol, 1),
            "ea_stderr_j_mol": round(fit.ea_stderr_j_mol, 1),
            "rmse_log_r": round(fit.rmse_log_r, 5),
            "n_bins": fit.n_bins,
            "soc_quad": round(fit.soc_quad, 4),
            "soc_lin": round(fit.soc_lin, 4),
        },
        coolant_heat_capacity_w_k=round(fit.coolant_heat_capacity(6.0), 1),
        notes=notes,
        timestamp=utcnow_iso(),
    )


@router.post("/calibrate", response_model=NacaCalibrationResponse,
             summary="Force a re-fit of the internal-resistance model")
def recalibrate(force: bool = True) -> NacaCalibrationResponse:
    """Re-run the calibration from scratch and overwrite the cache.

    Use this after dropping the real NASA ``.mat`` files into
    ``backend/data/nasa/BatteryData`` or after changing ``NASA_BATTERY_DATA_ROOT``.
    """
    try:
        root = Path(os.getenv("NASA_BATTERY_DATA_ROOT", str(NASA_DATA_ROOT)))
        dataset = NASABatteryDataset(root=root, from_synthetic=not root.exists())
        from ..models.nasa_dataset import fit_internal_resistance, save_calibration

        fit = fit_internal_resistance(dataset)
        save_calibration(fit)
        # Bust the module-level cache so subsequent GETs see the new fit.
        from ..models.nasa_dataset import _CALIBRATION_CACHE

        _CALIBRATION_CACHE.clear()
    except Exception as exc:
        raise HTTPException(500, f"Calibration failed: {exc!r}") from exc
    return calibration()


@router.get("/dataset/stats", summary="Dataset summary statistics")
def dataset_stats(limit: int = 5) -> dict:
    """Counts, channel statistics and the first few cycles of the dataset."""
    root = Path(os.getenv("NASA_BATTERY_DATA_ROOT", str(NASA_DATA_ROOT)))
    try:
        dataset = NASABatteryDataset(root=root, from_synthetic=not root.exists())
    except Exception as exc:
        raise HTTPException(500, f"Could not build dataset: {exc!r}") from exc

    import numpy as np

    cycles = dataset.cycles[:limit]
    return {
        "describe": dataset.describe(),
        "chemistry_inferred": dataset.chemistry,
        "window_len": dataset.window_len,
        "stride": dataset.stride,
        "stats": {k: {"mean": round(v[0], 6), "std": round(v[1], 6)}
                  for k, v in dataset.stats.items()},
        "first_cycles": [
            {
                "cell_id": c.cell_id, "cycle_index": c.cycle_index, "mode": c.mode,
                "samples": len(c), "capacity_ah": round(c.capacity_ah, 4),
                "current_a": round(c.current_a, 4),
                "c_rate": round(c.c_rate, 4),
                "voltage_range_v": [round(float(c.voltage_v.min()), 4),
                                    round(float(c.voltage_v.max()), 4)],
                "temperature_range_c": [round(float(c.temperature_k.min()) - 273.15, 3),
                                        round(float(c.temperature_k.max()) - 273.15, 3)],
            }
            for c in cycles
        ],
        "timestamp": utcnow_iso(),
    }


@router.get("/probe", summary="List the .mat files actually present on disk")
def probe() -> dict:
    """Enumerate candidate NASA ``.mat`` files and their sizes.

    This is the endpoint the auto-heal agent calls when it sees a
    ``FileNotFoundError`` for a dataset path -- it distinguishes "directory
    missing" from "directory present but empty" from "files present but
    undecodable", which are three different remediations.
    """
    root = Path(os.getenv("NASA_BATTERY_DATA_ROOT", str(NASA_DATA_ROOT)))
    files: List[Dict[str, Any]] = []
    for cid in NMC_CELL_IDS:
        p = root / f"{cid}.mat"
        entry: Dict[str, Any] = {"cell_id": cid, "path": str(p), "present": p.exists()}
        if p.exists():
            entry["size_bytes"] = p.stat().st_size
            try:
                with p.open("rb") as fh:
                    header = fh.read(128)
                entry["mat_header"] = header[:6].decode("ascii", "ignore")
                probe_raw = _read_mat(p)
                entry["keys"] = sorted(probe_raw.keys())
                entry["n_keys"] = len(probe_raw)
                entry["decodable"] = bool(probe_raw)
            except Exception as exc:
                entry["decodable"] = False
                entry["error"] = repr(exc)
        files.append(entry)
    return {
        "root": str(root),
        "root_exists": root.exists(),
        "root_is_dir": root.is_dir() if root.exists() else False,
        "n_mat_files": sum(1 for f in files if f["present"]),
        "files": files,
        "fallback": "synthetic_run_to_failure will be used",
        "timestamp": utcnow_iso(),
    }


@router.get("/chemistry-prior", summary="Blended resistance prior per chemistry")
def chemistry_prior(temp_c: float = 25.0, soc: float = 0.5) -> dict:
    r"""Show the inverse-variance blend of the NASA fit and the datasheet anchor.

    This is the number that initialises ``MicrochannelPINN.r_int_head``; making it
    a GET means the blend weights can be inspected rather than trusted.
    """
    fit = get_calibration()
    out: Dict[str, Any] = {}
    for key, spec in CHEMISTRY_REGISTRY.items():
        blended = prior_mean_resistance(fit, spec, temp_c, soc)
        empirical = float(fit.predict(temp_c, soc)) if fit.usable else float("nan")
        out[key] = {
            "datasheet_ohm": round(spec.r_int(temp_c, soc), 6),
            "nasa_empirical_ohm": round(empirical, 6) if fit.usable else None,
            "blended_prior_ohm": round(blended, 6),
            "usable": fit.usable,
        }
    return {"temp_c": temp_c, "soc": soc, "per_chemistry": out, "timestamp": utcnow_iso()}