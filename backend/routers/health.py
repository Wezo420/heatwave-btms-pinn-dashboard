r"""
backend.routers.health
======================

``/api/health``, ``/api/catalog/chemistries``, ``/api/model/info`` --
liveness, the chemistry catalogue, and model introspection.

``/api/catalog/chemistries`` is the single source of truth for the dashboard's
chemistry toggle: material properties, thermal-runaway threshold, reference
internal resistance, channel geometry and the recommended set-points all come
from here, so the frontend never hard-codes a physics constant.  If NMC and LFP
values change in :mod:`backend.config`, the UI changes with them.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from ..config import (
    CELL_TEMPERATURE_CEILING_C,
    CHECKPOINT_PATH,
    COLD_PLATE,
    CRITICAL_TEMPERATURE_C,
    EMISSIVITY,
    INFERENCE_LATENCY_BUDGET_MS,
    MAX_COOLANT_FLOW_LPM,
    MIN_COOLANT_FLOW_LPM,
    NASA_DATA_ROOT,
    R_GAS,
    SIGMA_SB,
    DEFAULT_CHEMISTRY,
)
from ..schemas import (
    CatalogueResponse,
    ChannelSpec,
    ChemistryInfo,
    CoolantSpec,
    HealthResponse,
    ModelInfoResponse,
    utcnow_iso,
)
from ..services.heatwave import COMFORT_CEILING_C
from ..services.inference import InferenceEngine

router = APIRouter(prefix="/api", tags=["meta"])

#: Process start time, used for the uptime field.
_START = time.time()


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency probe")
def health() -> HealthResponse:
    """Report health of every runtime dependency.

    ``status`` is ``degraded`` (not ``error``) when the PINN checkpoint or the
    NASA dataset is absent, because the API is still fully functional via the
    documented fallbacks -- it just cannot claim research-grade provenance.
    Lying about this is worse than the fallback itself.
    """
    try:
        np.__version__  # noqa: B018
        numpy_v = np.__version__
    except Exception:
        numpy_v = "unavailable"

    checkpoint = CHECKPOINT_PATH.exists()
    nasa_root = NASA_DATA_ROOT
    nasa_present = nasa_root.exists() and any(nasa_root.glob("B*.mat"))

    detail: Dict[str, Any] = {}
    engine_ok = True
    try:
        eng = InferenceEngine.instance()
        detail = {
            "device": str(eng.device),
            "parameters": eng.model.n_parameters(),
            "latency_ema_ms": round(eng.latency.ema, 3),
            "calibration_usable": bool(eng.calibration.usable),
        }
    except Exception as exc:
        engine_ok = False
        detail = {"engine_error": repr(exc)}

    status = "ok"
    if not engine_ok:
        status = "error"
    elif not checkpoint or not nasa_present:
        status = "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_s=round(time.time() - _START, 2),
        device=(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available() else "cpu"
        ),
        torch_version=torch.__version__,
        numpy_version=numpy_v,
        checkpoint_present=checkpoint,
        nasa_dataset_present=nasa_present,
        chemistry_defaults={
            k: v.r_int_ref for k, v in __import__(
                "backend.config", fromlist=["CHEMISTRY_REGISTRY"]
            ).CHEMISTRY_REGISTRY.items()
        },
        timestamp=utcnow_iso(),
        detail=detail,
    )


@router.get("/catalog/chemistries", response_model=CatalogueResponse,
            summary="Chemistry catalogue (drives the UI toggle)")
def catalogue() -> CatalogueResponse:
    """Return every supported chemistry with the constants the UI needs."""
    from ..config import CHEMISTRY_REGISTRY

    infos: List[ChemistryInfo] = []
    for key, spec in CHEMISTRY_REGISTRY.items():
        infos.append(
            ChemistryInfo(
                key=key,
                label=spec.label,
                cathodes=spec.cathodes,
                anode=spec.anode,
                cp_j_kgk=spec.cp,
                thermal_runaway_onset_c=spec.runaway_c,
                internal_resistance_reference_ohm=spec.r_int_ref,
                activation_energy_j_mol=spec.ea,
                nominal_capacity_ah=spec.nominal_capacity_ah,
                energy_density_wh_kg=spec.energy_density_wh_kg,
                thermal_diffusivity_m2_s=round(spec.thermal_diffusivity(), 9),
                coolant=CoolantSpec(
                    density_kg_m3=spec.coolant_props["density"],
                    cp_j_kgk=spec.coolant_props["cp"],
                    k_w_mk=spec.coolant_props["k"],
                    mu_pa_s=spec.coolant_props["mu"],
                    prandtl=spec.coolant_props["prandtl"],
                ),
                channels=ChannelSpec(
                    n_channels=COLD_PLATE.n_channels,
                    channel_width_mm=COLD_PLATE.channel_width_m * 1000.0,
                    channel_depth_mm=COLD_PLATE.channel_depth_m * 1000.0,
                    length_mm=COLD_PLATE.length_m * 1000.0,
                    width_mm=COLD_PLATE.width_m * 1000.0,
                ),
                notes=spec.notes,
                recommended_max_cell_temp_c=COMFORT_CEILING_C[key],
                default_flow_setpoint_lpm=6.0,
            )
        )

    return CatalogueResponse(
        chemistries=infos,
        default=DEFAULT_CHEMISTRY,
        max_flow_lpm=MAX_COOLANT_FLOW_LPM,
        min_flow_lpm=MIN_COOLANT_FLOW_LPM,
        critical_temperature_c=CRITICAL_TEMPERATURE_C,
        inference_latency_budget_ms=INFERENCE_LATENCY_BUDGET_MS,
        physics_constants={
            "stefan_boltzmann": SIGMA_SB,
            "emissivity": EMISSIVITY,
            "gas_constant": R_GAS,
            "cell_heat_capacity_ceiling_c": CELL_TEMPERATURE_CEILING_C,
            "hydraulic_diameter_mm": COLD_PLATE.hydraulic_diameter_m * 1000.0,
            "flow_area_mm2": COLD_PLATE.fluid_area_m2 * 1e6,
        },
    )


@router.get("/model/info", response_model=ModelInfoResponse, summary="PINN introspection")
def model_info() -> ModelInfoResponse:
    """Architecture, parameter count, device, gradient-penalty config."""
    try:
        eng = InferenceEngine.instance()
    except Exception as exc:
        raise HTTPException(503, f"Inference engine unavailable: {exc!r}") from exc
    info = eng.model_info()
    return ModelInfoResponse(**info)  # type: ignore[arg-type]


@router.get("/routes", summary="Self-describing route index")
def routes() -> dict:
    """List every registered route -- a lightweight OpenAPI alternative."""
    from ..main import app

    out = []
    for r in app.routes:
        methods = getattr(r, "methods", None)
        out.append({
            "path": getattr(r, "path", "?"),
            "methods": sorted(methods) if methods else ["WS"],
            "name": getattr(r, "name", "?"),
        })
    return {"routes": sorted(out, key=lambda x: x["path"]),
            "n": len(out), "timestamp": utcnow_iso()}