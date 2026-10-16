"""Model layer: PINN, physics operators, and the NASA prognostics dataset."""

from .pinn import (
    DualChemistryLoss,
    FieldOutput,
    GradientNormBalancer,
    MicrochannelPINN,
    build_model,
    chemistry_invariance_report,
    load_checkpoint,
    save_checkpoint,
)
from .nasa_dataset import (
    InternalResistanceFit,
    NASABatteryDataset,
    SupervisedTelemetry,
    build_supervised_telemetry,
    fit_internal_resistance,
    get_calibration,
    get_dataset,
    load_or_fit_calibration,
    prior_mean_resistance,
    synthetic_run_to_failure,
)

__all__ = [
    "DualChemistryLoss",
    "FieldOutput",
    "GradientNormBalancer",
    "MicrochannelPINN",
    "build_model",
    "chemistry_invariance_report",
    "load_checkpoint",
    "save_checkpoint",
    "InternalResistanceFit",
    "NASABatteryDataset",
    "SupervisedTelemetry",
    "build_supervised_telemetry",
    "fit_internal_resistance",
    "get_calibration",
    "get_dataset",
    "load_or_fit_calibration",
    "prior_mean_resistance",
    "synthetic_run_to_failure",
]