"""Shared pytest configuration for the backend test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the test run hermetic: never let a developer's real NASA dataset or a
# stale checkpoint change what the tests observe.
os.environ.setdefault("BTMS_TESTING", "1")
os.environ.setdefault("API_DEVICE", "cpu")


@pytest.fixture(scope="session", autouse=True)
def _artifacts(tmp_path_factory):
    """Redirect artefact writes to a temporary directory for the session."""
    from backend import config as cfg

    tmp = tmp_path_factory.mktemp("btms-artifacts")
    cfg.ARTIFACT_DIR = tmp
    cfg.CHECKPOINT_PATH = tmp / "pinn_dual_chemistry.pt"
    cfg.CALIBRATION_PATH = tmp / "nasa_calibration.json"
    yield tmp


@pytest.fixture(autouse=True)
def _reset_engine():
    """Drop the inference-engine singleton between tests.

    The engine caches the model, the device and the latency tracker; sharing one
    across tests would make a latency assertion depend on test ordering.
    """
    from backend.services.inference import InferenceEngine

    InferenceEngine.reset()
    yield
    InferenceEngine.reset()


@pytest.fixture(autouse=True)
def _reset_actuator():
    """Clear the coolant-actuator state between tests."""
    from backend.routers.control import ACTUATOR

    ACTUATOR.applied_flow_lpm.clear()
    ACTUATOR.applied_flow_lpm.update({"nmc": 6.0, "lfp": 6.0})
    ACTUATOR.mode.clear()
    ACTUATOR.mode.update({"nmc": "auto", "lfp": "auto"})
    ACTUATOR.pending_flow_lpm.clear()
    ACTUATOR.safety.clear()
    ACTUATOR.history.clear()
    yield
    ACTUATOR.safety.clear()