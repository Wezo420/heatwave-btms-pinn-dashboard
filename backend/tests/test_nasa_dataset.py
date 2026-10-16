"""Tests for the NASA dataset loader, calibration and the auto-heal primitives."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.config import ARTIFACT_DIR, CALIBRATION_PATH
from backend.models.nasa_dataset import (
    InternalResistanceFit,
    NASABatteryDataset,
    build_supervised_telemetry,
    estimate_series_resistance,
    fit_internal_resistance,
    prior_mean_resistance,
    synthetic_run_to_failure,
)


# ---------------------------------------------------------------------------
# Series-resistance estimator
# ---------------------------------------------------------------------------


class TestSeriesResistance:
    def test_estimator_recovers_a_known_resistance(self):
        """A synthetic V-I step of known magnitude must be recovered."""
        rng = np.random.default_rng(0)
        current = np.concatenate([np.zeros(20), np.full(20, 2.0), np.zeros(20)])
        v_open = 4.0
        r_true = 0.02
        voltage = v_open - current * r_true
        temp = 298.15 + rng.normal(0, 0.1, current.size)
        r_est = estimate_series_resistance(voltage, current, temp)
        # The rolling median over the transition should approach r_true.
        assert 0.5 * r_true < float(np.median(r_est[15:30])) < 2.0 * r_true

    def test_estimator_is_bounded_and_finite(self):
        rng = np.random.default_rng(1)
        v = rng.normal(3.7, 0.2, 400)
        i = rng.normal(0, 1.0, 400)
        t = 298.15 + rng.normal(0, 2.0, 400)
        r = estimate_series_resistance(v, i, t)
        assert np.all(np.isfinite(r))
        assert np.all(r > 0)

    def test_estimator_rises_with_temperature(self):
        """R increases with T, so the same waveform at higher T must give higher R."""
        current = np.concatenate([np.zeros(30), np.full(30, 2.0)])
        voltage = 4.0 - current * 0.02
        cold = estimate_series_resistance(voltage, current, np.full(60, 295.0))
        hot = estimate_series_resistance(voltage, current, np.full(60, 315.0))
        assert float(np.median(hot[25:40])) >= float(np.median(cold[25:40]))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


#: Repository root, used by the path-hygiene assertions below.
REPO = Path(__file__).resolve().parents[2]


class TestDataset:
    def test_synthetic_fallback_is_labelled(self):
        ds = NASABatteryDataset(from_synthetic=True, window_len=32, stride=8)
        assert ds.synthetic is True
        assert ds.describe()["synthetic"] is True

    def test_describe_does_not_leak_an_absolute_path(self):
        """``describe()`` feeds a committed artefact, so it must be path-clean.

        ``scripts/train_pinn.py`` writes this dict verbatim into
        ``artifacts/pinn_provenance.json``, which is checked into the repository.
        An absolute ``root`` there records the author's home directory into
        permanent git history, where no later edit to the file can remove it. The
        summary must report a path relative to the repository root.
        """
        described = NASABatteryDataset(from_synthetic=True,
                                       window_len=32, stride=8).describe()
        root = str(described["root"])

        assert "Users" not in root
        assert not root.startswith(("/", "\\", "~"))
        assert ":" not in root            # no drive letter
        assert root.endswith("BatteryData")
        assert json.dumps(described).count(REPO.as_posix()) == 0

    def test_repo_relative_helper_degrades_to_a_basename(self, tmp_path):
        """A path outside the repository collapses to its name, not a full path."""
        from backend.models.nasa_dataset import _repo_relative

        outside = tmp_path / "elsewhere" / "BatteryData"
        outside.mkdir(parents=True)
        assert _repo_relative(outside) == "BatteryData"

        inside = REPO / "backend" / "data" / "nasa" / "BatteryData"
        assert _repo_relative(inside) == "backend/data/nasa/BatteryData"

    def test_missing_root_automatically_falls_back(self, tmp_path):
        ds = NASABatteryDataset(root=tmp_path / "does_not_exist", window_len=32)
        assert ds.synthetic is True
        assert len(ds) > 0

    def test_real_dataset_is_never_mislabelled_as_synthetic(self, tmp_path):
        """If .mat files exist, the loader must not claim a fallback."""
        bdir = tmp_path / "BatteryData"
        bdir.mkdir()
        # A minimal but genuine MAT v5 file containing the expected keys.
        _write_minimal_mat(bdir / "B0005.mat")
        ds = NASABatteryDataset(root=bdir, cell_ids=["B0005"], window_len=16, stride=4)
        assert ds.synthetic is False

    def test_item_shape_and_target(self):
        ds = NASABatteryDataset(from_synthetic=True, window_len=64, stride=8, normalize=True)
        item = ds[0]
        assert item["x"].shape == (64, 5)
        assert item["target"].shape == (64, 1)
        assert torch.isfinite(item["x"]).all()
        assert torch.isfinite(item["target"]).all()

    def test_normalisation_is_exactly_invertible(self):
        """``standardised * std + mean`` recovers the raw value, per channel.

        This is the property that actually matters for a consumer: the reported
        ``channel_means`` and the dataset statistics must undo the transform
        exactly.  Asserting "mean 0, std 1" on a *windowed* sample instead
        would be wrong -- windows cover the cycles unevenly (a stride of 64 over
        220-sample cycles leaves gaps), so the windowed sample is not the
        underlying sample distribution and its spread is legitimately
        different.
        """
        ds = NASABatteryDataset(from_synthetic=True, window_len=64, stride=16,
                                normalize=True)
        item = ds[0]
        x, means = item["x"], item["channel_means"]

        # Invert the standardisation on the four calibrated channels.
        stds = torch.tensor([ds.stats[ds.channels[i]][1] for i in (0, 1, 3, 4)])
        mu = torch.tensor([ds.stats[ds.channels[i]][0] for i in (0, 1, 3, 4)])
        restored = x[:, (0, 1, 3, 4)] * stds + mu

        raw = ds.cycles[0]
        assert torch.allclose(restored[:, 0], raw.voltage_v[:64], atol=1e-4)
        assert torch.allclose(restored[:, 1], raw.temperature_k[:64], atol=1e-3)
        assert torch.allclose(restored[:, 2], raw.soc[:64], atol=1e-4)
        assert torch.allclose(restored[:, 3], raw.resistance_ohm[:64], atol=1e-6)

    def test_standardised_raw_sample_has_unit_variance(self):
        """Over *all* raw samples, each standardised channel has unit spread."""
        ds = NASABatteryDataset(from_synthetic=True, window_len=64, stride=16,
                                normalize=True)
        for idx, name in ((0, "voltage_v"), (1, "temperature_k"),
                          (3, "soc"), (4, "resistance_ohm")):
            mean, std = ds.stats[name]
            values = torch.cat([
                (getattr(c, ds.channels[idx]) - mean) / std for c in ds.cycles
            ])
            assert abs(float(values.mean())) < 1e-3, name
            assert abs(float(values.std(correction=0)) - 1.0) < 0.05, name

    def test_time_channel_is_not_standardised(self):
        """``time_s`` is a per-cycle clock, not a calibrated quantity."""
        ds = NASABatteryDataset(from_synthetic=True, window_len=64, stride=16)
        item = ds[0]
        raw_time = ds.cycles[0].time_s[:64]
        assert torch.allclose(item["x"][:, 2], raw_time, atol=1e-3)
        assert float(item["x"][:, 2].max()) > 10.0

    def test_channel_means_vector_lines_up_with_the_channels(self):
        """``channel_means`` is indexed by channel, not by the stats dict."""
        ds = NASABatteryDataset(from_synthetic=True, window_len=16, stride=8)
        means = ds[0]["channel_means"]
        assert means.shape == (len(ds.channels),)
        for i, name in enumerate(ds.channels):
            expected = ds.stats[name][0] if name in ds.stats else 0.0
            assert float(means[i]) == pytest.approx(expected, abs=1e-5)

    def test_normalisation_can_be_disabled(self):
        ds = NASABatteryDataset(from_synthetic=True, window_len=64, stride=8,
                                normalize=False)
        raw = torch.stack([ds[i]["x"] for i in range(8)])
        # Voltage is in volts (tens), not standardised units.
        assert 1.0 < float(raw[:, :, 0].min()) < 6.0

    def test_dataloader_yields_batches(self):
        ds = NASABatteryDataset(from_synthetic=True, window_len=32, stride=8)
        batch = next(iter(ds.dataloader(batch_size=4, shuffle=False)))
        assert batch["x"].shape[0] == 4

    def test_channel_names_are_documented(self):
        assert NASABatteryDataset.channels == (
            "voltage_v", "temperature_k", "time_s", "soc", "resistance_ohm"
        )

    def test_chemistry_inference_from_cell_ids(self):
        assert NASABatteryDataset(cell_ids=["B0005", "B0006"]).chemistry == "nmc"
        assert NASABatteryDataset(cell_ids=["B0033"]).chemistry == "lfp"


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


class TestSyntheticGenerator:
    def test_capacity_fades_over_cycles(self):
        cycles = synthetic_run_to_failure(cell_ids=["B0005"], n_cycles=60)
        first = [c for c in cycles if c.cycle_index == 1][0].capacity_ah
        last = [c for c in cycles if c.cycle_index == 60][0].capacity_ah
        assert last < first

    def test_is_deterministic(self):
        a = synthetic_run_to_failure(cell_ids=["B0005"], n_cycles=5, seed=11)
        b = synthetic_run_to_failure(cell_ids=["B0005"], n_cycles=5, seed=11)
        assert torch.allclose(a[0].voltage_v, b[0].voltage_v)

    def test_discharge_is_galvanostatic(self):
        cycles = synthetic_run_to_failure(cell_ids=["B0005"], n_cycles=3)
        for c in cycles:
            assert c.current_a < 0
            assert c.mode == "discharge"

    def test_lfp_generator_produces_flatter_voltage(self):
        nmc = synthetic_run_to_failure(cell_ids=["B0005"], n_cycles=1)
        lfp = synthetic_run_to_failure(cell_ids=["B0033"], n_cycles=1)
        nmc_span = float(nmc[0].voltage_v.max() - nmc[0].voltage_v.min())
        lfp_span = float(lfp[0].voltage_v.max() - lfp[0].voltage_v.min())
        assert lfp_span < nmc_span


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_fit_produces_usable_coefficients(self):
        ds = NASABatteryDataset(from_synthetic=True)
        fit = fit_internal_resistance(ds)
        assert fit.n_bins > 0
        assert fit.r0_ohm > 0
        assert math_is_finite(fit.ea_j_mol)
        assert fit.usable or fit.n_bins < 8  # usable requires enough bins

    def test_prediction_reproduces_the_fit(self):
        fit = InternalResistanceFit(r0_ohm=0.0135, ea_j_mol=23500.0,
                                    soc_quad=0.0, soc_lin=0.0, n_bins=20,
                                    rmse_log_r=0.01, synthetic=False)
        r25 = float(fit.predict(25.0, 0.5))
        r60 = float(fit.predict(60.0, 0.5))
        assert r25 == pytest.approx(0.0135, rel=0.02)
        assert r60 > r25

    def test_unused_fit_is_flagged(self):
        fit = InternalResistanceFit(r0_ohm=0.0135, n_bins=2)
        assert fit.usable is False

    def test_round_trip_through_json(self, tmp_path):
        fit = InternalResistanceFit(r0_ohm=0.014, ea_j_mol=22000.0, n_bins=30,
                                    rmse_log_r=0.02, synthetic=False)
        from backend.models.nasa_dataset import save_calibration

        path = tmp_path / "cal.json"
        save_calibration(fit, path)
        payload = json.loads(path.read_text())
        assert payload["r0_ohm"] == 0.014
        assert "cell_capacity" in payload

    def test_prior_blend_sits_between_empical_and_datasheet(self):
        from backend.config import get_chemistry

        spec = get_chemistry("nmc")
        fit = InternalResistanceFit(r0_ohm=0.030, r0_stderr_ohm=0.001,
                                    ea_j_mol=20000.0, n_bins=50, synthetic=False)
        blended = prior_mean_resistance(fit, spec, 25.0, 0.5)
        empirical = float(fit.predict(25.0, 0.5))
        datasheet = spec.r_int(25.0, 0.5)
        assert min(empirical, datasheet) <= blended <= max(empirical, datasheet)

    def test_prior_falls_back_to_datasheet_when_unusable(self):
        from backend.config import get_chemistry

        spec = get_chemistry("lfp")
        assert prior_mean_resistance(InternalResistanceFit(n_bins=0), spec) == \
            pytest.approx(spec.r_int(25.0, 0.5))


# ---------------------------------------------------------------------------
# Supervised telemetry projection
# ---------------------------------------------------------------------------


class TestSupervisedTelemetry:
    @pytest.mark.parametrize("chem", ["nmc", "lfp"])
    def test_projection_produces_located_labels(self, chem):
        ds = NASABatteryDataset(from_synthetic=True)
        tel = build_supervised_telemetry(ds, chemistry=chem, max_samples=512)
        assert len(tel) > 0
        assert tel.points_m.shape[1] == 3
        # Every point must lie inside the cold plate.
        assert torch.all(tel.points_m[:, 0] >= 0) and torch.all(tel.points_m[:, 0] <= 0.36)
        assert torch.all(tel.points_m[:, 1] >= 0) and torch.all(tel.points_m[:, 1] <= 0.23)
        assert torch.all(tel.points_m[:, 2] >= 0) and torch.all(tel.points_m[:, 2] <= 0.012)

    def test_soc_projects_onto_the_streamwise_axis(self):
        """Deeper discharge (lower SOC) must map to a smaller x."""
        ds = NASABatteryDataset(from_synthetic=True)
        tel = build_supervised_telemetry(ds, max_samples=400)
        corr = float(np.corrcoef(tel.points_m[:, 0].numpy(), tel.soc.numpy())[0, 1])
        assert corr < -0.5

    def test_resistance_channel_is_populated(self):
        ds = NASABatteryDataset(from_synthetic=True)
        tel = build_supervised_telemetry(ds, max_samples=256)
        assert torch.all(tel.resistance_ohm > 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def math_is_finite(x: float) -> bool:
    """Local alias to avoid importing math at module scope in tests."""
    import math

    return math.isfinite(x)


def _write_minimal_mat(path: Path) -> None:
    """Write a spec-compliant MAT v5 file containing the loader's expected keys.

    Layout per ``miMATRIX`` (14) subelement, per the MAT-file format:

    ``array flags`` (miUINT32, 8) | ``dimensions`` (miINT32) | ``array name``
    (miINT8) | real data (miDOUBLE)

    Each subelement is 4-byte type tag + 4-byte length + payload padded to an
    8-byte boundary.  Getting the subelement order wrong is not cosmetic -- a
    reader that assumes it will mis-slice every field after the first, which
    surfaces as nonsense values rather than a parse error.
    """
    def subelement(data_type: int, payload: bytes) -> bytes:
        """Encode one MAT subelement.

        The declared byte count **includes the 8-byte header**, per the MAT-file
        format spec. This is the single easiest thing to get wrong when writing
        a MAT file by hand: declaring the payload length instead makes every
        reader compute a field count of ``(declared - 8) / 4``, which for a
        two-dimension array comes out zero, so the reader concludes the array is
        empty and returns nothing -- with no error anywhere.
        """
        n = 8 + len(payload)
        pad = (8 - (n % 8)) % 8
        return struct.pack("<II", data_type, n) + payload + b"\x00" * pad

    def mi_matrix(name: str, values: np.ndarray) -> bytes:
        arr = np.asarray(values, dtype="<f8").ravel()
        # Array flags: 4-byte flag word, 1-byte mxClass, 3 unused.
        flags = subelement(6, struct.pack("<I", 0) + bytes([6, 0, 0, 0]))
        # Dimensions: 2-D (rows, cols) as miINT32.
        dims = subelement(5, np.array([arr.size, 1], dtype="<i4").tobytes())
        # Array name: miINT8 -- a 4-byte length followed by the characters.
        nb = name.encode("ascii")
        name_el = subelement(1, struct.pack("<i", len(nb)) + nb)
        # Real data: miDOUBLE.
        data = subelement(9, arr.tobytes())
        return subelement(14, flags + dims + name_el + data)

    keys = {
        "step": np.arange(1, 121, dtype=np.float64),
        "cycle": np.repeat(np.arange(1, 4), 40).astype(np.float64),
        "voltage": np.linspace(4.2, 3.0, 120),
        "current": np.full(120, -2.0),
        "temperature": np.full(120, 30.0),
        "capacity": np.full(120, 2.0),
        "Time": np.arange(1, 121, dtype=np.float64) * 60.0,
    }
    elements = b"".join(mi_matrix(k, v) for k, v in keys.items())

    # 128-byte header: 116 bytes of text, subsys offset, version, endian indicator.
    header = b"MATLAB 5.0 MAT-file, Platform: PCWIN64"
    header += b"\x00" * (116 - len(header))
    header += struct.pack("<H", 0x0100)      # version
    header += b"IM"                        # little-endian indicator
    header += b"\x00" * (128 - len(header))

    path.write_bytes(header + elements)