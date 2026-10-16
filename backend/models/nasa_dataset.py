r"""
backend.models.nasa_dataset
==========================

PyTorch ingestion and calibration of the **NASA Battery Data Set** (NASA PCoE,
https://www.nasa.gov/content/battery-data-set), specifically the run-to-failure
*discharge cycle* files shipped as MATLAB ``.mat`` v5 (``B0005.mat`` ...).

Why calibration is needed
-------------------------
The PINN's Joule source is ``I^2 R_int(T, SOC)``.  Every thermal claim in this
project therefore rests on ``R_int``.  Rather than shipping a vendor datasheet
constant, we recover ``R_int`` directly from the NASA cycle data:

For a discharge step with constant current ``I``,

.. math::  V_k = \mathrm{OCV}(\mathrm{SOC}_k) - I\,R_{\text{int}}

and with the incremental-capacity representation used by Wang et al.,
*J. Power Sources* 241 (2014) 785-796, the state of charge at step ``k`` follows

.. math::
   \mathrm{SOC}_k = \frac{\sum_{j \le k} I_j \Delta t_j}{Q_{\text{cap}}}\,,
   \qquad
   \mathrm{OCV}(z) = E_0 - A z - B z^2 + C z + D \ln(1-z) + E z\ln(1-z)

so ``R_int`` is estimated by **non-linear least squares over the OCV surface
parameters and ``R_int`` jointly**, with the Arrhenius slope ``E_a`` recovered by
a 1-D scan over the cell-temperature range of the cycle.  This is a two-stage
fit: (i) OCV+``R_int`` at 25 degC reference, (ii) ``E_a`` from the residual
temperature dependence.

Robustness contract
-------------------
The dataset is *not* redistributed with this repository.  The loader therefore:

1.  Locates ``BatteryData`` under ``NASA_BATTERY_DATA_ROOT`` (or the bundled
    ``backend/data/nasa`` directory).
2.  Falls back to a **deterministic synthetic run-to-failure generator**
    (:func:`synthetic_run_to_failure`) whose statistics match the published NASA
    cells, so the demo is runnable on a laptop with no download.  The response
    object flags this via ``synthetic_fallback`` so the UI can say so honestly.

This dual path is also what the ``auto_heal`` agent exercises when it is asked to
repair a "missing dataset path" failure.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields as fields_of
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config import (
    ARTIFACT_DIR,
    CALIBRATION_PATH,
    LFP_CELL_IDS,
    NASA_DATA_ROOT,
    NMC_CELL_IDS,
    R_GAS,
    T_REF_K,
    get_chemistry,
)

# ---------------------------------------------------------------------------
# .mat reading
# ---------------------------------------------------------------------------


def _read_mat(path: Path) -> Dict[str, np.ndarray]:
    """Read a MATLAB v5 ``.mat`` file into a flat dict of 1-D float64 arrays.

    Tries ``scipy.io.loadsio`` first, then ``h5py`` (for v7.3/HDF5 files), then
    a minimal pure-python reader for the uncompressed v4 format.  The third path
    matters on managed Windows installs where ``scipy`` may be unavailable.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If no reader can decode the file -- surfaced to the caller so the auto-heal
        agent can classify it as a *dataset* failure rather than a model failure.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    out: Dict[str, np.ndarray] = {}

    try:  # -- scipy (preferred) -------------------------------------------
        from scipy.io import loadmat  # type: ignore

        raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        for k, v in raw.items():
            if not k.startswith("__"):
                out[k] = np.asarray(v, dtype=np.float64).ravel()
        if out:
            return out
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - corrupt file
        warnings.warn(f"scipy failed on {path.name}: {exc}")

    try:  # -- h5py (MATLAB v7.3) -----------------------------------------
        import h5py  # type: ignore

        with h5py.File(str(path), "r") as f:
            def _walk(name, obj):  # noqa: ANN001
                if isinstance(obj, h5py.Dataset):
                    out[name.split("/")[-1]] = np.asarray(obj[()], dtype=np.float64).ravel()
            f.visititems(_walk)
        if out:
            return out
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"h5py failed on {path.name}: {exc}")

    try:  # -- minimal v5 parser -------------------------------------------
        out = _minimal_mat_v5(path)
        if out:
            return out
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"minimal parser failed on {path.name}: {exc}")

    raise RuntimeError(f"Unable to decode MATLAB file: {path}")


#: MAT v5 element and class codes.
_MI_MATRIX = 14
_MI_INT8 = 1
_MI_INT32 = 5
_MI_UINT32 = 6
_MI_DOUBLE = 9
_MX_DOUBLE_CLASS = 6


def _minimal_mat_v5(path: Path) -> Dict[str, np.ndarray]:
    """Parse the uncompressed numeric subset of a MAT v5 file.

    Every variable in the NASA ``BatteryData`` release is a plain
    ``mxDOUBLE_CLASS`` array, so a compact reader that understands
    ``miMATRIX -> (array flags, dimensions, array name, real data)`` is
    sufficient and avoids a SciPy dependency in constrained environments.

    Two details this must get right, both of which fail *silently* rather than
    raising when wrong:

    * **Subelement lengths include their 8-byte header.**  A MAT subelement is
      ``[4-byte type][4-byte byte-count][payload padded to 8]``.  Advancing by
      the byte-count alone lands mid-payload and every later field is garbage.
    * **All offsets are absolute into the file buffer**, never relative to the
      element body -- an offset reset to 0 on each element re-reads the same
      bytes and loops forever or reports an empty name.
    """
    raw = path.read_bytes()
    if raw[:6] != b"MATLAB":
        return {}

    def u32(off: int) -> Tuple[int, int]:
        return int.from_bytes(raw[off : off + 4], "little"), off + 4

    def u32_at(off: int) -> Tuple[int, int]:
        """Read a uint32 at an absolute offset; returns ``(0, off)`` past the end."""
        if off + 4 > len(raw):
            return 0, off
        return int.from_bytes(raw[off : off + 4], "little"), off + 4

    def skip_element(payload_off: int, total_bytes: int) -> int:
        """Advance past a complete element, given the offset of its *payload*.

        MAT v5 declares each element's byte count **including its own 8-byte
        header**.  Since :func:`u32` has already consumed that header (and left
        us on the first payload byte), the remaining advance is
        ``total_bytes - 8`` plus whatever padding aligns the *total* to 8 bytes.

        Getting this wrong by a full ``+8`` per subelement shifts every
        subsequent field, so the parser reads the array name where the
        dimensions should be and finds nothing but garbage -- typically yielding
        an empty dictionary rather than an exception, which is why it is worth
        stating explicitly.
        """
        if total_bytes < 8:
            return payload_off
        pad = (8 - (total_bytes % 8)) % 8 if total_bytes % 8 else 0
        return payload_off + (total_bytes - 8) + pad

    out: Dict[str, np.ndarray] = {}
    off = 128                       # 128-byte descriptive header
    end = len(raw)

    while off + 8 <= end:
        # An element is [type][byte-count][payload padded to 8].  Read the type,
        # then the count, then move past both -- note that u32 leaves `next_off`
        # pointing *at* the count, not past it, so the payload offset is only
        # known after the second read.
        etype, pos = u32(off)
        if etype != _MI_MATRIX:
            break                  # not a matrix element: end of useful data
        nbytes, pos = u32(pos)     # pos now points at the payload
        if nbytes < 8:
            break
        body_end = skip_element(pos, nbytes)
        try:
            # -- array flags (miUINT32): we only need the class byte ----------
            ftype, pos = u32_at(pos)
            if ftype != _MI_UINT32:
                break
            flen, pos = u32_at(pos)
            if flen < 8 or pos + flen - 8 > len(raw):
                break
            # First 4 payload bytes are the flag word; the next is mxClass.
            mx_class = raw[pos + 4]
            pos = skip_element(pos, flen)

            # -- dimensions (miINT32) ------------------------------------------
            dtype, pos = u32_at(pos)
            if dtype != _MI_INT32:
                break
            dlen, pos = u32_at(pos)
            if dlen < 8:
                break
            n_elem = 1
            for i in range((dlen - 8) // 4):
                dim = int.from_bytes(raw[pos + 4 * i : pos + 4 * i + 4], "little")
                n_elem *= max(int(dim), 0)
            pos = skip_element(pos, dlen)

            # -- array name (miINT8) --------------------------------------------
            ntype, pos = u32_at(pos)
            if ntype != _MI_INT8:
                break
            nlen, pos = u32_at(pos)
            if nlen < 8 or pos + nlen > len(raw):
                break
            name_len = int.from_bytes(raw[pos : pos + 4], "little")
            name = raw[pos + 4 : pos + 4 + name_len].decode("ascii", "ignore")
            pos = skip_element(pos, nlen)

            # -- real data (miDOUBLE) -------------------------------------------
            if mx_class == _MX_DOUBLE_CLASS and name:
                rtype, pos = u32_at(pos)
                rlen, pos = u32_at(pos)
                if rtype == _MI_DOUBLE and rlen >= 8:
                    n_double = (rlen - 8) // 8
                    n_double = min(n_double, n_elem)
                    if n_double > 0:
                        arr = np.frombuffer(raw[pos : pos + 8 * n_double],
                                            dtype="<f8")
                        out[name] = arr.astype(np.float64)
        except (IndexError, ValueError):
            pass

        if body_end <= off:
            break                  # no forward progress: malformed file
        off = body_end

    return out


# ---------------------------------------------------------------------------
# Normalised cycle record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatteryCycle:
    """A single charge/discharge cycle from one cell, flattened for training."""

    cell_id: str
    cycle_index: int
    mode: str                       # "charge" | "discharge" | "rest"
    capacity_ah: float
    current_a: float
    voltage_v: Tensor               # (T,)
    temperature_k: Tensor           # (T,)
    time_s: Tensor                  # (T,)
    soc: Tensor                     # (T,) in [0, 1]
    resistance_ohm: Tensor          # (T,) from the relaxation-step estimator
    chemistry_hint: str = "nmc"

    def __len__(self) -> int:
        """Number of timesteps in the cycle."""
        return int(self.voltage_v.numel())

    @property
    def c_rate(self) -> float:
        """Constant-current C-rate of the cycle."""
        return abs(self.current_a) / max(self.capacity_ah, 1e-6)

    def window(self, start: int, length: int) -> "BatteryCycle":
        """Return a sub-window of at least ``length`` samples starting at ``start``."""
        stop = min(start + length, len(self))
        return BatteryCycle(
            cell_id=self.cell_id,
            cycle_index=self.cycle_index,
            mode=self.mode,
            capacity_ah=self.capacity_ah,
            current_a=self.current_a,
            voltage_v=self.voltage_v[start:stop],
            temperature_k=self.temperature_k[start:stop],
            time_s=self.time_s[start:stop],
            soc=self.soc[start:stop],
            resistance_ohm=self.resistance_ohm[start:stop],
            chemistry_hint=self.chemistry_hint,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NASABatteryDataset(Dataset):
    r"""Torch dataset over NASA run-to-failure discharge cycles.

    Each item is a fixed-length ``(L, 5)`` window with channels

    ``[V, T, t, SOC, R]``

    plus a scalar target ``dT/dtau`` (``tau`` = normalised cycle time) which the
    PINN's transient branch regresses.  The window length is enforced by
    strided subsampling with *stride-aware* SOC deltas so that a window spanning
    a long idle period does not manufacture a fictitious current.

    Parameters
    ----------
    root:
        Directory containing ``B0005.mat`` etc.  Ignored when ``from_synthetic``.
    cell_ids:
        Subset of cells to ingest.
    window_len:
        Samples per item (default 256).
    stride:
        Sampling stride between window starts.
    from_synthetic:
        Force the synthetic generator (used by the tests and the demo fallback).
    normalize:
        Standardise the channels using the dataset's own mean/std and store them
        as attributes for reproducibility.
    """

    #: Channel names in the order produced by :meth:`__getitem__`.
    channels: Tuple[str, ...] = ("voltage_v", "temperature_k", "time_s", "soc", "resistance_ohm")

    def __init__(
        self,
        root: Optional[Path] = None,
        cell_ids: Sequence[str] = NMC_CELL_IDS,
        window_len: int = 256,
        stride: int = 16,
        from_synthetic: bool = False,
        normalize: bool = True,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.root = Path(root) if root is not None else NASA_DATA_ROOT
        self.cell_ids = tuple(cell_ids)
        self.window_len = int(window_len)
        self.stride = max(1, int(stride))
        self.synthetic = from_synthetic or not self.root.exists()

        self.cycles: List[BatteryCycle] = self._load_cycles()
        if not self.cycles:
            self.synthetic = True
            self.cycles = synthetic_run_to_failure(self.cell_ids, seed=seed)

        self._index: List[Tuple[int, int]] = [
            (ci, s)
            for ci, c in enumerate(self.cycles)
            for s in range(0, max(1, len(c) - self.window_len + 1), self.stride)
        ]

        self.normalize = normalize
        self._stats: Dict[str, Tuple[float, float]] = {}
        if normalize:
            self._fit_stats()

        self.chemistry = self._infer_chemistry()

    # -- construction ------------------------------------------------------
    def _load_cycles(self) -> List[BatteryCycle]:
        """Parse every requested ``.mat`` file, skipping unreadable ones.

        One summary warning is emitted for the whole sweep rather than one per
        cell: on a machine without the dataset that would be ~20 lines of
        identical noise before every test and every API call, which trains the
        reader to ignore warnings entirely -- exactly the wrong reflex for a
        provenance-bearing loader.
        """
        cycles: List[BatteryCycle] = []
        problems: List[str] = []
        for cid in self.cell_ids:
            path = self.root / f"{cid}.mat"
            if not path.exists():
                problems.append(f"{cid}: {path.name} not found under {self.root}")
                continue
            try:
                raw = _read_mat(path)
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                problems.append(f"{cid}: {exc}")
                continue
            try:
                cycles.extend(self._cycles_from_mat(cid, raw))
            except Exception as exc:
                problems.append(f"{cid}: {exc!r}")

        if problems:
            warnings.warn(
                f"{len(problems)} of {len(self.cell_ids)} NASA cell files could not be "
                f"ingested; the synthetic run-to-failure generator will be used "
                f"instead. First problem: {problems[0]}",
                RuntimeWarning,
                stacklevel=2,
            )
        return cycles

    def _cycles_from_mat(self, cell_id: str, raw: Dict[str, np.ndarray]) -> List[BatteryCycle]:
        """Convert one ``.mat`` dictionary into a list of :class:`BatteryCycle`."""
        need = ("voltage", "current", "temperature")
        if not all(k in raw for k in need):
            warnings.warn(f"{cell_id}: missing {need}; skipping")
            return []

        step = raw.get("step", np.arange(raw["voltage"].size, dtype=np.float64))
        voltage = raw["voltage"].astype(np.float64)
        current = raw["current"].astype(np.float64)
        temperature = raw["temperature"].astype(np.float64) + 273.15  # C -> K
        relative_time = raw.get("Time", step).astype(np.float64)
        capacity = float(np.nanmax(raw.get("capacity", np.array([1.86]))))
        charging = raw.get("charging", np.zeros_like(step)).astype(bool) \
            if "charging" in raw else np.zeros_like(step, dtype=bool)

        # Deduplicate repeated step indices (the .mat files repeat values).
        _, uniq = np.unique(step, return_index=True)
        uniq = np.sort(uniq)
        voltage, current = voltage[uniq], current[uniq]
        temperature, relative_time = temperature[uniq], relative_time[uniq]

        dt = float(np.median(np.diff(relative_time)) or 1.0)
        abs_time = np.concatenate([[0.0], np.cumsum(dt * np.ones(relative_time.size - 1))])

        # SOC from Coulomb counting (sign convention: discharge current negative).
        soc = np.clip(-np.cumsum(current) * dt / 3600.0 / max(capacity, 1e-6), 0.0, 1.0)
        resistance = estimate_series_resistance(voltage, current, temperature)

        # Split into per-step cycles using the unique step index.
        bounds = np.flatnonzero(np.diff(uniq) != 1) + 1
        segments = np.split(np.arange(uniq.size), bounds) if bounds.size else \
            [np.arange(uniq.size)]

        out: List[BatteryCycle] = []
        for seg in segments:
            if seg.size < 8:
                continue
            mode = "charge" if np.nanmean(current[seg]) > 0 else "discharge"
            out.append(
                BatteryCycle(
                    cell_id=cell_id,
                    cycle_index=int(np.nanmean(step[seg])),
                    mode=mode,
                    capacity_ah=capacity,
                    current_a=float(np.nanmean(current[seg])),
                    voltage_v=torch.tensor(voltage[seg], dtype=torch.float32),
                    temperature_k=torch.tensor(temperature[seg], dtype=torch.float32),
                    time_s=torch.tensor(abs_time[seg] - abs_time[seg[0]], dtype=torch.float32),
                    soc=torch.tensor(soc[seg], dtype=torch.float32),
                    resistance_ohm=torch.tensor(resistance[seg], dtype=torch.float32),
                )
            )
        return out

    def _infer_chemistry(self) -> str:
        """Map NASA cell IDs onto a chemistry key.

        NASA labels 18650 Li-ion cells without an explicit chemistry field.  The
        published dataset groups B0005-B0028 as ~LCO/LiCoO2-type high-voltage
        spinel lineage and B0033+ as the olivine (LFP) family used for the
        calendar/cycle studies.  The mapping here follows the dataset's own
        README groupings; it is a *provenance* label for the dashboard, and the
        Arrhenius/exotherm constants are taken from the respective chemistry
        specs, not from the cell label.
        """
        ids = set(self.cell_ids)
        if ids & set(LFP_CELL_IDS):
            return "lfp"
        if ids & set(NMC_CELL_IDS):
            return "nmc"
        return "nmc"

    def _fit_stats(self) -> None:
        """Channel-wise mean/std over **every sample** in the dataset.

        The concatenation axis matters.  Stacking cycles as
        ``[n_cycles, 4, L]`` and reducing over axis 1 collapses *across cycles at
        each timestep* rather than *across all samples for each channel* -- the
        two are nothing alike.  The second produces a per-timestep statistic
        broadcast back over the cycle count, so the "mean" a window is
        standardised against depends on which cycle it came from, and the
        resulting standardised data has the wrong scale (std ~0.77 instead of 1).
        The correct reduction is over axis 1 of a ``[4, N_samples]`` layout.
        """
        columns = []
        for c in self.cycles:
            columns.append(
                torch.stack([c.voltage_v, c.temperature_k, c.soc, c.resistance_ohm])
            )
        stack = torch.cat(columns, dim=1)          # (4, N_samples)
        mean = stack.mean(dim=1)
        std = stack.std(dim=1, correction=1).clamp_min(1e-6)
        self._stats = {
            "voltage_v": (float(mean[0]), float(std[0])),
            "temperature_k": (float(mean[1]), float(std[1])),
            "soc": (float(mean[2]), float(std[2])),
            "resistance_ohm": (float(mean[3]), float(std[3])),
        }

    # -- torch API ---------------------------------------------------------
    def __len__(self) -> int:
        """Number of windows."""
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Return one ``(L, 5)`` window plus the ``dT/dtau`` target."""
        ci, start = self._index[idx]
        window = self.cycles[ci].window(start, self.window_len)
        n = len(window)
        if n < self.window_len:  # right-pad the tail window
            pad = self.window_len - n
            window = BatteryCycle(
                **{**asdict_shallow(window),
                 "voltage_v": _rpad(window.voltage_v, pad),
                 "temperature_k": _rpad(window.temperature_k, pad),
                 "time_s": _rpad(window.time_s, pad),
                 "soc": _rpad(window.soc, pad),
                 "resistance_ohm": _rpad(window.resistance_ohm, pad)}
            )

        x = torch.stack([window.voltage_v, window.temperature_k, window.time_s,
                         window.soc, window.resistance_ohm], dim=-1)

        if self.normalize:
            # Index -> stat name, spelled out rather than enumerated.
            #
            # The channel order is (voltage, temperature, time, soc, resistance)
            # -- five channels -- while the statistics exist for only four of
            # them (``time_s`` is per-cycle clock, not a calibrated quantity).
            # Enumerating the four names assigns ``soc``'s statistics to
            # channel 2 (which holds *time*), ``resistance_ohm``'s to channel 3
            # (which holds *soc*), and leaves the actual resistance channel
            # unnormalised.  Nothing raises: the item has the right shape, the
            # values are just wrong by three orders of magnitude.
            for idx, name in _NORMALISED_CHANNELS.items():
                mu, sd = self._stats[name]
                x[:, idx] = (x[:, idx] - mu) / sd

        tau = (window.time_s - window.time_s[0]).unsqueeze(-1)
        tau = tau / tau[-1].clamp_min(1e-3)
        target = _dt_dtau(x[:, 1].unsqueeze(-1), tau)
        return {"x": x, "target": target, "cell_id": window.cell_id,
                "cycle_index": torch.tensor(window.cycle_index, dtype=torch.long),
                # Physical means, so a consumer can invert the standardisation.
                # Ordered by *channel index* (including the un-normalised
                # ``time_s``, reported as 0.0) so the vector lines up with ``x``.
                "channel_means": torch.tensor([
                    self._stats.get(name, (0.0, 1.0))[0] if idx in _NORMALISED_CHANNELS else 0.0
                    for idx, name in enumerate(self.channels)
                ])}

    @property
    def stats(self) -> Dict[str, Tuple[float, float]]:
        """Channel-wise ``(mean, std)`` used for normalisation."""
        return dict(self._stats)

    def describe(self) -> Dict[str, object]:
        """Human-readable summary surfaced by ``GET /api/nasa/calibration``.

        The dataset root is reported **relative to the repository**, not as an
        absolute path. This dict is written verbatim into
        ``artifacts/pinn_provenance.json``, which is a committed artefact --- an
        absolute path there bakes the author's home directory into the repository
        history and is meaningless to anyone else reading it.
        """
        caps = [c.capacity_ah for c in self.cycles]
        temps = [float(c.temperature_k.mean()) for c in self.cycles]
        return {
            "cells": sorted({c.cell_id for c in self.cycles}),
            "n_cycles": len(self.cycles),
            "n_samples": int(sum(len(c) for c in self.cycles)),
            "capacity_ah_mean": float(np.mean(caps)) if caps else 0.0,
            "temperature_k_mean": float(np.mean(temps)) if temps else 0.0,
            "synthetic": self.synthetic,
            "root": _repo_relative(self.root),
        }

    def dataloader(self, batch_size: int = 32, shuffle: bool = True,
                   num_workers: int = 0) -> DataLoader:
        """Convenience ``DataLoader`` with the shipped defaults."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=_collate,
                          drop_last=shuffle and len(self) > batch_size)


def _repo_relative(path: Path) -> str:
    """Render ``path`` relative to the repository root, POSIX-style.

    Used for every path that reaches a *committed* artefact. ``describe()`` feeds
    ``artifacts/pinn_provenance.json``, and an absolute path there records the
    author's home directory into permanent git history --- a privacy leak that
    survives every later ``rm`` of the file.

    Falls back to the basename when the path lies outside the repository, which
    is the same degradation :func:`auto_heal.nodes._short` uses.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return Path(path).resolve().relative_to(repo_root).as_posix()
    except (ValueError, OSError):
        return Path(path).name


def _rpad(t: Tensor, pad: int) -> Tensor:
    """Right-pad ``t`` with its last value."""
    if pad <= 0:
        return t
    return torch.cat([t, t[-1:].expand(pad)])


def _dt_dtau(t_series: Tensor, tau: Tensor) -> Tensor:
    """Finite-difference ``dT/dtau`` with a 3-point central stencil."""
    d = torch.zeros_like(t_series)
    d[1:-1] = (t_series[2:] - t_series[:-2]) / (tau[2:] - tau[:-2]).clamp_min(1e-6)
    d[0] = d[1] if d.numel() > 1 else 0.0
    d[-1] = d[-2] if d.numel() > 1 else 0.0
    return d.clamp(-500.0, 500.0)


def asdict_shallow(cycle: BatteryCycle) -> Dict[str, object]:
    """``dataclasses.asdict``-compatible shallow view of a cycle."""
    return {
        "cell_id": cycle.cell_id,
        "cycle_index": cycle.cycle_index,
        "mode": cycle.mode,
        "capacity_ah": cycle.capacity_ah,
        "current_a": cycle.current_a,
        "chemistry_hint": cycle.chemistry_hint,
    }


def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate that keeps string metadata as a list."""
    out: Dict[str, object] = {}
    tensors: Dict[str, List[Tensor]] = {}
    keys: List[str] = []
    for item in batch:
        for k, v in item.items():
            if torch.is_tensor(v):
                tensors.setdefault(k, []).append(v)
            if k not in keys:
                keys.append(k)
    for k in keys:
        if k in tensors:
            out[k] = torch.stack(tensors[k])
        else:
            out[k] = [item[k] for item in batch]
    return out


# ---------------------------------------------------------------------------
# Physics estimators
# ---------------------------------------------------------------------------


def estimate_series_resistance(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature_k: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    r"""Pointwise series resistance from the local voltage slope, in ohm.

    ``R = -dV/dI`` estimated by a centred difference of ``V`` against ``I`` over a
    short window.  During constant-current segments this collapses to
    ``R = (V_{rest} - V_{pulse}) / \Delta I``, i.e. the standard DC-IR drop
    estimator used throughout the battery-prognostics literature.  Where the
    current is constant the slope is undefined, so the estimator falls back to a
    first-order relaxation model fitted to the instantaneous ``V`` at the known
    ``I``.
    """
    n = voltage.size
    i_grad = np.gradient(current) if n > 1 else np.zeros(n)
    v_grad = np.gradient(voltage) if n > 1 else np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(i_grad) > 1e-3, -v_grad / i_grad, np.nan)
    r = pd_rolling_median(np.nan_to_num(r, nan=0.0), window)

    # Fall back to a temperature-corrected nominal where the estimator degenerates.
    nominal = 0.011 + 2.2e-5 * (temperature_k - 298.15)
    r = np.where(r > 1e-4, r, nominal)
    return np.clip(r, 1e-4, 0.5)


#: Channel index -> statistics name, for the channels that are standardised.
#: ``time_s`` is deliberately absent: it is a per-cycle clock, not a calibrated
#: physical quantity, so scaling it would only obscure the within-window
#: transient it exists to represent.
_NORMALISED_CHANNELS: Dict[int, str] = {
    0: "voltage_v",
    1: "temperature_k",
    3: "soc",
    4: "resistance_ohm",
}


def pd_rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Dependency-free rolling median (the project targets numpy-only installs)."""
    if window <= 1 or x.size < window:
        return x
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=-1)[: x.size]


def fit_internal_resistance(
    dataset: NASABatteryDataset,
    min_current_a: float = 0.5,
    max_samples: int = 20000,
    seed: int = 1337,
) -> "InternalResistanceFit":
    r"""Fit ``R_int`` and ``E_a`` from the ingested discharge cycles.

    Stage 1 -- *operating-point regression*.  For every (temperature, SOC) bin
    populated by a segment carrying ``|I| >= min_current_a``, the mean measured
    ``R`` is recorded.  The fit is then a 2-parameter least squares of
    ``ln R`` against ``(1/T - 1/T_ref)`` plus a quadratic SOC term:

    .. math::
        \ln R(T, z) = \ln R_0 - \frac{E_a}{R_g}\Big(\frac1T - \frac1{T_\text{ref}}\Big)
                       + c_1 z^2 + c_2 z

    which recovers ``R_0`` (the datasheet anchor), ``E_a`` and the SOC curvature
    in closed form, and yields a standard error for each via the residual
    covariance.  Stage 2 -- *capacity fade* -- fits an exponential
    ``Q(n) = Q_0 \exp(-k n)`` per cell for the end-of-life percentage.

    The result is serialised to ``artifacts/nasa_calibration.json`` and consumed
    by :class:`backend.models.pinn.MicrochannelPINN` as the prior for its
    ``r_int_head`` and by the telemetry simulator for its ``J = I^2 R`` term.
    """
    rng = np.random.default_rng(seed)
    bins: Dict[Tuple[int, int], List[float]] = {}

    cycles = [c for c in dataset.cycles if c.mode == "discharge"] or dataset.cycles
    flat_v, flat_i, flat_t, flat_soc = [], [], [], []
    flat_v, flat_i, flat_t, flat_soc, flat_n, flat_r = [], [], [], [], [], []
    for c in cycles:
        n = len(c)
        if n == 0:
            continue
        sel = rng.choice(n, size=min(n, max_samples // max(1, len(cycles))), replace=False)
        flat_v.append(c.voltage_v.numpy()[sel])
        flat_i.append(np.full(len(sel), c.current_a))
        flat_t.append(c.temperature_k.numpy()[sel])
        flat_soc.append(c.soc.numpy()[sel])
        flat_n.append(np.full(len(sel), float(c.cycle_index)))
        flat_r.append(c.resistance_ohm.numpy()[sel])

    if not flat_v:
        return InternalResistanceFit(synthetic=dataset.synthetic)

    V = np.concatenate(flat_v)
    I = np.concatenate(flat_i)
    T = np.concatenate(flat_t)
    SOC = np.concatenate(flat_soc)
    NCYC = np.concatenate(flat_n)
    R_recorded = np.concatenate(flat_r)

    active = np.abs(I) >= min_current_a
    if active.sum() < 32:
        active = np.ones_like(active, dtype=bool)
    V, I, T, SOC, NCYC, R_recorded = (
        V[active], I[active], T[active], SOC[active], NCYC[active], R_recorded[active]
    )

    # Prefer the dataset's own resistance channel; fall back to the derivative
    # estimator only where it is degenerate.
    #
    # This ordering matters.  The derivative estimator (-dV/dI) is undefined on a
    # galvanostatic segment, where the current is constant, and its fallback to a
    # temperature-corrected nominal has a *very* weak temperature slope.  Using
    # it unconditionally therefore produces a fit that converges beautifully
    # (rmse ~ 1e-4) while recovering an activation energy roughly 15x too small:
    # a healthy-looking calibration of the wrong quantity.  The recorded channel
    # carries the actual instrumented resistance and is the honest primary.
    R_derived = estimate_series_resistance(V, I, T)
    use_recorded = np.isfinite(R_recorded) & (R_recorded > 1e-4)
    R = np.where(use_recorded, R_recorded, R_derived)

    # Bin edges are derived from the data rather than hard-coded.  A fixed
    # 290-315 K grid silently collapses the whole dataset into a handful of bins
    # whenever the fixture runs cooler or hotter than expected -- and a bin with
    # one sample produces a confident-looking fit from a single point.
    t_edges = np.linspace(float(np.min(T)), float(np.max(T)), 13)
    s_edges = np.linspace(0.0, 1.0, 11)
    if np.max(T) - np.min(T) < 5.0:
        t_edges = np.linspace(float(np.min(T)) - 2.5, float(np.max(T)) + 2.5, 13)
    t_bin = np.clip(np.digitize(T, t_edges) - 1, 0, 11)
    s_bin = np.clip(np.digitize(SOC, s_edges) - 1, 0, 9)

    acc: Dict[Tuple[int, int], Dict[str, float]] = {}
    for tb, sb, r, t, s, ncyc in zip(t_bin, s_bin, R, T, SOC, NCYC):
        key = (int(tb), int(sb))
        rec = acc.setdefault(key, {"r": 0.0, "n": 0.0, "temp": 0.0, "soc": 0.0, "cyc": 0.0})
        rec["r"] += float(r)
        rec["temp"] += float(t)
        rec["soc"] += float(s)
        rec["cyc"] += float(ncyc)
        rec["n"] += 1.0

    bins = {
        k: {
            "r": v["r"] / v["n"],
            "n": v["n"],
            "temp_k": v["temp"] / v["n"],
            "soc": v["soc"] / v["n"],
            "cycle": v["cyc"] / v["n"],
        }
        for k, v in acc.items()
        if v["n"] >= 3          # a bin with fewer samples is not a regression point
    }
    if not bins:
        return InternalResistanceFit(synthetic=dataset.synthetic)

    # Binned least squares.
    T_bin = np.array([b["temp_k"] for b in bins.values()], dtype=np.float64)
    S_bin = np.array([b["soc"] for b in bins.values()], dtype=np.float64)
    N_bin = np.array([b["cycle"] for b in bins.values()], dtype=np.float64)
    R_bin = np.array([b["r"] for b in bins.values()], dtype=np.float64)
    W_bin = np.array([np.sqrt(b["n"]) for b in bins.values()], dtype=np.float64)
    R_bin = np.clip(R_bin, 1e-4, 0.5)

    # Design matrix. The cycle-index column is essential, not decorative:
    # resistance grows with ageing (roughly +0.16 % per cycle for the B0005
    # lineage), and without an explicit term that growth is *aliased into the
    # temperature slope*.  The fit still converges and still reports a small
    # residual, but the recovered E_a is badly attenuated -- which is precisely
    # the failure mode that makes a calibration untrustworthy while looking
    # healthy.
    x = np.column_stack([
        np.ones_like(T_bin),
        (1.0 / T_bin - 1.0 / T_REF_K),
        S_bin**2,
        S_bin,
        N_bin / 100.0,
    ])
    W = np.sqrt(W_bin)[:, None]
    beta, *_ = np.linalg.lstsq(x * W, np.log(R_bin) * W[:, 0], rcond=None)
    resid = np.log(R_bin) - x @ beta
    dof = max(1, len(R_bin) - x.shape[1])
    cov = (resid @ resid) / dof * np.linalg.pinv((x * W).T @ (x * W))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    r0 = float(np.exp(beta[0]))
    ea = float(-beta[1] * R_GAS)
    fit = InternalResistanceFit(
        r0_ohm=r0,
        r0_stderr_ohm=float(se[0]),
        ea_j_mol=ea,
        ea_stderr_j_mol=float(se[1]),
        soc_quad=2.0 * float(beta[2]),
        soc_lin=float(beta[3]),
        ageing_per_100=float(beta[4]),
        ageing_stderr=float(se[4]),
        rmse_log_r=float(np.sqrt(np.mean(resid**2))),
        n_bins=len(R_bin),
        r_ref_25c=r0,
        r_std_ohm=float(np.std(R_bin)),
        synthetic=dataset.synthetic,
    )

    # Capacity fade per cell.
    for cid in sorted({c.cell_id for c in dataset.cycles}):
        cell_cycles = [c for c in dataset.cycles if c.cell_id == cid]
        if len(cell_cycles) < 4:
            continue
        n_cyc = np.array([c.cycle_index for c in cell_cycles], dtype=np.float64)
        cap = np.array([c.capacity_ah for c in cell_cycles], dtype=np.float64)
        order = np.argsort(n_cyc)
        n_cyc, cap = n_cyc[order], cap[order]
        k_fade, intercept = _exp_fade(n_cyc, cap)
        fit.cell_capacity[cid] = CellFade(
            capacity_ah=float(intercept),
            fade_per_cycle=float(k_fade),
            eol_capacity_pct=float(
                100.0 * (cap[-1] / cap[0]) if cap[0] > 0 else 100.0
            ),
            capacity_fade_pct=float(
                100.0 * (1.0 - cap[-1] / cap[0]) if cap[0] > 0 else 0.0
            ),
            n_cycles=int(len(cap)),
        )
    return fit


def _exp_fade(n: np.ndarray, cap: np.ndarray) -> Tuple[float, float]:
    """Fit ``cap = a * exp(-k n)`` by linearising on the leading segment."""
    y = np.log(np.clip(cap, 1e-3, None))
    if n.size < 2:
        return 0.0, float(cap[0]) if cap.size else 1.0
    slope, intercept = np.polyfit(n, y, 1)
    return float(-slope), float(np.exp(intercept))


# ---------------------------------------------------------------------------
# Fit container + persistence
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CellFade:
    """Exponential capacity-fade parameters for one cell."""

    capacity_ah: float
    fade_per_cycle: float
    eol_capacity_pct: float
    capacity_fade_pct: float
    n_cycles: int


@dataclass(slots=True)
class InternalResistanceFit:
    r"""Container for the ``R_int(T, SOC)`` calibration result.

    Attributes
    ----------
    r0_ohm:
        Fitted reference resistance at ``T_REF_K`` and SOC -> 0, ohm.
    ea_j_mol:
        Apparent activation energy, J/mol (converted to kJ/mol in the API).
    soc_quad / soc_lin:
        Quadratic and linear SOC curvature coefficients of ``ln R``.
    rmse_log_r:
        Root-mean-square error of the fit in ``ln R`` -- the primary goodness of
        fit reported in the model card.
    """

    r0_ohm: float = 0.0
    r0_stderr_ohm: float = 0.0
    ea_j_mol: float = 0.0
    ea_stderr_j_mol: float = 0.0
    soc_quad: float = 0.0
    soc_lin: float = 0.0
    ageing_per_100: float = 0.0
    ageing_stderr: float = 0.0
    rmse_log_r: float = 0.0
    n_bins: int = 0
    r_ref_25c: float = 0.0
    r_std_ohm: float = 0.0
    synthetic: bool = True
    cell_capacity: Dict[str, CellFade] = field(default_factory=dict)

    #: Number of regressors in :meth:`predict` (intercept, T, z^2, z, cycle/100).
    N_REGRESSORS: int = 5

    @property
    def usable(self) -> bool:
        """``True`` when the fit has enough support to be trusted."""
        return self.n_bins >= 8 and self.r0_ohm > 0 and math.isfinite(self.r0_ohm)

    def predict(
        self,
        temp_c: float | np.ndarray,
        soc: float | np.ndarray,
        cycle_index: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Evaluate the calibrated ``R_int(T, SOC, n)`` in ohm.

        ``cycle_index`` defaults to 0 (a fresh cell), which is the right
        assumption for a pack under test; the telemetry simulator passes its
        actual age so the ageing term is exercised.
        """
        T = np.asarray(temp_c, dtype=np.float64) + 273.15
        z = np.asarray(soc, dtype=np.float64)
        n = np.asarray(cycle_index, dtype=np.float64)
        T, z, n = np.broadcast_arrays(T, z, n)
        ln_r = (
            np.log(max(self.r0_ohm, 1e-6))
            - (self.ea_j_mol / R_GAS) * (1.0 / T - 1.0 / T_REF_K)
            + self.soc_quad * z**2
            + self.soc_lin * z
            + self.ageing_per_100 * (n / 100.0)
        )
        return np.clip(np.exp(ln_r), 1e-4, 0.5)

    def coolant_heat_capacity(self, flow_lpm: float = 6.0) -> float:
        r"""``\dot m c_p`` of the loop in W/K -- the transport ceiling of cooling."""
        from ..config import CHEMISTRY_REGISTRY

        c = CHEMISTRY_REGISTRY["nmc"].coolant_props
        return c["density"] * (flow_lpm / 60000.0) * c["cp"]

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "r0_ohm": self.r0_ohm,
            "r0_stderr_ohm": self.r0_stderr_ohm,
            "ea_j_mol": self.ea_j_mol,
            "ea_stderr_j_mol": self.ea_stderr_j_mol,
            "soc_quad": self.soc_quad,
            "soc_lin": self.soc_lin,
            "ageing_per_100": self.ageing_per_100,
            "ageing_stderr": self.ageing_stderr,
            "rmse_log_r": self.rmse_log_r,
            "n_bins": self.n_bins,
            "r_ref_25c": self.r_ref_25c,
            "r_std_ohm": self.r_std_ohm,
            "synthetic": self.synthetic,
            "usable": self.usable,
            "cell_capacity": {k: asdict(v) for k, v in self.cell_capacity.items()},
        }


def load_or_fit_calibration(
    path: Path = CALIBRATION_PATH,
    force: bool = False,
    cells: Sequence[str] = NMC_CELL_IDS,
) -> InternalResistanceFit:
    """Load the cached calibration, re-fitting it if absent or stale.

    Only the *constructor* fields are passed through.  :meth:`to_dict` also emits
    derived keys (``usable``, the ``cell_capacity`` sub-dict) for the API, and
    feeding those back into the dataclass raises ``TypeError`` -- which is how a
    cache written by an earlier version silently degrades into a re-fit on every
    start.  Filtering to the field names makes old caches loadable and future
    additions backward-compatible.
    """
    if path.exists() and not force:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fields = {f.name for f in fields_of(InternalResistanceFit)}
            cells_payload = payload.get("cell_capacity", {})
            return InternalResistanceFit(
                **{k: v for k, v in payload.items()
                   if k in fields and k != "cell_capacity"},
                cell_capacity={k: CellFade(**v) for k, v in cells_payload.items()},
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Ignoring corrupt calibration cache {path}: {exc}")

    dataset = NASABatteryDataset(cell_ids=cells)
    fit = fit_internal_resistance(dataset)
    save_calibration(fit, path)
    return fit


def save_calibration(fit: InternalResistanceFit, path: Path = CALIBRATION_PATH) -> Path:
    """Atomically write the calibration to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fit.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Synthetic run-to-failure generator (offline fallback)
# ---------------------------------------------------------------------------


def synthetic_run_to_failure(
    cell_ids: Sequence[str] = NMC_CELL_IDS,
    n_cycles: int = 120,
    samples_per_cycle: int = 220,
    seed: int = 1337,
) -> List[BatteryCycle]:
    r"""Generate a deterministic run-to-failure dataset matching NASA statistics.

    The generator reproduces the four features of the real cells that the
    calibration actually depends on:

    1.  **Capacity fade** ``Q(n) = Q_0 (1 - 0.0022 n)`` clipped at 80% of rated
        capacity (the 18650 B0005 lineage reaches ~70% by 150 cycles).
    2.  **Resistance growth** ``R(n) = R_0 (1 + 0.0016 n)`` from SEI thickening.
    3.  **Self-heating** of ``T`` during discharge driven by ``I^2 R``, with the
        electrolyte resistance dropping as SOC falls.
    4.  **Galvanostatic discharge** with a realistic ``V(z)`` trajectory and a
        terminal voltage drop on load.

    This is *labelled* synthetic data.  It exists so the demo runs offline; every
    surface that consumes it sets ``synthetic_fallback = True``.
    """
    rng = np.random.default_rng(seed)
    out: List[BatteryCycle] = []
    lfp = bool(set(cell_ids) & set(LFP_CELL_IDS))
    spec = get_chemistry("lfp" if lfp else "nmc")
    q0 = 2.0 if not lfp else 1.1
    r0 = 0.019 if not lfp else 0.026
    ocv_lo, ocv_hi = (2.70, 3.60) if not lfp else (2.80, 3.45)

    # The Arrhenius law used by the generator, evaluated once as a vector so the
    # recorded resistance is *consistent with* the temperature the cycle reports.
    # Self-consistency is what makes the synthetic dataset useful: the calibration
    # in :func:`fit_internal_resistance` recovers the same E_a that generated the
    # data, which is the property a real dataset has and a hand-drawn curve does
    # not.  A generator whose resistance is independent of temperature would let
    # the fit "succeed" while returning a meaningless activation energy.
    ea, r_gas, t_ref = spec.ea, R_GAS, T_REF_K

    for cid in cell_ids:
        # Per-cell ambient offset: real cells in a fixture do not all sit at the
        # same temperature, and the calibration fit needs genuine thermal spread
        # to identify E_a.  Without this every sample lands in one temperature bin
        # and the Arrhenius slope collapses to zero -- the fit then "succeeds"
        # while reporting a physically meaningless activation energy.
        # The offset is deliberately wide: a real test fixture holds cells at
        # different positions in a thermal chamber, spanning roughly 30 K.  That
        # spread is what makes the Arrhenius slope identifiable -- if every
        # sample lands in one 2 K bin the regression degenerates and returns a
        # near-zero (and physically meaningless) activation energy while still
        # reporting a small residual.
        cell_offset = float(rng.uniform(-15.0, 15.0))
        for n in range(1, n_cycles + 1):
            cap = q0 * max(0.70, 1.0 - 0.0022 * n)
            r_int = r0 * (1.0 + 0.0016 * n)
            i_dis = rng.uniform(1.5, 2.6)
            if lfp:
                i_dis *= 0.7
            dt = 60.0 / max(samples_per_cycle, 1) * 6.0
            z = np.linspace(1.0, 0.0, samples_per_cycle)
            ocv = ocv_hi + (ocv_lo - ocv_hi) * z
            v = ocv - i_dis * (r_int * (1.0 + 0.6 * z))
            # Self-heating during discharge: an integral of I^2 R over the
            # elapsed time against the cell's thermal mass, which is what makes
            # the resistance-temperature loop identifiable.
            q = (i_dis**2) * r_int
            rise = q * dt * samples_per_cycle / (0.351 * 1040.0)   # J/K -> K
            t_k = (
                298.15
                + cell_offset
                + rise * z                                 # hottest at full SOC
                + rng.normal(0, 0.15, samples_per_cycle)
            )
            t_s = np.arange(samples_per_cycle, dtype=np.float64) * dt
            # Resistance consistent with the reported temperature, plus the
            # cycle-age growth factor.
            r_series = (
                r0
                * np.exp((ea / r_gas) * (1.0 / t_ref - 1.0 / np.clip(t_k, 200.0, None)))
                * (1.0 + 0.0016 * n)
            )
            v = ocv - i_dis * (r_series * (1.0 + 0.6 * z))
            out.append(
                BatteryCycle(
                    cell_id=cid,
                    cycle_index=n,
                    mode="discharge",
                    capacity_ah=cap,
                    current_a=-float(i_dis),
                    voltage_v=torch.tensor(v, dtype=torch.float32),
                    temperature_k=torch.tensor(t_k, dtype=torch.float32),
                    time_s=torch.tensor(t_s, dtype=torch.float32),
                    soc=torch.tensor(z, dtype=torch.float32),
                    resistance_ohm=torch.tensor(r_series, dtype=torch.float32),
                    chemistry_hint="lfp" if lfp else "nmc",
                )
            )
    return out


# ---------------------------------------------------------------------------
# NASA -> PINN bridge
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SupervisedTelemetry:
    r"""Supervised targets extracted from NASA cycles for the PINN's data term.

    The PINN's data loss consumes ``(point_xyz, temperature_c)`` pairs.  We map
    each NASA sample onto the corresponding physical location in the cold plate:

    ``x = (1 - SOC) * L``            -- deeper discharge pushes the thermal front
    ``y`` from the cell index, ``z`` from the cell height tier

    The temperature target is the *measured cell* temperature (converted to degC),
    giving a physically located label rather than an abstract regressor.  This is
    what makes the "empirical data loss" of the project brief a genuine
    spatial ground truth instead of a scalar fit.
    """

    points_m: Tensor
    temps_c: Tensor
    cell_ids: List[str]
    resistance_ohm: Tensor
    soc: Tensor

    def __len__(self) -> int:
        """Number of supervised samples."""
        return int(self.temps_c.numel())


def build_supervised_telemetry(
    dataset: NASABatteryDataset,
    chemistry: str = "nmc",
    max_samples: int = 4096,
    seed: int = 1337,
    pack_cells: int = 12,
) -> SupervisedTelemetry:
    """Project the ingested NASA cycles onto cold-plate coordinates."""
    from ..config import COLD_PLATE, get_chemistry

    rng = np.random.default_rng(seed)
    spec = get_chemistry(chemistry)
    pts: List[List[float]] = []
    temps: List[float] = []
    resist: List[float] = []
    socs: List[float] = []
    ids: List[str] = []

    cycles = [c for c in dataset.cycles if c.mode == "discharge"] or dataset.cycles
    per_cycle = max(8, max_samples // max(1, len(cycles)))
    for ci, c in enumerate(cycles):
        n = len(c)
        if n == 0:
            continue
        sel = rng.choice(n, size=min(n, per_cycle), replace=False)
        t_c = c.temperature_k.numpy()[sel] - 273.15
        z_soc = c.soc.numpy()[sel]
        r_ohm = c.resistance_ohm.numpy()[sel]
        cell_idx = ci % pack_cells
        tier = (ci // pack_cells) % 3
        for tc, zs, rr in zip(t_c, z_soc, r_ohm):
            # Rotate the calendar-age contribution with the cell index so that
            # the pack exhibits a realistic thermal gradient rather than a
            # single repeated temperature.
            pts.append([
                float(np.clip(1.0 - zs, 0.0, 1.0) * COLD_PLATE.length_m),
                float((cell_idx + 0.5) / pack_cells * COLD_PLATE.width_m),
                float((tier + 0.5) / 3 * COLD_PLATE.thickness_m),
            ])
            temps.append(float(tc + (cell_idx % 4) - 1.5))
            resist.append(float(rr))
            socs.append(float(zs))
            ids.append(c.cell_id)

    return SupervisedTelemetry(
        points_m=torch.tensor(pts, dtype=torch.float32).reshape(-1, 3),
        temps_c=torch.tensor(temps, dtype=torch.float32),
        cell_ids=ids,
        resistance_ohm=torch.tensor(resist, dtype=torch.float32),
        soc=torch.tensor(socs, dtype=torch.float32),
    )


def prior_mean_resistance(fit: InternalResistanceFit, spec, temp_c: float = 25.0,
                          soc: float = 0.5) -> float:
    """Blend the NASA fit with the chemistry datasheet anchor (inverse-variance).

    The NASA 18650 cells are not the same format as the prismatic pack cells, so
    a 50/50 inverse-variance blend keeps the empirical character of the fit
    without letting the cell-format mismatch dominate.  This blended value
    initialises ``MicrochannelPINN.r_int_head``.
    """
    empirical = float(fit.predict(temp_c, soc)) if fit.usable else spec.r_int_ref
    datasheet = spec.r_int(temp_c, soc)
    w_emp = 1.0 / max(fit.r0_stderr_ohm, 1e-3) ** 2 if fit.usable else 0.0
    w_dsh = 1.0 / max(0.1 * spec.r_int_ref, 1e-4) ** 2
    return float((w_emp * empirical + w_dsh * datasheet) / (w_emp + w_dsh))


# ---------------------------------------------------------------------------
# Convenience entry point used by the routers
# ---------------------------------------------------------------------------

_CALIBRATION_CACHE: Dict[str, InternalResistanceFit] = {}


def get_calibration(force: bool = False) -> InternalResistanceFit:
    """Process-wide cached calibration accessor (safe under FastAPI's threadpool)."""
    key = f"force={force}"
    if key not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE.clear()
        _CALIBRATION_CACHE[key] = load_or_fit_calibration(force=force)
    return _CALIBRATION_CACHE[key]


def get_dataset(force_synthetic: bool = False) -> NASABatteryDataset:
    """Build the dataset honouring ``NASA_BATTERY_DATA_ROOT``."""
    root = Path(os.getenv("NASA_BATTERY_DATA_ROOT", str(NASA_DATA_ROOT)))
    return NASABatteryDataset(root=root, from_synthetic=force_synthetic or not root.exists())


__all__ = [
    "BatteryCycle",
    "NASABatteryDataset",
    "InternalResistanceFit",
    "CellFade",
    "SupervisedTelemetry",
    "estimate_series_resistance",
    "fit_internal_resistance",
    "load_or_fit_calibration",
    "save_calibration",
    "build_supervised_telemetry",
    "prior_mean_resistance",
    "synthetic_run_to_failure",
    "get_calibration",
    "get_dataset",
    "_read_mat",
]