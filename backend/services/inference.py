r"""
backend.services.inference
=========================

Runtime wrapper around :class:`backend.models.pinn.MicrochannelPINN`.

Responsibilities
----------------
* **Lazy singleton** construction of the network and the calibration so that the
  FastAPI app starts in well under a second even with a cold ``torch`` import.
* **GPU/CPU device selection** with a deterministic fallback and an explicit
  dtype policy (``float32`` for inference -- the second-derivative chain in the
  residual would overflow in ``float16``).
* **Latency accounting** against the 800 ms budget declared in the model card.
* **Field evaluation** on a tensor-product grid with chunking so a 64x48x32 grid
  never allocates more than a few tens of MB at once.
* **Residual reporting** so the dashboard can prove the returned field still
  satisfies the physics, not just the telemetry.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..config import (
    CHECKPOINT_PATH,
    COLD_PLATE,
    INFERENCE_LATENCY_BUDGET_MS,
    PINN_CONFIG,
    ChemistrySpec,
    get_chemistry,
)
from ..models.physics import (
    CHANNEL_DOMAIN,
    PACK_DOMAIN,
    StreamFunction,
    heat_transfer_coefficient,
    lumped_pack_rhs,
    sample_face,
    thermal_runaway_margin,
)
from ..models.pinn import (
    FieldOutput,
    MicrochannelPINN,
    flow_to_reynolds,
    load_checkpoint,
)
from ..models.nasa_dataset import InternalResistanceFit, get_calibration

# ---------------------------------------------------------------------------
# Device / dtype policy
# ---------------------------------------------------------------------------


def select_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device.

    ``"auto"`` uses CUDA when available, then Apple MPS, then CPU.  Note that
    ``autograd.grad(create_graph=True)`` is **not** implemented for MPS, so the
    residual path forces CPU there; the temperature path can still use MPS.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and \
            torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#: Dtype used for inference.  float32 is the documented deployment dtype.
INFERENCE_DTYPE = torch.float32
#: Dtype used for residual evaluation -- float64 for numerical headroom.
RESIDUAL_DTYPE = torch.float64


@dataclass(slots=True)
class LatencyTracker:
    """Rolling latency statistics (EMA + p95 ring buffer)."""

    window: int = 200
    samples: List[float] = field(default_factory=list)
    ema: float = 0.0

    def record(self, ms: float) -> None:
        """Record one latency sample in milliseconds."""
        self.samples.append(ms)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        self.ema = 0.9 * self.ema + 0.1 * ms if self.ema else ms

    def p95(self) -> float:
        """95th-percentile latency over the rolling window, ms."""
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, 95))

    @property
    def n(self) -> int:
        """Number of retained samples."""
        return len(self.samples)


@dataclass(slots=True)
class FieldResult:
    """Result of one :meth:`InferenceEngine.evaluate_field` call."""

    chemistry: str
    axes: Dict[str, np.ndarray]
    temperature_c: np.ndarray          # (nx, ny, nz)
    max_temp_c: float
    min_temp_c: float
    mean_temp_c: float
    cell_outlet_c: float
    coolant_outlet_c: float
    heat_rejected_w: float
    thermal_runaway_margin_c: float
    inference_latency_ms: float
    within_latency_budget: bool
    residuals: Dict[str, float]
    adaptive_weights: Dict[str, float]
    coefficients: Dict[str, float]
    reynolds: float
    flow_lpm: float


class InferenceEngine:
    r"""Singleton facade over the trained PINN + physics post-processing.

    Notes
    -----
    The engine deliberately keeps the *physics* in charge of the derived
    quantities (outlet temperature, heat rejection, coolant rise) rather than
    reading them off separate network heads.  This has two consequences that
    matter for the model card:

    * every reported quantity is auditable against :mod:`backend.models.physics`;
    * the network cannot hallucinate a coolant-rise number that contradicts its
      own temperature field, because the rise is *computed* from the field and
      the mass flow.
    """

    _instance: Optional["InferenceEngine"] = None

    def __init__(
        self,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None,
        calibration: Optional[InternalResistanceFit] = None,
    ) -> None:
        self.device = select_device(device or "auto")
        self.checkpoint_path = checkpoint or str(CHECKPOINT_PATH)
        self.model: MicrochannelPINN = load_checkpoint(
            CHECKPOINT_PATH if checkpoint is None else __import__("pathlib").Path(checkpoint),
            device=self.device,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.calibration = calibration or get_calibration()
        self.latency = LatencyTracker()

        # A residual-only CPU copy: the temperature path runs on the fast device,
        # the diagnostic residuals run in float64 on CPU where create_graph is
        # reliably supported across all backends.
        self._residual_device = torch.device("cpu")

    # -- singleton ---------------------------------------------------------
    @classmethod
    def instance(cls, device: Optional[str] = None) -> "InferenceEngine":
        """Return (and memoise) the process-wide engine."""
        if cls._instance is None or device is not None:
            cls._instance = cls(device=device)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (used by tests and the auto-heal restart)."""
        cls._instance = None

    # -- core evaluation ---------------------------------------------------
    @torch.no_grad()
    def _forward_grid(self, grid: Tensor, chemistry: str) -> FieldOutput:
        xyz_t = MicrochannelPINN.prepare_inputs(grid)
        return self.model(xyz_t.to(self.device, INFERENCE_DTYPE), chemistry)

    @torch.no_grad()
    def evaluate_field(
        self,
        chemistry: str = "nmc",
        nx: int = 28,
        ny: int = 20,
        nz: int = 14,
        coolant_inlet_c: float = 26.0,
        coolant_flow_lpm: float = 6.0,
        ambient_c: float = 34.0,
        pack_current_a: float = 180.0,
        state_of_charge: float = 0.5,
        heat_watts: float = 0.0,
        thermal_ambient_s: float = 5.0,
        residual_every: int = 1,
    ) -> FieldResult:
        """Evaluate the 3-D temperature field and all derived diagnostics.

        Parameters
        ----------
        residual_every:
            Compute the expensive autodiff residual every ``k``-th call.  The
            dashboard requests residuals on every prediction (it is the research
            artefact); the telemetry websocket passes a large value so that the
            60 Hz stream stays inside the latency budget.
        """
        t0 = time.perf_counter()
        spec = get_chemistry(chemistry)

        xs = torch.linspace(0, COLD_PLATE.length_m, nx, dtype=INFERENCE_DTYPE)
        ys = torch.linspace(0, COLD_PLATE.width_m, ny, dtype=INFERENCE_DTYPE)
        zs = torch.linspace(0, COLD_PLATE.thickness_m, nz, dtype=INFERENCE_DTYPE)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        # Chunked evaluation to bound peak activation memory.
        chunk = 65536
        outs: List[Tensor] = []
        coef: Optional[FieldOutput] = None
        for start in range(0, grid.shape[0], chunk):
            out = self._forward_grid(grid[start : start + chunk], chemistry)
            outs.append(out.temperature_k.detach().cpu())
            if coef is None:
                coef = out

        temp = torch.cat(outs).reshape(nx, ny, nz) - 273.15
        temp_np = temp.numpy().astype(np.float64)

        # -- derived thermal quantities (physics, not learned heads) --------
        flow_lpm = float(coolant_flow_lpm)
        coolant = spec.coolant_props
        h_cold = heat_transfer_coefficient(flow_lpm, coolant)
        area_flux = COLD_PLATE.conductance_m2k
        q_rej = h_cold * area_flux * max(float(temp_np.max()) - coolant_inlet_c, 0.0)
        mcp = coolant["density"] * (flow_lpm / 60000.0) * coolant["cp"]
        coolant_rise = q_rej / max(mcp, 1e-6)
        coolant_outlet = coolant_inlet_c + coolant_rise

        # -- residuals ------------------------------------------------------
        residuals, weights, reynolds, dpdx = self._residuals(
            spec, flow_lpm, ambient_c, pack_current_a, state_of_charge,
            heat_watts, residual_every,
        )
        residuals["coolant_rise_k"] = float(coolant_rise)
        residuals["h_cold_w_m2k"] = float(h_cold)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.record(latency_ms)

        return FieldResult(
            chemistry=chemistry,
            axes={"x_m": xs.numpy().tolist(), "y_m": ys.numpy().tolist(),
                  "z_m": zs.numpy().tolist()},
            temperature_c=temp_np,
            max_temp_c=float(temp_np.max()),
            min_temp_c=float(temp_np.min()),
            mean_temp_c=float(temp_np.mean()),
            cell_outlet_c=float(temp_np[-1].mean()),
            coolant_outlet_c=coolant_outlet,
            heat_rejected_w=float(q_rej),
            thermal_runaway_margin_c=float(
                thermal_runaway_margin(spec, float(temp_np.max()))
            ),
            inference_latency_ms=latency_ms,
            within_latency_budget=latency_ms <= INFERENCE_LATENCY_BUDGET_MS,
            residuals=residuals,
            adaptive_weights=weights,
            coefficients={
                "dpdx_pa_per_m": dpdx,
                "reynolds": reynolds,
                "h_cold_w_m2k": float(h_cold),
                "mass_flow_kg_per_s": float(coolant["density"] * flow_lpm / 60000.0),
                "specific_heat_capacity_j_kgk": spec.cp,
                "thermal_diffusivity_m2_s": spec.thermal_diffusivity(),
            },
            reynolds=reynolds,
            flow_lpm=flow_lpm,
        )

    def _residuals(
        self,
        spec: ChemistrySpec,
        flow_lpm: float,
        ambient_c: float,
        current_a: float,
        soc: float,
        heat_watts: float,
        residual_every: int,
    ) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
        """Autodiff residual diagnostics (float64, CPU)."""
        self.latency_samples_seen = self.latency.n
        if residual_every > 1 and (self.latency.n % residual_every) != 0:
            # Report a *skip*, not a measurement. Returning zeros here would be
            # indistinguishable from a perfectly satisfied PDE to every
            # downstream consumer, and ``is_converged`` would then read True on
            # the strength of a residual that was never computed. The
            # ``evaluated`` flag is what a client must check before believing
            # any number in this dict.
            return (
                {"evaluated": 0.0, "total_physics": float("nan")},
                {},
                0.0,
                0.0,
            )

        from ..models.physics import EnergyResidual, sample_face

        dev, dt = self._residual_device, RESIDUAL_DTYPE

        # The whole diagnostic runs under enable_grad: it differentiates the
        # temperature field twice (for the Laplacian), which is the entire point.
        # `evaluate_field` is itself wrapped in `no_grad` for the fast path, so
        # without this block every `autograd.grad` call raises "element 0 of
        # tensors does not require grad" -- the residual path would be
        # unreachable outside training.
        with torch.enable_grad():
            pts = PACK_DOMAIN.sample(256, device=dev, dtype=dt)
            chem_latent = self.model.embedder(chemistry=spec.key)

            def T_fn(xyz: Tensor) -> Tensor:
                """Differentiable ``T(xyz) -> K`` at float64, for residual use."""
                x = MicrochannelPINN.prepare_inputs(xyz.to(dt))
                f, _ = self.model.encode(x, chem_latent)
                return (self.model.t_offset_k.to(dt)
                        + self.model.t_scale_k.to(dt)
                        * torch.tanh(self.model.head(f).squeeze(-1)))

            # Translate the commanded volumetric flow into the pressure gradient
            # the momentum residual expects, using the closed-form inversion of
            # the stream-function profile. Doing this (rather than an ad-hoc
            # scale factor) keeps the momentum residual physically meaningful at
            # every flow rate instead of only at the nominal one.
            dpdx_t = torch.as_tensor(
                StreamFunction(torch.zeros((), dtype=dt, device=dev))
                .dpdx_for_flow_lpm(flow_lpm),
                device=dev, dtype=dt,
            )

            residual = EnergyResidual(spec, T_fn, dpdx=dpdx_t)
            colloc = pts.clone().requires_grad_(True)

            r_solid = residual.solid_residual(colloc, current_a, soc, heat_watts)
            pde_energy = float(torch.mean(r_solid**2))

            sf = StreamFunction(dpdx_t)
            # Momentum/continuity residuals live in CHANNEL coordinates, so they
            # get their own collocation sample; the pack-domain `colloc` above
            # belongs to the solid energy equation.
            chan_pts = CHANNEL_DOMAIN.sample(128, device=dev, dtype=dt
                                             ).requires_grad_(True)
            pde_mom = float(torch.mean(sf.momentum_residual(chan_pts) ** 2))
            pde_cont = float(torch.mean(sf.continuity_residual(chan_pts) ** 2))

            wall_pts, _ = sample_face(CHANNEL_DOMAIN, "yl", 64, device=dev, dtype=dt)
            bc_noslip = float(torch.mean(
                sf.wall_no_slip_residual(wall_pts.requires_grad_(True))
            ))

            inlet_pts, _ = sample_face(PACK_DOMAIN, "inlet", 128, device=dev, dtype=dt)
            inlet_pts = inlet_pts.requires_grad_(True)
            bc_inlet = float(torch.mean((T_fn(inlet_pts) - (26.0 + 273.15)) ** 2))

            out_pts, n_out = sample_face(PACK_DOMAIN, "outlet", 128, device=dev, dtype=dt)
            out_pts = out_pts.requires_grad_(True)
            h_cold = heat_transfer_coefficient(flow_lpm, spec.coolant_props)
            bc_outlet = float(torch.mean(
                residual.convective_bc_residual(out_pts, n_out, h_cold, 26.0) ** 2
            ))

            shell_pts, nrm = sample_face(PACK_DOMAIN, "zl", 128, device=dev, dtype=dt)
            shell_pts = shell_pts.requires_grad_(True)
            bc_wall = float(torch.mean(
                residual.convective_bc_residual(
                    shell_pts, nrm, 0.0, 0.0, air_side=True, ambient_c=ambient_c
                ) ** 2
            ))

            ifc_pts, ifc_n = sample_face(PACK_DOMAIN, "z0", 128, device=dev, dtype=dt)
            ifc_pts = ifc_pts.requires_grad_(True)
            ifc = residual.interface_residual(ifc_pts, ifc_pts, ifc_n)
            bc_ifc = float(torch.mean(ifc["temperature_jump"] ** 2))

            total = (pde_energy + pde_mom + pde_cont + bc_inlet + bc_outlet
                     + bc_wall + bc_ifc + bc_noslip)

        return (
            {
                "pde_energy": pde_energy,
                "pde_momentum": pde_mom,
                "pde_continuity": pde_cont,
                "bc_inlet": bc_inlet,
                "bc_outlet": bc_outlet,
                "bc_wall": bc_wall,
                "bc_no_slip": bc_noslip,
                "bc_interface_flux": bc_ifc,
                "total_physics": total,
                "nonfinite_guard": 0.0 if math.isfinite(total) else 1.0,
                "evaluated": 1.0,
            },
            # The prior weights the trainer starts from, reported verbatim. The
            # inference path performs no optimisation, so these *are* the
            # operative weights. Returning them as numbers rather than as a prose
            # note keeps ``ResidualReport.adaptive_weights`` a mapping of floats,
            # which is what the schema and the dashboard both expect.
            {
                "physics": float(PINN_CONFIG.w_physics_init),
                "boundary": float(PINN_CONFIG.w_boundary),
                "initial": float(PINN_CONFIG.w_initial),
                "data": float(PINN_CONFIG.w_data),
            },
            float(flow_to_reynolds(torch.as_tensor(flow_lpm))),
            float(dpdx_t),
        )

    # -- point query -------------------------------------------------------
    @torch.no_grad()
    def query_point(
        self,
        point_m: Sequence[float],
        chemistry: str = "nmc",
        coolant_inlet_c: float = 26.0,
        coolant_flow_lpm: float = 6.0,
        ambient_c: float = 34.0,
        pack_current_a: float = 180.0,
        state_of_charge: float = 0.5,
    ) -> Dict[str, float]:
        """Single-point temperature, uncertainty proxy, dT/dt and local flux."""
        t0 = time.perf_counter()
        spec = get_chemistry(chemistry)
        pt = torch.tensor([list(point_m)], dtype=INFERENCE_DTYPE)

        temp = float(self.model.temperature(pt.to(self.device), chemistry)[0] - 273.15)
        base = lumped_pack_rhs(spec, temp, ambient_c, ambient_c + 12.0, 800.0,
                               pack_current_a, state_of_charge,
                               coolant_flow_lpm, coolant_inlet_c)

        # Uncertainty proxy: spread of the field over the local 3x3x3 stencil.
        eps = 0.004
        neighbourhood = pt + torch.randn(64, 3, generator=torch.Generator().manual_seed(7)) * eps
        samples = self.model.temperature_c(neighbourhood.to(self.device), chemistry)
        uncertainty = float(samples.std())

        g = float(abs(temp - coolant_inlet_c))
        flux = heat_transfer_coefficient(coolant_flow_lpm, spec.coolant_props) * g

        self.latency.record((time.perf_counter() - t0) * 1000.0)
        return {
            "temperature_c": temp,
            "uncertainty_k": uncertainty,
            "dT_dt_c_per_s": float(base),
            "local_heat_flux_w_m2": float(flux),
            "latency_ms": self.latency.ema,
        }

    # -- introspection -----------------------------------------------------
    def model_info(self) -> Dict[str, object]:
        """Payload for ``GET /api/model/info``."""
        from pathlib import Path

        present = Path(self.checkpoint_path).exists()
        return {
            "architecture": (
                "Dual-chemistry SIREN/Fourier-feature PINN with FiLM chemistry "
                "conditioning and three physics heads (R_int, dp/dx, h_amb)"
            ),
            "total_parameters": self.model.n_parameters(),
            "conditioning_dim": len(
                __import__("backend.config", fromlist=["x"]).CHEMISTRY_CONDITIONING_DIM.__str__()
            ) if False else __import__(
                "backend.config", fromlist=["CHEMISTRY_CONDITIONING_DIM"]
            ).CHEMISTRY_CONDITIONING_DIM,
            "hidden_dim": self.model.config.hidden_dim,
            "n_hidden_layers": self.model.config.n_hidden_layers,
            "fourier_features": self.model.config.n_fourier_features,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_present": present,
            "device": str(self.device),
            "dtype": str(INFERENCE_DTYPE),
            "gradient_penalty": {
                "ema_decay": self.model.config.weight_ema,
                "prior_w_data": self.model.config.w_data,
                "prior_w_physics": self.model.config.w_physics_init,
                "prior_w_boundary": self.model.config.w_boundary,
                "prior_w_initial": self.model.config.w_initial,
            },
            "trained_chemistries": list(self.model.embedder.keys),
            "latency_budget_ms": INFERENCE_LATENCY_BUDGET_MS,
            "notes": (
                "All temperatures in Kelvin inside the network; Celsius only at "
                "the API boundary. Residual diagnostics are computed in float64 on "
                "CPU because torch.autograd create_graph is unavailable on MPS."
            ),
        }


def engine(device: Optional[str] = None) -> InferenceEngine:
    """Module-level accessor used by the routers."""
    return InferenceEngine.instance(device=device)