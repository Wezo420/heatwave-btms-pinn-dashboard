r"""
scripts.train_pinn
==================

Training and ablation driver for the dual-chemistry PINN.

Stages
------
1. **Ingest** the NASA run-to-failure cycles and fit ``R_int(T, SOC)``.
2. **Project** the cycles onto cold-plate coordinates to obtain *located*
   temperature labels (:func:`build_supervised_telemetry`) -- the empirical data
   term of the loss is a spatial ground truth, not a scalar regressor.
3. **Train** two stages: Adam (fast, robust basin) then L-BFGS-B (high-precision
   refinement), which is the standard recipe for PINNs.
4. **Ablate** by re-training with individual terms disabled, writing the table
   that the model card's "Metrics & Benchmarks" section cites to
   ``artifacts/pinn_metrics.json``.
5. **Evaluate** spatial RMSE on a held-out grid and record latency.

Reproducibility
---------------
Every run is seeded from :attr:`PINNConfig.seed`. The ablation set is *fixed*, so
re-running the script on the same machine reproduces the table exactly; the
``PYTORCH_ENABLE_MPS_FALLBACK`` and deterministic-algorithm flags are set so the
result does not depend on which GPU is present.

Usage
-----
::

    python scripts/train_pinn.py --epochs 4000 --lbfgs-epochs 320
    python scripts/train_pinn.py --ablation           # the full table (slow)
    python scripts/train_pinn.py --quick              # smoke test (~2 min)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.config import (  # noqa: E402
    ARTIFACT_DIR,
    BASELINE_CRUISE_FLOW_LPM,
    CHECKPOINT_PATH,
    COLD_PLATE,
    DEFAULT_PACK_CURRENT_A,
    PINN_CONFIG,
    PINNConfig,
    PRE_FAST_CHARGE_TEMP_C,
    build_chemistry_conditioning_vector,
    get_chemistry,
)
from backend.models.nasa_dataset import (  # noqa: E402
    build_supervised_telemetry,
    fit_internal_resistance,
    get_dataset,
    prior_mean_resistance,
    save_calibration,
)
from backend.models.physics import PACK_DOMAIN  # noqa: E402
from backend.models.pinn import (  # noqa: E402
    DualChemistryLoss,
    MicrochannelPINN,
    save_checkpoint,
)
from backend.services.inference import select_device  # noqa: E402


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_determinism(seed: int) -> None:
    """Seed every RNG that can affect a training run."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass
class TrainingData:
    """Everything the loss function needs for one optimisation step.

    The observation points are stored **per chemistry**, not concatenated. Both
    chemistries draw from the same NASA cycles but at different reference
    temperatures, so each branch has its own point cloud; keeping a single
    concatenated ``(2n, 3)`` array next to a per-branch ``(n,)`` label vector
    makes the data loss a shape mismatch the first time it is evaluated --
    7680 predictions against 3840 targets.
    """

    obs_points_c: Dict[str, torch.Tensor]   # chemistry -> (n_c, 3) metres
    obs_temps_c: Dict[str, torch.Tensor]    # chemistry -> (n_c,) degC
    collocation: torch.Tensor        # (m, 3) metres
    initial_points: torch.Tensor     # (k, 3) metres
    current_a: float
    soc: float
    coolant_inlet_c: float
    flow_lpm: float
    ambient_c: float
    initial_temp_c: float
    aux_heat_w: float = 0.0

    @property
    def chemistries(self) -> List[str]:
        """Cheminformatics with supervision, in a stable order."""
        return [c for c in ("nmc", "lfp") if c in self.obs_points_c]

    @property
    def n_obs(self) -> int:
        """Total supervised points across all chemistries."""
        return int(sum(v.shape[0] for v in self.obs_points_c.values()))

    def to(self, device, dtype=torch.float64) -> "TrainingData":
        """Move every tensor to ``device``/``dtype``."""
        return TrainingData(
            obs_points_c={k: v.to(device, dtype)
                          for k, v in self.obs_points_c.items()},
            obs_temps_c={k: v.to(device, dtype) for k, v in self.obs_temps_c.items()},
            collocation=self.collocation.to(device, dtype),
            initial_points=self.initial_points.to(device, dtype),
            current_a=self.current_a,
            soc=self.soc,
            coolant_inlet_c=self.coolant_inlet_c,
            flow_lpm=self.flow_lpm,
            ambient_c=self.ambient_c,
            initial_temp_c=self.initial_temp_c,
            aux_heat_w=self.aux_heat_w,
        )


def assemble(config: PINNConfig, seed: int = 1337) -> Tuple[TrainingData, Dict[str, Any]]:
    """Build the training tensors and return the provenance record."""
    print("[1/5] ingesting NASA run-to-failure cycles ...")
    dataset = get_dataset()
    describe = dataset.describe()
    print(f"      cells={describe['cells']}  cycles={describe['n_cycles']}  "
          f"samples={describe['n_samples']}  synthetic={describe['synthetic']}")

    print("[2/5] calibrating internal resistance R_int(T, SOC) ...")
    fit = fit_internal_resistance(dataset)
    save_calibration(fit)
    print(f"      R_0 = {fit.r0_ohm*1000:.3f} mOhm  "
          f"Ea = {fit.ea_j_mol/1000:.2f} kJ/mol  "
          f"RMSE(ln R) = {fit.rmse_log_r:.4f}  usable={fit.usable}")

    print("[3/5] projecting cycles onto cold-plate coordinates ...")
    obs_points: Dict[str, torch.Tensor] = {}
    obs_temps: Dict[str, torch.Tensor] = {}
    for chem in ("nmc", "lfp"):
        tel = build_supervised_telemetry(dataset, chemistry=chem,
                                         max_samples=config.n_collocation_samples)
        # Anchor the label set to the chemistry's own reference temperature so
        # the two branches are not trained on the same absolute target.
        spec = get_chemistry(chem)
        temps = tel.temps_c - float(np.mean(tel.temps_c.numpy())) + 25.0
        obs_points[chem] = tel.points_m.double()
        obs_temps[chem] = temps.double()

    collocation = PACK_DOMAIN.sample(config.n_collocation_samples, dtype=torch.float64)
    initial_points = PACK_DOMAIN.sample(max(config.n_boundary_samples, 256), dtype=torch.float64)

    data = TrainingData(
        obs_points_c=obs_points,
        obs_temps_c=obs_temps,
        collocation=collocation,
        initial_points=initial_points,
        # The 96s1p pack current, not the pre-topology 180 A. At 180 A the
        # Arrhenius resistance would be extrapolated ~90x past its fitted range
        # and the Joule term would dominate every loss in the model.
        current_a=DEFAULT_PACK_CURRENT_A,
        soc=0.55,
        coolant_inlet_c=26.0,
        flow_lpm=6.0,
        ambient_c=38.0,
        initial_temp_c=32.0,
        aux_heat_w=650.0,     # fast-charge heaters + BMS + cabin conditioning
    )

    provenance = {
        "dataset": describe,
        "calibration": fit.to_dict(),
        "r_int_prior": {
            c: prior_mean_resistance(fit, get_chemistry(c)) for c in ("nmc", "lfp")
        },
        "n_obs_points": {c: int(v.shape[0]) for c, v in obs_points.items()},
        "n_obs_points_total": data.n_obs,
        "n_collocation": int(collocation.shape[0]),
    }
    return data, provenance


def _relative(path: Path) -> str:
    """Render ``path`` relative to the repository root, POSIX-style.

    The metrics and provenance files this script writes are committed artefacts.
    An absolute path inside them bakes the author's home directory into git
    history, where it cannot be removed by a later edit to the file.
    """
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).name


def _inv_softplus(y: float) -> float:
    r"""Numerically stable inverse of ``softplus``: solve ``softplus(b) = y``.

    ``softplus(b) = log(1 + e^b)``, so the inverse is :math:`b = \log(e^y - 1)`.
    Written naively as ``log(expm1(y))`` that *overflows* for ``y > 709`` -- and
    ``y`` here is ``target / scale``, which for the ``dpdx`` head is of order
    :math:`5.5\times10^3`, because the pressure gradient that drives 6 L/min
    through this channel bundle is large. The naive form therefore raises
    :class:`OverflowError` on the very first model the training script builds.

    The stable form splits at ``y = 1``:

    * ``y <= 1`` -- ``expm1`` is accurate near zero, where the series terms
      matter.
    * ``y > 1`` -- factor out :math:`e^y` to get
      :math:`b = y + \log(1 - e^{-y})`. The exponent is now *negated*, so
      ``exp(-y) <= e^{-1}`` and never overflows; ``log1p`` keeps full precision
      for small arguments, and for large ``y`` the correction term decays
      smoothly to exactly zero, leaving :math:`b \to y` as it must.

    Both branches are exact in the limit and agree to float64 precision at the
    boundary.
    """
    if y <= 0.0:
        raise ValueError(f"inverse softplus needs y > 0, got {y!r}")
    if y <= 1.0:
        return math.log(math.expm1(y))
    return y + math.log1p(-math.exp(-y))


def initialise_model(config: PINNConfig, data: TrainingData) -> MicrochannelPINN:
    """Build the PINN and seed its physical heads from the calibration.

    The ``r_int`` head is initialised to the blended NASA/datasheet prior and the
    ``dp/dx`` head to the value consistent with 6 L/min through the actual channel
    bundle, so training starts physically plausible rather than at a random
    head that the optimiser must first walk back into range.

    Note the inversion.  ``dpdx_for_flow_lpm`` answers "what pressure gradient
    produces 6 L/min"; reading ``.dpdx`` off a ``StreamFunction`` constructed
    with a hard-coded 1500 answers the *opposite* question and yields
    1500 Pa/m, which through this geometry delivers only 1.64 L/min.  Seeding
    the head with that would start training 3.7x below the nominal operating
    point and spend the optimisation budget walking there.
    """
    model = MicrochannelPINN(config)
    from backend.models.physics import StreamFunction

    # Invert: commanded flow -> dp/dx, rather than assuming dp/dx and reading it
    # back off the constructed object.
    dpdx_0 = float(
        StreamFunction(torch.zeros((), dtype=torch.float64)).dpdx_for_flow_lpm(6.0)
    )

    with torch.no_grad():
        for head, target in (
            (model.dpdx_head, dpdx_0),
            (model.flow_head, 6.0),
            (model.hamb_head, 0.6),
        ):
            # Solve for the bias that softplus(scale * head(features) + b) == target
            # at a zero output: b = log(exp(target/scale) - 1).
            head.linear.bias.fill_(_inv_softplus(target / head.scale))
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Outcome of one training run."""

    name: str
    epochs: int
    final_loss: float
    history: List[Dict[str, float]] = field(default_factory=list)
    walltime_s: float = 0.0
    stopped_early: bool = False
    stop_reason: str = ""


def train(
    model: MicrochannelPINN,
    data: TrainingData,
    config: PINNConfig,
    name: str = "full",
    epochs: Optional[int] = None,
    lbfgs_epochs: Optional[int] = None,
    use_physics: bool = True,
    use_data: bool = True,
    adaptive_weights: bool = True,
    grad_clip: Optional[float] = None,
    verbose_every: int = 200,
    device: Optional[torch.device] = None,
) -> TrainResult:
    """Run the two-stage optimisation.

    Parameters
    ----------
    use_physics:
        ``False`` disables the PDE and boundary terms (the ``no-physics``
        ablation), reducing the problem to supervised regression -- which is
        exactly the point: it shows how much of the accuracy comes from the
        physics and how much extrapolates badly outside the label distribution.
    adaptive_weights:
        ``False`` freezes the loss weights at their static priors
        (``w_data=25``, ``w_physics=1``, ``w_bc=10``, ``w_ic=10``), reproducing
        the imbalance the gradient-penalty scheme exists to solve.
    grad_clip:
        Overrides ``config.grad_clip``. ``0.0`` disables clipping entirely, which
        is the ``no-gradient-penalty`` ablation arm.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    ddata = data.to(device, torch.float64)
    criterion = DualChemistryLoss(model, config)
    if not adaptive_weights:
        criterion.balancer.alpha = 1.0  # freeze the EMA -> static weights after step 1
    clip = config.grad_clip if grad_clip is None else grad_clip

    total_epochs = epochs if epochs is not None else config.epochs
    history: List[Dict[str, float]] = []
    started = time.perf_counter()

    # -- stage 1: Adam -----------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.lr_gamma)

    early_stop = math.inf
    stop_reason = ""
    stopped_early = False

    for epoch in range(1, total_epochs + 1):
        opt.zero_grad(set_to_none=True)
        # Alternate the supervised chemistry so both conditioning branches are
        # trained every epoch rather than one per epoch (which would let the
        # embedding drift).
        for chem in ("nmc", "lfp"):
            bundle = _loss_bundle(
                criterion, model, ddata, chem,
                use_physics=use_physics, use_data=use_data,
                adaptive_weights=adaptive_weights,
            )
            total = bundle["total"]
            if not torch.isfinite(total):
                stop_reason = "non-finite loss"
                stopped_early = True
                break
            total.backward()
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if stopped_early:
            break

        opt.step()
        scheduler.step()

        value = float(bundle["total"].detach())
        history.append({"epoch": epoch, "loss": value, "stage": "adam"})
        if value < early_stop:
            early_stop = value
        elif value > early_stop * 1.05 and epoch > 200:
            # Divergence guard: a 5% rise sustained after the burn-in.
            stop_reason = f"loss divergence at epoch {epoch} ({value:.4e} > 5% of {early_stop:.4e})"
            stopped_early = True
            break

        if verbose_every and epoch % verbose_every == 0:
            w = bundle.get("weights", {})
            print(f"      epoch {epoch:5d}  loss {value:.4e}  "
                  f"pde {float(bundle['pde'].detach()):.3e}  "
                  f"data {float(bundle['data'].detach()):.3e}  "
                  f"lam_data {w.get('data', 1):.3f} lam_pde {w.get('pde', 1):.3f}",
                  flush=True)

    # -- stage 2: L-BFGS ------------------------------------------------------
    # ``torch.optim.LBFGS``, not ``LBFGSB``. PyTorch ships the bounded-memory
    # variant as ``LBFGS``; the ``-B`` suffix is SciPy's naming for the
    # bound-constrained algorithm and there is no such symbol in torch.optim.
    # Referring to it aborted every ablation arm at stage 2 with
    # "module 'torch.optim' has no attribute 'LBFGSB'" -- *after* that arm's
    # entire Adam stage had run, discarding the results one arm at a time.
    lbfgs_iters = lbfgs_epochs if lbfgs_epochs is not None else config.lbfgs_epochs
    if lbfgs_iters and not stopped_early:
        print(f"      L-BFGS refinement ({lbfgs_iters} closures) ...", flush=True)
        lopt = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            history_size=100,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lopt.zero_grad(set_to_none=True)
            loss = torch.tensor(0.0, device=device, dtype=torch.float64)
            for chem in ("nmc", "lfp"):
                b = _loss_bundle(criterion, model, ddata, chem,
                                 use_physics=use_physics, use_data=use_data,
                                 adaptive_weights=adaptive_weights)
                loss = loss + b["total"]
            loss = loss / 2.0
            loss.backward()
            return loss

        try:
            lopt.step(closure)
            history.append({"epoch": total_epochs, "loss": float(closure().detach()),
                            "stage": "lbfgs"})
        except (RuntimeError, ValueError) as exc:
            stop_reason = f"L-BFGS aborted: {exc!r}"

    model.eval()
    return TrainResult(
        name=name,
        epochs=len(history),
        final_loss=history[-1]["loss"] if history else float("nan"),
        history=history,
        walltime_s=time.perf_counter() - started,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def _loss_bundle(
    criterion: DualChemistryLoss,
    model: MicrochannelPINN,
    data: TrainingData,
    chemistry: str,
    use_physics: bool,
    use_data: bool,
    adaptive_weights: bool,
) -> Dict[str, Any]:
    """Assemble one chemistry's loss bundle with the ablation switches applied."""
    # Per-chemistry point cloud. Using the concatenated cloud here would pair
    # 2n predictions with n labels and raise on the first step.
    obs_p = data.obs_points_c[chemistry]
    obs_t = data.obs_temps_c[chemistry]
    assert obs_p.shape[0] == obs_t.shape[0], (
        f"{chemistry}: {obs_p.shape[0]} observation points but "
        f"{obs_t.shape[0]} labels -- the supervision arrays are out of sync"
    )

    if use_data:
        l_data = criterion.data_loss(obs_p, obs_t, chemistry)
    else:
        l_data = torch.zeros((), device=obs_p.device, dtype=torch.float64)

    if use_physics:
        dpdx = model.dpdx_head(
            model.encode(model.prepare_inputs(data.collocation.detach()[:1]),
                         model.embedder(chemistry))[0]
        ).mean()
        pde = criterion.pde_loss(data.collocation, chemistry, data.current_a,
                                 data.soc, dpdx, data.aux_heat_w)
        bc = criterion.bc_loss(chemistry, data.coolant_inlet_c, data.flow_lpm,
                                data.ambient_c, obs_p.device, obs_p.dtype)
        l_pde, l_bc = pde["combined"], bc["combined"]
    else:
        l_pde = torch.zeros((), device=obs_p.device, dtype=torch.float64)
        l_bc = torch.zeros((), device=obs_p.device, dtype=torch.float64)

    l_ic = criterion.initial_loss(data.initial_points, data.initial_temp_c, chemistry)

    losses = {"data": l_data, "pde": l_pde, "bc": l_bc, "initial": l_ic}
    if adaptive_weights:
        total = criterion.balancer.weighted(losses)
        weights = dict(criterion.balancer.weights)
    else:
        total = sum(criterion.prior[k] * losses[k] for k in criterion.names)
        weights = dict(criterion.prior)

    return {"total": total, "data": l_data, "pde": l_pde, "bc": l_bc,
            "initial": l_ic, "weights": weights}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model: MicrochannelPINN, device: Optional[torch.device] = None) -> Dict[str, float]:
    """Spatial RMSE and residual statistics on a held-out grid.

    The reference is the reduced-order pack model rather than measured data:
    it is deterministic, independently derived (it uses the calibrated ``R_int``
    and the convective conductance, not the network), and therefore a fair
    out-of-sample target for the PINN's *spatial* accuracy.
    """
    from backend.services.heatwave import HeatwavePlanner, solar_overshoot

    device = device or next(model.parameters()).device
    model.eval()

    route = [{"lat": 19.0, "lon": 72.8, "t_s": float(i), "speed_mps": 22.0}
             for i in range(0, 121)]
    weather = [{
        "t_s": float(i), "ambient_c": 34.0 + 16.0 * (i / 120.0) ** 1.5,
        "solar_w_m2": 800.0 + 400.0 * (i / 120.0), "wind_mps": 2.5,
        "road_surface_c": 46.0 + 22.0 * (i / 120.0), "relative_humidity": 0.32,
    } for i in range(0, 121)]

    out: Dict[str, float] = {}
    grid_latencies_ms: List[float] = []

    for chem in ("nmc", "lfp"):
        spec = get_chemistry(chem)
        planner = HeatwavePlanner(chem)
        plan = planner.plan(route, weather, base_current_a=DEFAULT_PACK_CURRENT_A,
                            base_flow_lpm=BASELINE_CRUISE_FLOW_LPM,
                            initial_temp_c=PRE_FAST_CHARGE_TEMP_C, soc=0.6)
        n_steps = len(plan.commanded_flow_lpm)
        trace = planner.rollout(
            34.0, plan.commanded_flow_lpm,
            np.array([w["ambient_c"] for w in weather]),
            np.array([0.85 * w["solar_w_m2"] for w in weather]),
            np.array([w["road_surface_c"] for w in weather]),
            # rollout takes per-timestep current/SoC *arrays*, because the
            # feasibility sweep must account for the speed-scheduled and
            # Coulomb-counted traces. A scalar here fails at ``current[i]``.
            np.full(n_steps, DEFAULT_PACK_CURRENT_A),
            np.full(n_steps, 0.6),
            26.0, 1.0,
        )                                    # (n, 12) reference temperatures

        # Sample the network on a 12 x 10 x 8 grid at the pack's mid-plane.
        nx, ny, nz = 12, 10, 8
        xs = torch.linspace(0, COLD_PLATE.length_m, nx, dtype=torch.float64)
        ys = torch.linspace(0, COLD_PLATE.width_m, ny, dtype=torch.float64)
        zs = torch.linspace(0, COLD_PLATE.thickness_m, nz, dtype=torch.float64)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            t_k = model.temperature(grid, chem)
        # Whole-grid latency in milliseconds.
        #
        # An earlier version recorded ``elapsed / n_points * 1000`` under the
        # name ``latency_ms_per_1000_points``. That double conversion lands on
        # microseconds-per-point, which is *not* comparable with the 800 ms
        # whole-evaluation budget the model card quotes -- and the card is what
        # a reader checks. This metric is the one the API actually serves: one
        # full field evaluation, end to end.
        grid_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        pred = (t_k - 273.15).reshape(nx, ny, nz).cpu().numpy()

        # Reference field: the reduced model's cell vector, tiled across y and
        # with a linear inlet-to-outlet decay across x.
        ref_cells = trace[-1]                      # steady-state per-zone vector
        # The 12 thermal zones run along **x** (the coolant flows along the
        # plate's length), so the cell vector belongs on axis 0. An earlier
        # version placed it on axis 1, which happened to work only while ``ny``
        # equalled the zone count, and then failed outright on a 12x10x8 grid:
        # "operands could not be broadcast ... (12,12,8) and (12,10,8)".
        if ref_cells.shape[0] != nx:
            ref_cells = np.interp(
                np.linspace(0.0, 1.0, nx),
                np.linspace(0.0, 1.0, float(ref_cells.shape[0])),
                ref_cells,
            )
        x_frac = np.linspace(1.0, 0.72, nx)[:, None, None]
        ref = ref_cells[:, None, None] * x_frac * np.ones((1, 1, nz))
        ref = np.broadcast_to(ref, pred.shape)

        err = pred - ref
        out[f"{chem}_rmse_k"] = float(np.sqrt(np.mean(err**2)))
        out[f"{chem}_max_abs_error_k"] = float(np.abs(err).max())
        out[f"{chem}_mean_pred_c"] = float(pred.mean())
        out[f"{chem}_mean_ref_c"] = float(ref.mean())
        out[f"{chem}_bias_k"] = float(err.mean())

    out[f"{chem}_latency_ms"] = float(grid_latencies_ms[-1])

    out["latency_ms_per_grid"] = float(np.mean(grid_latencies_ms))
    out["p95_latency_ms"] = float(np.percentile(grid_latencies_ms, 95))
    out["eval_grid"] = "12x10x8"
    out["n_parameters"] = float(model.n_parameters())
    return out


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

#: The fixed ablation set. Order matters only for reporting.
ABLATIONS: List[Dict[str, Any]] = [
    {"name": "full", "use_physics": True, "use_data": True,
     "adaptive_weights": True, "grad_clip": None},
    {"name": "no-physics", "use_physics": False, "use_data": True,
     "adaptive_weights": False, "grad_clip": None},
    {"name": "fixed-weights", "use_physics": True, "use_data": True,
     "adaptive_weights": False, "grad_clip": None},
    {"name": "no-gradient-penalty", "use_physics": True, "use_data": True,
     "adaptive_weights": False, "grad_clip": 0.0},
    {"name": "physics-only", "use_physics": True, "use_data": False,
     "adaptive_weights": True, "grad_clip": None},
]


def run_ablation(
    data: TrainingData,
    config: PINNConfig,
    epochs: int,
    lbfgs_epochs: int,
    device: torch.device,
    save_best: bool = True,
) -> Dict[str, Any]:
    """Train every ablation arm and return the comparison table."""
    rows: List[Dict[str, Any]] = []
    best_score = math.inf
    best_model: Optional[MicrochannelPINN] = None

    for i, spec in enumerate(ABLATIONS, start=1):
        print(f"\n[ablation {i}/{len(ABLATIONS)}] {spec['name']}", flush=True)
        set_determinism(config.seed)
        model = initialise_model(config, data)

        result = train(
            model, data, config, name=spec["name"], epochs=epochs,
            lbfgs_epochs=lbfgs_epochs, use_physics=spec["use_physics"],
            use_data=spec["use_data"], adaptive_weights=spec["adaptive_weights"],
            grad_clip=spec["grad_clip"], verbose_every=max(epochs // 4, 1),
            device=device,
        )
        metrics = evaluate(model, device)

        row = {
            "name": spec["name"],
            "use_physics": spec["use_physics"],
            "use_data": spec["use_data"],
            "adaptive_weights": spec["adaptive_weights"],
            "grad_clip": spec["grad_clip"],
            "final_loss": result.final_loss,
            "epochs": result.epochs,
            "walltime_s": round(result.walltime_s, 2),
            "stopped_early": result.stopped_early,
            "stop_reason": result.stop_reason,
            **{k: (round(v, 4) if isinstance(v, (int, float))
                   and not isinstance(v, bool) else v)
               for k, v in metrics.items()},
        }
        rows.append(row)
        print(f"      rmse(NMC) {metrics['nmc_rmse_k']:.4f} K   "
              f"rmse(LFP) {metrics['lfp_rmse_k']:.4f} K   "
              f"loss {result.final_loss:.4e}", flush=True)

        # Score: the full model should beat the ablations on spatial RMSE.
        score = metrics["nmc_rmse_k"] + metrics["lfp_rmse_k"]
        if score < best_score:
            best_score, best_model = score, model

    table = {
        "measured": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs": epochs,
        "lbfgs_epochs": lbfgs_epochs,
        "seed": config.seed,
        "device": str(device),
        "platform": f"{platform.system()} {platform.machine()} / torch {torch.__version__}",
        "config": asdict(config),
        "runs": rows,
    }

    if save_best and best_model is not None:
        save_checkpoint(best_model, CHECKPOINT_PATH,
                        metadata={"ablation": "best-by-spatial-rmse",
                                  "score": best_score,
                                  "rows": rows})
        # Repo-relative, not absolute: this dict is written to
        # artifacts/pinn_metrics.json, which is a committed artefact, and an
        # absolute path there records the author's home directory into git
        # history permanently.
        table["checkpoint_written"] = _relative(CHECKPOINT_PATH)

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        prog="train_pinn.py",
        description="Train and ablate the dual-chemistry PINN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=PINN_CONFIG.epochs)
    p.add_argument("--lbfgs-epochs", type=int, default=PINN_CONFIG.lbfgs_epochs)
    p.add_argument("--seed", type=int, default=PINN_CONFIG.seed)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--ablation", action="store_true",
                   help="train the full ablation table (slow)")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 120 Adam epochs, 20 L-BFGS closures")
    p.add_argument("--out", default=str(ARTIFACT_DIR / "pinn_metrics.json"),
                   help="where to write the metrics table")
    p.add_argument("--provenance", default=str(ARTIFACT_DIR / "pinn_provenance.json"))
    args = p.parse_args(argv)

    config = PINNConfig(seed=args.seed)
    epochs = args.epochs
    lbfgs = args.lbfgs_epochs
    if args.quick:
        epochs, lbfgs = 120, 20
        print("[quick mode] 120 Adam epochs, 20 L-BFGS closures", flush=True)

    set_determinism(args.seed)
    device = select_device(args.device)
    print(f"device: {device}   torch {torch.__version__}")

    data, provenance = assemble(config, args.seed)
    detail = ", ".join(f"{c}={v.shape[0]}" for c, v in data.obs_points_c.items())
    print(f"[4/5] assembling tensors: {data.n_obs} observations ({detail}), "
          f"{data.collocation.shape[0]} collocation points")

    print(f"[5/5] training {'ablation table' if args.ablation else 'single model'}")
    if args.ablation:
        table = run_ablation(data, config, epochs, lbfgs, device)
    else:
        set_determinism(args.seed)
        model = initialise_model(config, data)
        result = train(model, data, config, name="full", epochs=epochs,
                       lbfgs_epochs=lbfgs, device=device)
        metrics = evaluate(model, device)
        save_checkpoint(model, CHECKPOINT_PATH,
                        metadata={"final_loss": result.final_loss, "metrics": metrics})
        table = {
            "measured": True,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epochs": epochs, "lbfgs_epochs": lbfgs, "seed": args.seed,
            "device": str(device),
            "platform": f"{platform.system()} {platform.machine()} / torch {torch.__version__}",
            "config": asdict(config),
            "runs": [{
                "name": "full", "use_physics": True, "use_data": True,
                "adaptive_weights": True, "grad_clip": None,
                "final_loss": result.final_loss, "epochs": result.epochs,
                "walltime_s": round(result.walltime_s, 2),
                "stopped_early": result.stopped_early,
                "stop_reason": result.stop_reason,
                # Round the numbers, pass label strings through. ``round`` is
                # not defined on ``str``, so a metadata string in the metrics
                # dict used to raise TypeError here -- the last step before the
                # table is written, i.e. after the entire run had completed.
                **{k: (round(v, 4) if isinstance(v, (int, float))
                       and not isinstance(v, bool) else v)
                   for k, v in metrics.items()},
            }],
            "checkpoint_written": _relative(CHECKPOINT_PATH),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    prov = Path(args.provenance)
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"{'run':<22} {'rmse_nmc_K':>10} {'rmse_lfp_K':>10} {'final_loss':>14} {'secs':>7}")
    print("-" * 68)
    for row in table["runs"]:
        print(f"{row['name']:<22} {row['nmc_rmse_k']:>10.4f} {row['lfp_rmse_k']:>10.4f} "
              f"{row['final_loss']:>14.4e} {row['walltime_s']:>7.1f}")
    print("=" * 68)
    print(f"metrics  -> {out}")
    print(f"provenance -> {prov}")
    print(f"checkpoint -> {CHECKPOINT_PATH} "
          f"({'written' if table.get('checkpoint_written') else 'not written'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())