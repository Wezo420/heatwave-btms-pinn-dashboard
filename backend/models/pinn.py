r"""
backend.models.pinn
==================

Dual-chemistry Physics-Informed Neural Network for the 3-D temperature field of
a liquid-cooled microchannel battery pack.

Architecture
------------
::

    (x, y, z, t)  --RandomFourierFeatures(48, sigma=8)-->  (x̃, ỹ, z̃, t̃)
                                              |
                                          SIREN trunk (5 x 128, w0=12/4.5)
                                              |
                                    +-----o-------+  (residual trunk)
                                    |             |
                       chem MLP(12->32) --> FiLM   |
                                    |     g, b  |
                                    v             v
                              concat(h, chem, g*h + b)
                                              |
                                       linear head ->  T  [K]
                                              |
                   +--------------------------+---------------------------+
                   |                          |                           |
            R_int log-head            bulk-flow log-head          pressure log-head
            R(T,SOC) [ohm]             Q [L/min]                  G = -dp/dx [Pa/m]

Three physical scalars are learned *jointly with* the temperature field:

1.  ``r_int`` -- the NASA-calibrated internal resistance (the network predicts
    its temperature/SOC dependence; the NASA run-to-failure fit supplies the
    prior mean and its uncertainty, see :mod:`backend.models.nasa_dataset`).
2.  ``dpdx``  -- the microchannel pressure gradient, which fixes the coolant
    mass flow via the stream-function ansatz.  Making ``dpdx`` a network output
    is what turns the coolant loop into a *learned* rather than prescribed
    boundary condition: the PDE residual alone determines the flow that is
    consistent with the observed temperature field.
3.  ``h_amb`` -- the external convection coefficient multiplier, absorbing
    un-modelled stack conduction and solar geometry.

Dual-chemistry conditioning
---------------------------
Chemistry enters through two parallel paths, both driven by the 12-dimensional
conditioning vector of :func:`backend.config.build_chemistry_conditioning_vector`:

* a **learned embedding** ``E in R^{2 x chem_dim}`` indexed by the chemistry key,
  and
* a **FiLM modulation** of every hidden trunk activation, ``h <- h * (1 + g) + b``.

The FiLM path is what makes the model *multi-physics* rather than *multi-output*:
because it re-parameterises the trunk itself, an NMC query and an LFP query
traverse genuinely different functions, which is required given the ~2.3x spread
in ``c_p`` and the different ``R_int(T)`` Arrhenius slopes.

Gradient penalty loss rebalancing
--------------------------------
The PDE residual (``r_s``, O(1)-O(1e3) W/m^3), the boundary residual
(``O(1e-2)`` K) and the data residual (``O(1e-1)`` K) start three to five orders
of magnitude apart.  A fixed weighting either stalls the PDE or ignores the
boundaries.  We therefore follow Wang, Teng & Perdikaris (2021) and use an
adaptive weight driven by the **gradient-norm ratio at the last shared layer**::

    L_i_hat = w_i * ||grad_{theta_shared} L_i||_2
    lambda_i <- (1 - alpha) * lambda_i + alpha * L_i_hat / max_j(L_j_hat)
    L_total   = sum_i lambda_i * L_i

The EMA (``alpha = 0.999``) stabilises the ratio, and a gradient-norm floor keeps
``lambda_i`` from collapsing onto a vanishing term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..config import (
    CHEMISTRY_CONDITIONING_DIM,
    CHEMISTRY_REGISTRY,
    DEFAULT_CHEMISTRY,
    INFERENCE_LATENCY_BUDGET_MS,
    PINN_CONFIG,
    PINNConfig,
    ChemistrySpec,
    build_chemistry_conditioning_vector,
    get_chemistry,
)
from .physics import (
    CHANNEL_DOMAIN,
    PACK_DOMAIN,
    StreamFunction,
    sample_face,
)

# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------


class SirenLayer(nn.Module):
    r"""SIREN layer ``sin(w0 * (W x + b))``.

    Siren et al. (2020) show that the sine activations with a geometrically
    re-scaled first-layer frequency represent the derivatives required by a PDE
    residual (``\\nabla^2 T`` needs the network to admit second derivatives of
    bounded variation) far better than ReLU/tanh MLPs, which exhibit spectral
    bias and produce spurious residual minima.
    """

    def __init__(self, in_features: int, out_features: int, w0: float = 1.0,
                 is_first: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters(is_first)

    def reset_parameters(self, is_first: bool) -> None:
        """SIREN initialisation: U(-1/in, 1/in), first layer U(-1/in, 1/in)."""
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                            1.0 / self.in_features)
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound) if not is_first else \
                self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the layer with the layer's own ``w0``."""
        return torch.sin(self.w0 * self.linear(x))


class RandomFourierFeatures(nn.Module):
    """Axis-aligned Fourier features for high-frequency spatial detail.

    The trunk is fed ``[sin(2 pi B x), cos(2 pi B x)]`` with ``B`` a fixed Gaussian
    draw scaled by ``sigma``.  This lifts the SIREN's effective bandwidth so a
    ~0.36 m cold plate with 2.2 mm channels needs no depth increase.  ``sigma``
    is annealed during training (a *spectral curriculum*): start coarse for a
    well-conditioned basin, end fine for boundary-layer accuracy.

    Output width
    ------------
    Each frequency contributes a sine and a cosine, so the flattened width is
    ``2 * n_features``.  The factor of two is easy to lose: sizing the trunk from
    ``n_features`` instead makes the forward pass fail with
    ``mat1 and mat2 shapes cannot be multiplied (N x n_features and
    N x 2*n_features)``.  :attr:`out_dim` is therefore computed once here and
    used by the trunk, so the two can never drift apart.
    """

    def __init__(self, in_dim: int, n_features: int, sigma: float = 1.0) -> None:
        r"""Random Fourier features with Gaussian frequencies ``B ~ N(0, sigma^2)``.

        ``sigma`` is the single most consequential number in this class.  The
        Fourier term is ``sin(2*pi * sigma * x @ B)``, so the per-axis
        wavenumber is ``2*pi*sigma*b`` and the network's *second* derivative --
        which is exactly what the physics residual differentiates -- scales as
        ``k^2``:

        ===============  ==================  =====================  ==============
        ``sigma``       max ``k`` [1/unit]  ``k^2`` over the 12 mm   lap of a
                                           thickness               35 K field
        ===============  ==================  =====================  ==============
        ``0.5``          8.7                 7.6e4                  2.6e6
        ``1.0``          17.4                3.0e5                  1.1e7
        ``8.0``          138.9               1.9e7                  6.8e8
        ===============  ==================  =====================  ==============

        This class previously defaulted to ``sigma = 8.0``.  The resulting
        Laplacian of ~5e8 K/m^2 makes the solid energy residual ~5e8 W/m^3 at
        initialisation, whose square is 2e17 -- sixteen orders of magnitude
        above the data term.  The gradient-norm balancer then assigns the PDE a
        weight of 1.0 and everything else 1e-3, so the model can only chase a
        residual it has no gradient budget left to reduce, and the training loss
        never moves.  The measured Laplacian was 4.8e8 K/m^2 against a physical
        source term of ~2e4 W/m^3: the initial guess was simply not physical.

        ``sigma = 1.0`` keeps the Fourier term informative (it spans the unit
        cube well) while putting the initial second derivative inside the range
        where the physics and data terms are comparable, which is the
        precondition the gradient-norm balancing scheme assumes.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features
        self.sigma = float(sigma)
        self.register_buffer("B", torch.randn(in_dim, n_features) * sigma)
        self.sigma_scale = 1.0
        self.register_buffer("phase", torch.rand(n_features) * 2.0 * math.pi)

    @property
    def out_dim(self) -> int:
        """Flattened feature width, ``2 * n_features``.

        Note this is **not** ``in_dim * 2 * n_features``: ``x @ B`` already mixes
        the input axes, since ``B`` is a full ``(in_dim, n_features)`` Gaussian
        draw.  Every one of the ``n_features`` frequencies therefore spans all
        input axes, which is the standard formulation of random Fourier features.
        A per-axis (block-diagonal) variant would give ``in_dim`` times as many
        features but correlates strongly across scales, which the SIREN trunk's
        own ``w0`` already handles.
        """
        return 2 * self.n_features

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(n, in_dim)`` -> ``(n, 2 * n_features)``.

        The buffers are cast to the input dtype on every call.  They are
        registered as float32 at construction (so a checkpoint serialises
        predictably), but the residual operators run the network in float64 for
        double-backward headroom -- and a float64 input against a float32 buffer
        raises ``expected m1 and m2 to have the same dtype``.  Casting here keeps
        one set of weights usable at both precisions, which is what lets the
        deployment dtype and the training dtype differ.
        """
        b = self.B.to(dtype=x.dtype)
        phase = self.phase.to(dtype=x.dtype)
        proj = (2.0 * math.pi * self.sigma_scale) * (x @ b) + phase
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation producing per-unit ``(gamma, beta)``."""

    def __init__(self, cond_dim: int, n_units: int, zero_init_beta: bool = True) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, n_units)
        self.beta = nn.Linear(cond_dim, n_units)
        if zero_init_beta:
            # Zero-init beta keeps the *untrained* network identical for both
            # chemistries, so the two branches start from a shared solution and
            # diverge only where the data/physics demand it. This is what makes
            # a single set of weights valid for two chemistries.
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(gamma, beta)`` broadcastable against the trunk activations."""
        return self.gamma(cond), self.beta(cond)


class ChemistryConditioner(nn.Module):
    """Turns a chemistry key (or explicit descriptor vector) into a latent code."""

    def __init__(self, chem_dim: int = PINN_CONFIG.chem_dim,
                 cond_dim: int = CHEMISTRY_CONDITIONING_DIM) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(CHEMISTRY_REGISTRY), chem_dim)
        nn.init.normal_(self.embedding.weight, std=0.35)
        self.projector = nn.Sequential(
            nn.Linear(cond_dim, chem_dim),
            nn.SiLU(),
            nn.Linear(chem_dim, chem_dim),
        )
        self.keys = list(CHEMISTRY_REGISTRY.keys())

    def index(self, chemistry: str | Sequence[str]) -> Tensor:
        """Map chemistry key(s) to embedding indices."""
        if isinstance(chemistry, str):
            idx = self.keys.index(chemistry.lower())
            return torch.tensor([idx], dtype=torch.long, device=self.embedding.weight.device)
        return torch.tensor([self.keys.index(k.lower()) for k in chemistry],
                            dtype=torch.long, device=self.embedding.weight.device)

    def descriptors(self, chemistry: str | Sequence[str]) -> Tensor:
        """Build the normalised descriptor matrix for the given key(s)."""
        keys = [chemistry] if isinstance(chemistry, str) else list(chemistry)
        return torch.tensor(
            [build_chemistry_conditioning_vector(get_chemistry(k)) for k in keys],
            dtype=torch.float32, device=self.embedding.weight.device,
        )

    def forward(self, chemistry: str | Sequence[str]) -> Tensor:
        """Return the ``(n, chem_dim)`` conditioning latent for ``chemistry``."""
        idx = self.index(chemistry)
        desc = self.descriptors(chemistry)
        return self.embedding(idx) + self.projector(desc)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------


class PositiveHead(nn.Module):
    """Linear head followed by a softplus positivity transform.

    Physics-constrained heads: ``R_int > 0``, ``Q > 0`` and ``dp/dx > 0`` are not
    optional -- a negative flow rate would make ``\\mathbf{u}\\cdot\\nabla T``
    physically meaningless and silently corrupt the adjoint residual.
    """

    def __init__(self, in_features: int, beta: float = -3.0, scale: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        nn.init.normal_(self.linear.weight, std=0.05)
        nn.init.constant_(self.linear.bias, beta)
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """Return a strictly positive ``(n,)`` tensor."""
        return self.scale * nn.functional.softplus(self.linear(x).squeeze(-1)) + 1e-8


# ---------------------------------------------------------------------------
# The PINN
# ---------------------------------------------------------------------------


@dataclass
class FieldOutput:
    """Container returned by :meth:`MicrochannelPINN.temperature`."""

    temperature_k: Tensor
    r_int_ohm: Tensor
    flow_lpm: Tensor
    dpdx: Tensor
    h_amb_scale: Tensor
    reynolds: Tensor


class MicrochannelPINN(nn.Module):
    r"""Physics-informed network for the 3-D pack temperature field.

    Parameters
    ----------
    config:
        :class:`backend.config.PINNConfig` holding widths / depths / optimizer
        hyper-parameters.
    condition_dim:
        Dimensionality of the raw input ``(x, y, z, t)``.  ``t`` is expressed in
        *normalised* units (hours since t0) so the SIREN frequencies are
        commensurate with the spatial coordinates.
    n_chemistries:
        Size of the chemistry embedding table.

    Notes
    -----
    *The model is chemistry-agnostic at the parameter level*: the same weights
    serve NMC and LFP.  This is verified by :func:`chemistry_invariance_report`.
    """

    def __init__(self, config: PINNConfig = PINN_CONFIG,
                 condition_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.condition_dim = int(condition_dim if condition_dim is not None
                                 else config.condition_dim)

        self.fourier = RandomFourierFeatures(
            in_dim=self.condition_dim,
            n_features=config.n_fourier_features,
            # The frequency *band* is a config value in its own right. It used
            # to be tied to the feature count (``n_features / 6``), which made
            # adding capacity *also* sharpen the band -- and since the physics
            # residual differentiates the field twice, a sharper band means an
            # ``n^2`` larger initial residual. See PINNConfig.fourier_sigma.
            sigma=config.fourier_sigma,
        )
        # Sized from the module's own `out_dim` (in_dim * 2 * n_features) rather
        # than recomputed here, so the two can never disagree.
        trunk_in = self.fourier.out_dim

        self.embedder = ChemistryConditioner(chem_dim=config.chem_dim)

        layers: List[nn.Module] = [SirenLayer(trunk_in, config.hidden_dim,
                                              w0=config.w0, is_first=True)]
        for _ in range(config.n_hidden_layers - 1):
            layers.append(SirenLayer(config.hidden_dim, config.hidden_dim,
                                     w0=config.hidden_w0, is_first=False))
        self.trunk = nn.Sequential(*layers)

        self.film = nn.ModuleList(
            [FiLMLayer(config.chem_dim, config.hidden_dim) for _ in range(config.n_hidden_layers)]
        )

        head_in = 2 * config.hidden_dim + config.chem_dim
        self.head = nn.Linear(head_in, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Physical scalar heads (see module docstring).
        self.r_int_head = PositiveHead(head_in, beta=-3.0, scale=0.02)
        self.flow_head = PositiveHead(head_in, beta=math.log(math.e**1.7 - 1.0), scale=1.0)
        self.dpdx_head = PositiveHead(head_in, beta=math.log(math.e**0.5 - 1.0), scale=1.0)
        self.hamb_head = PositiveHead(head_in, beta=0.0, scale=0.6)

        # A small learnable ambient/temperature calibration offset in kelvin keeps
        # the network from having to encode the trivial mean.
        self.register_buffer("t_offset_k", torch.tensor(298.15, dtype=torch.float32))
        self.register_buffer("t_scale_k", torch.tensor(35.0, dtype=torch.float32))

    def n_parameters(self) -> int:
        """Total parameter count.

        Counts every parameter, not only those with ``requires_grad``: the
        inference engine freezes the weights after construction (no gradients are
        needed to predict), and a filter on ``requires_grad`` would then report
        zero for a fully-built deployed model -- which is exactly the number an
        edge-deployment budget check needs to be true.
        """
        return sum(p.numel() for p in self.parameters())

    # -- forward -----------------------------------------------------------
    def encode(self, xyz_t: Tensor, chem_latent: Tensor) -> Tuple[Tensor, Tensor]:
        """Shared trunk computation; returns ``(feature, last_hidden)``.

        The whole trunk runs in the *parameter* dtype, whatever the caller's
        input dtype.  That matters because the residual operators need float64
        (the double-backward chain for the Laplacian cancels badly in float32)
        while the deployment path is float32 -- so the same module is entered at
        two precisions.  Casting the parameters instead of the input keeps one
        set of weights valid for both and avoids duplicating them.
        """
        param_dtype = self.trunk[0].linear.weight.dtype
        xyz_t = xyz_t.to(param_dtype)
        chem_latent = chem_latent.to(param_dtype)

        feats = self.fourier(xyz_t)
        h = self.trunk[0](feats)
        last_hidden = h
        for i in range(1, len(self.trunk)):
            gamma, beta = self.film[i - 1](chem_latent)
            if gamma.shape[0] == 1 and h.shape[0] > 1:
                gamma = gamma.expand(h.shape[0], -1)
                beta = beta.expand(h.shape[0], -1)
            h = self.trunk[i](h * (1.0 + gamma) + beta)
            last_hidden = h
        return torch.cat([h, last_hidden, chem_latent.expand(h.shape[0], -1)], dim=-1), last_hidden

    def forward(
        self,
        xyz_t: Tensor,
        chemistry: str | Sequence[str] = DEFAULT_CHEMISTRY,
    ) -> FieldOutput:
        r"""Evaluate the PINN.

        Parameters
        ----------
        xyz_t:
            ``(n, 4)`` tensor of ``(x/L, y/W, z/H, t)`` -- **normalised**
            coordinates.  Callers using metres should divide by
            :attr:`backend.config.COLD_PLATE` extents; ``prepare_inputs`` does it.
        chemistry:
            ``"nmc"``, ``"lfp"``, or a sequence matching ``xyz_t.shape[0]``.

        Returns
        -------
        FieldOutput
            ``temperature_k`` in kelvin, plus the learned physical scalars.
        """
        chem_latent = self.embedder(chemistry)
        feature, _ = self.encode(xyz_t, chem_latent)

        # Affine temperature decoder: T = t_offset + t_scale * tanh(.).  tanh keeps
        # the field in a bounded range so a diverging training run trips the
        # non-finite guard in the loss rather than producing inf cells.
        raw_t = torch.tanh(self.head(feature).squeeze(-1))
        temperature_k = self.t_offset_k + self.t_scale_k * raw_t

        r_int = self.r_int_head(feature).mean()
        flow = self.flow_head(feature).mean()
        return FieldOutput(
            temperature_k=temperature_k,
            r_int_ohm=r_int,
            flow_lpm=flow,
            dpdx=self.dpdx_head(feature).mean(),
            h_amb_scale=self.hamb_head(feature).mean(),
            reynolds=flow_to_reynolds(flow),
        )

    # -- convenience wrappers ---------------------------------------------
    def temperature(
        self,
        xyz_m: Tensor,
        chemistry: str | Sequence[str] = DEFAULT_CHEMISTRY,
        t_hours: float | Tensor = 0.0,
    ) -> Tensor:
        """Temperature in kelvin at physical points ``xyz_m`` (metres)."""
        xyz_t = self.prepare_inputs(xyz_m, t_hours)
        return self.forward(xyz_t, chemistry).temperature_k

    def temperature_c(
        self,
        xyz_m: Tensor,
        chemistry: str | Sequence[str] = DEFAULT_CHEMISTRY,
        t_hours: float | Tensor = 0.0,
    ) -> Tensor:
        """Temperature in degrees Celsius at physical points ``xyz_m``."""
        return self.temperature(xyz_m, chemistry, t_hours) - 273.15

    @staticmethod
    def prepare_inputs(xyz_m: Tensor, t_hours: float | Tensor = 0.0) -> Tensor:
        """Normalise metric coordinates into the network's input space."""
        from ..config import COLD_PLATE

        span = torch.tensor(
            [COLD_PLATE.length_m, COLD_PLATE.width_m, COLD_PLATE.thickness_m],
            dtype=xyz_m.dtype, device=xyz_m.device,
        )
        xyz_n = xyz_m[..., :3] / span
        if not torch.is_tensor(t_hours):
            t_hours = torch.as_tensor(t_hours, dtype=xyz_m.dtype, device=xyz_m.device)
        t_n = (t_hours if t_hours.dim() else t_hours).reshape(-1, 1) * torch.ones(
            xyz_n.shape[0], 1, dtype=xyz_n.dtype, device=xyz_n.device
        ) if t_hours.dim() > 0 else torch.full((xyz_n.shape[0], 1), float(t_hours),
                                                dtype=xyz_n.dtype, device=xyz_n.device)
        return torch.cat([xyz_n, t_n], dim=-1)

    def physics_callable(
        self,
        chemistry: str,
        phase: str = "solid",
        t_hours: float = 0.0,
    ) -> Callable[[Tensor], Tensor]:
        r"""Return ``T(xyz_metres) -> T_kelvin`` as a differentiable callable.

        ``phase="fluid"`` re-evaluates the same trunk on points inside the channel
        bundle, offset in ``z`` so the fluid branch sees a shifted support --
        this is the reduced conjugate treatment used when a single trunk carries
        both phases.  A full two-trunk variant is a drop-in replacement.
        """
        offset = 0.0 if phase == "solid" else 0.5

        def fn(xyz_m: Tensor) -> Tensor:
            shifted = xyz_m.clone()
            shifted = shifted + torch.tensor(
                [0.0, 0.0, offset * 0.004], dtype=xyz_m.dtype, device=xyz_m.device
            )
            return self.forward(self.prepare_inputs(shifted, t_hours), chemistry).temperature_k

        return fn


def flow_to_reynolds(flow_lpm: Tensor) -> Tensor:
    """Bundle Reynolds number implied by a volumetric flow in L/min."""
    from ..config import COLD_PLATE
    from ..config import CHEMISTRY_REGISTRY

    coolant = CHEMISTRY_REGISTRY[DEFAULT_CHEMISTRY].coolant_props
    u_mean = (flow_lpm / 60000.0) / COLD_PLATE.fluid_area_m2
    return coolant["density"] * u_mean * COLD_PLATE.hydraulic_diameter_m / coolant["mu"]


# ---------------------------------------------------------------------------
# Gradient-penalty loss balancing
# ---------------------------------------------------------------------------


class GradientNormBalancer:
    r"""Wang-teng-Perdikaris adaptive loss weights from gradient-norm ratios.

    The balance is computed on the parameters of the **last shared layer**
    (here, the trunk SIREN blocks and the FiLM modulators), because that is the
    only place where every loss term has a non-trivial gradient; the
    chemistry-specific heads are excluded from the ratio so that the balancer
    does not bias toward whichever term happens to own the most parameters.

    Parameters
    ----------
    n_terms:
        Number of loss terms to balance.
    alpha:
        EMA decay of the running weights (0.999 in the shipped config).
    floor:
        Minimum gradient norm, guarding against division by ~0 when a term is
        already satisfied (its gradient legitimately vanishes).
    """

    def __init__(self, names: Sequence[str], shared_params: Sequence[nn.Parameter],
                 alpha: float = 0.999, floor: float = 1e-8) -> None:
        self.names = list(names)
        self.shared_params = list(shared_params)
        self.alpha = alpha
        self.floor = floor
        self.weights: Dict[str, float] = {n: 1.0 for n in names}
        self._running: Dict[str, float] = {n: 0.0 for n in names}
        self._initialised = False
        self.history: List[Dict[str, float]] = []

    def step(self, losses: Mapping[str, Tensor], retain_graph: bool = True) -> Dict[str, float]:
        """Recompute the weights; returns ``{name: weight}``.

        Each term is differentiated *separally* (``retain_graph=True``) so that
        ``p.grad`` holds that term's contribution alone.  Reading a single
        accumulated gradient instead would give every term the same norm and
        therefore identical weights -- which is exactly the failure the balancer
        exists to prevent, and one that produces a plausible-looking model that
        has silently degenerated to fixed weights.

        ``.grad`` is zeroed before each term's backward so that accumulation from
        a previous term cannot leak in.
        """
        norms: Dict[str, float] = {}
        for name in self.names:
            loss = losses[name]
            for p in self.shared_params:
                if p.grad is not None:
                    p.grad = None
            if loss.requires_grad:
                loss.backward(retain_graph=True)
            norms[name] = self._grad_norm(loss)

        if not self._initialised:
            for n in self.names:
                self._running[n] = max(norms[n], self.floor)
            self._initialised = True
        else:
            for n in self.names:
                self._running[n] = (
                    self.alpha * self._running[n]
                    + (1.0 - self.alpha) * max(norms[n], self.floor)
                )

        denom = max(self._running.values())
        for n in self.names:
            ratio = self._running[n] / max(denom, self.floor)
            self.weights[n] = float(np.clip(ratio, 1e-3, 1e3))

        # Leave the shared parameters' gradients clean: the caller is about to
        # call backward() on the composite loss, and a stale gradient would be
        # accumulated into it.
        for p in self.shared_params:
            if p.grad is not None:
                p.grad = None

        self.history.append({"raw_norm_" + k: v for k, v in norms.items()})
        return dict(self.weights)

    @torch.no_grad()
    def _grad_norm(self, loss: Tensor) -> float:
        """L2 norm of ``d loss / d theta_shared``, or 0 when undefined."""
        total = 0.0
        for p in self.shared_params:
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return math.sqrt(total)

    def weighted(self, losses: Mapping[str, Tensor]) -> Tensor:
        """Recompute the weights *and* return the scalar composite loss."""
        self.step(losses, retain_graph=True)
        total: Tensor | float = 0.0
        for n in self.names:
            total = total + self.weights[n] * losses[n]
        return total  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Training-time losses
# ---------------------------------------------------------------------------


@dataclass
class LossBundle:
    """Individual and weighted loss terms for one optimisation step."""

    data: Tensor
    pde: Tensor
    bc: Tensor
    initial: Tensor
    total: Tensor
    weights: Dict[str, float]
    diagnostics: Dict[str, float] = field(default_factory=dict)


class DualChemistryLoss(nn.Module):
    r"""The full PINN objective with conjugate-heat-transfer residuals.

    .. math::
        \mathcal{L} = \lambda_{d}\,\underbrace{\tfrac1{|\Omega_d|}\sum
            (T_{\text{obs}} - \hat T)^2}_{\text{NMC/LFP telemetry}}
        + \lambda_{p}\,\underbrace{\tfrac1{|\Omega_p|}\sum r_s^2}_{\text{PDE}}
        + \lambda_{b}\,\underbrace{\sum_{\partial\Omega} r_{\text{BC}}^2}_{\text{BC}}
        + \lambda_{i}\,\underbrace{\tfrac1{N}\sum (T(t_0) - T_{\text{init}})^2}_{\text{IC}}

    Boundary terms included:

    ``r_inlet``   Dirichlet, ``T = T_{coolant,in}``
    ``r_outlet``  convective, ``-k dT/dn = h (T - T_{cool,out})`` (low Biot)
    ``r_wall``    convective no-slip coupling to the cells
    ``r_interface``  conjugate temperature- and flux-continuity at the bonded
                    interface between jellyroll and cold plate
    ``r_momentum`` reduced Navier-Stokes ``mu lap(u) - dp/dx``
    ``r_cont``    ``div(u) = 0`` (machine-zero by construction; retained as a
                  regression test of the ansatz)

    The ``NaN`` guard is deliberate: the second derivative of a SIREN evaluated
    through two nested ``autograd`` calls can overflow in float32 when the
    Fourier sigma anneals upward. :meth:`forward` returns a ``diagnostics``
    entry ``nonfinite`` that the auto-heal monitor watches for.
    """

    def __init__(self, model: MicrochannelPINN, config: PINNConfig = PINN_CONFIG) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.names = ("data", "pde", "bc", "initial")
        self.balancer = GradientNormBalancer(
            self.names, self._shared_parameters(), alpha=config.weight_ema
        )
        self._register_prior_losses()

    def _register_prior_losses(self) -> None:
        """Attach the static prior weights (used before the EMA kicks in)."""
        self.prior = {
            "data": self.config.w_data,
            "pde": self.config.w_physics_init,
            "bc": self.config.w_boundary,
            "initial": self.config.w_initial,
        }

    def _shared_parameters(self) -> List[nn.Parameter]:
        """Parameters receiving gradients from *every* loss term."""
        shared: List[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if name.startswith(("trunk.", "film.", "embedder.")) and p.requires_grad:
                shared.append(p)
        return shared

    # -- term builders -----------------------------------------------------
    def _collocation(self, device, dtype) -> Tensor:
        """Interior collocation points for the PDE residual.

        Returns a single tensor, not the ``(pack, channel, interface)`` triple an
        earlier version did. The triple had exactly one caller, which unpacked
        two values and then discarded the first -- so the helper advertised a
        contract nothing honoured, and the unpacking raised
        ``ValueError: too many values to unpack`` the moment it was actually
        used. Each residual samples the frame it needs: the solid energy
        equation over the pack domain, the momentum balance over
        :data:`CHANNEL_DOMAIN`, and the interface conditions over the bonded
        face. Sharing one sample across frames would evaluate the
        stream-function ansatz outside its cross-section.
        """
        return PACK_DOMAIN.sample(
            self.config.n_collocation_samples, device=device, dtype=dtype
        )

    def data_loss(self, obs_points_m: Tensor, obs_temp_c: Tensor, chemistry: str) -> Tensor:
        """Mean squared error against labelled telemetry (sensor-supervised term)."""
        pred = self.model.temperature(obs_points_m, chemistry)
        target = obs_temp_c + 273.15
        return torch.mean((pred - target) ** 2)

    def pde_loss(
        self,
        collocation: Tensor,
        chemistry: str,
        current_a: float,
        soc: float,
        dpdx: Tensor,
        aux_heat_w: float = 0.0,
    ) -> Dict[str, Tensor]:
        r"""Solid energy residual + reduced Navier-Stokes/continuity residuals."""
        from .physics import EnergyResidual, StreamFunction

        spec = get_chemistry(chemistry)
        collocation = collocation.detach().requires_grad_(True)
        T_fn = self.model.physics_callable(chemistry, "solid")

        residual = EnergyResidual(spec, T_fn, dpdx=dpdx)
        r_solid = residual.solid_residual(
            collocation, current_a=current_a, soc=soc, aux_heat_w=aux_heat_w
        )
        mse_solid = torch.mean(r_solid**2)

        # The stream-function ansatz lives in channel coordinates, so its
        # residuals must be evaluated on CHANNEL_DOMAIN samples -- not on the
        # pack-domain collocation points used for the solid energy equation.
        # Mixing the two frames would evaluate the shape function outside its
        # cross-section, where it is unbounded.
        sf = StreamFunction(dpdx)
        channel_pts = CHANNEL_DOMAIN.sample(
            min(self.config.n_boundary_samples, 256),
            device=collocation.device, dtype=collocation.dtype,
        ).requires_grad_(True)
        r_mom = sf.momentum_residual(channel_pts)
        mse_mom = torch.mean(r_mom**2)
        r_cont = sf.continuity_residual(channel_pts)
        mse_cont = torch.mean(r_cont**2)

        return {
            "solid": mse_solid,
            "momentum": mse_mom,
            "continuity": mse_cont,
            "combined": mse_solid + 0.15 * mse_mom + 0.05 * mse_cont,
        }

    def bc_loss(
        self,
        chemistry: str,
        coolant_inlet_c: float,
        flow_lpm: float,
        ambient_c: float,
        device,
        dtype,
    ) -> Dict[str, Tensor]:
        """All boundary residuals evaluated on their respective faces."""
        from ..config import COLD_PLATE
        from .physics import EnergyResidual, heat_transfer_coefficient

        spec = get_chemistry(chemistry)
        n = self.config.n_boundary_samples
        T_fn = self.model.physics_callable(chemistry, "solid")
        dpdx = torch.as_tensor(
            StreamFunction(torch.zeros((), dtype=dtype, device=device))
            .dpdx_for_flow_lpm(flow_lpm),
            device=device, dtype=dtype,
        )
        residual = EnergyResidual(spec, T_fn, dpdx=dpdx)

        # -- inlet: Dirichlet coolant condition
        inlet_pts, _ = sample_face(PACK_DOMAIN, "inlet", n, device=device, dtype=dtype)
        inlet_pts = inlet_pts.requires_grad_(True)
        r_inlet = torch.mean(
            (T_fn(inlet_pts) - (coolant_inlet_c + 273.15)) ** 2
        )

        # -- outlet: convective (low-Biot) back-flow condition
        outlet_pts, n_out = sample_face(PACK_DOMAIN, "outlet", n, device=device, dtype=dtype)
        outlet_pts = outlet_pts.requires_grad_(True)
        h_cold = heat_transfer_coefficient(flow_lpm, spec.coolant_props)
        r_outlet = torch.mean(
            residual.convective_bc_residual(
                outlet_pts, n_out, h_cold, coolant_inlet_c, air_side=False
            ) ** 2
        )

        # -- lateral shell: combined convection + solar radiation
        shell_terms = []
        for face in ("y0", "yl", "z0", "zl"):
            pts, nrm = sample_face(PACK_DOMAIN, face, max(n // 2, 64), device=device,
                                   dtype=dtype)
            pts = pts.requires_grad_(True)
            shell_terms.append(
                residual.convective_bc_residual(
                    pts, nrm, 0.0, 0.0, air_side=True, ambient_c=ambient_c
                ) ** 2
            )
        r_wall = torch.stack(shell_terms).mean()

        # -- bonded interface: temperature + flux continuity
        ifc_pts, ifc_n = sample_face(PACK_DOMAIN, "z0", n, device=device, dtype=dtype)
        ifc_pts = ifc_pts.requires_grad_(True)
        ifc = residual.interface_residual(ifc_pts, ifc_pts, ifc_n)
        r_interface = torch.mean(ifc["temperature_jump"] ** 2) + 1e-3 * torch.mean(
            ifc["flux_jump"] ** 2
        )

        # -- momentum no-slip on the four channel walls
        # Evaluated in CHANNEL coordinates: the pack's "top" face is at
        # z = 12 mm while the channel depth is 3.5 mm, so sampling the pack
        # domain here would evaluate the shape function far outside its
        # cross-section and produce a meaningless (huge, non-zero) residual.
        sf = StreamFunction(dpdx)
        noslip_terms = []
        for face in ("y0", "yl", "z0", "zl"):
            wall_pts, _ = sample_face(CHANNEL_DOMAIN, face, max(n // 2, 64),
                                      device=device, dtype=dtype)
            wall_pts = wall_pts.requires_grad_(True)
            noslip_terms.append(sf.wall_no_slip_residual(wall_pts))
        r_noslip = torch.mean(torch.cat(noslip_terms))

        return {
            "inlet": r_inlet,
            "outlet": r_outlet,
            "wall": r_wall,
            "interface": r_interface,
            "noslip": r_noslip,
            "combined": r_inlet + 0.5 * r_outlet + 0.5 * r_wall + r_interface
            + 1e-6 * r_noslip,
        }

    def initial_loss(self, points: Tensor, initial_temp_c: float, chemistry: str) -> Tensor:
        """IC loss on the initial slice; bounds the transient branch of the solution."""
        pred = self.model.temperature(points, chemistry)
        return torch.mean((pred - (initial_temp_c + 273.15)) ** 2)

    # -- composite ---------------------------------------------------------
    def forward(
        self,
        obs_points_m: Tensor,
        obs_temp_c: Tensor,
        collocation: Tensor,
        boundary_points: Tensor,
        chemistry: str,
        current_a: float,
        soc: float,
        coolant_inlet_c: float,
        flow_lpm: float,
        ambient_c: float,
        initial_temp_c: float,
        aux_heat_w: float = 0.0,
    ) -> LossBundle:
        """Assemble, balance and total every loss term."""
        device = obs_points_m.device
        dtype = obs_points_m.dtype

        # The learned dp/dx head supplies the pressure gradient the momentum
        # residual uses.  Clamped to a physically admissible band so a bad
        # optimisation step cannot drive the implied flow to zero (which would
        # silently switch the advection term off) or to an absurd value.
        dpdx = self.model.dpdx_head(
            self.model.encode(
                self.model.prepare_inputs(collocation.detach()[:1]),
                self.model.embedder(chemistry),
            )[0]
        ).mean().clamp(1.0, 5.0e5)

        l_data = self.data_loss(obs_points_m, obs_temp_c, chemistry)
        pde_terms = self.pde_loss(collocation, chemistry, current_a, soc, dpdx, aux_heat_w)
        bc_terms = self.bc_loss(chemistry, coolant_inlet_c, flow_lpm, ambient_c, device, dtype)
        l_pde = pde_terms["combined"]
        l_bc = bc_terms["combined"]
        l_ic = self.initial_loss(boundary_points, initial_temp_c, chemistry)

        losses = {"data": l_data, "pde": l_pde, "bc": l_bc, "initial": l_ic}
        weights = self.balancer.step(losses)

        total = (
            weights["data"] * l_data
            + weights["pde"] * l_pde
            + weights["bc"] * l_bc
            + weights["initial"] * l_ic
        )

        finite = bool(torch.isfinite(total))
        if not finite:
            # NaN/Inf: fall back to the static prior weighting and add a large
            # finite penalty so gradients stay well defined and the optimiser
            # walks back out of the bad basin.
            total = sum(self.prior[k] * losses[k] for k in self.names)
            total = torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)

        diagnostics = {
            f"pde_{k}": float(v.detach()) for k, v in pde_terms.items()
        }
        diagnostics.update({f"bc_{k}": float(v.detach()) for k, v in bc_terms.items()})
        diagnostics["nonfinite"] = 0.0 if finite else 1.0
        diagnostics["dpdx"] = float(dpdx.detach())

        return LossBundle(
            data=l_data, pde=l_pde, bc=l_bc, initial=l_ic, total=total,
            weights=weights, diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def chemistry_invariance_report(model: MicrochannelPINN, points: Tensor) -> Dict[str, float]:
    """Quantify how differently the two chemistries are represented.

    Returns the mean absolute temperature delta between the NMC and LPF
    evaluations at identical inputs (must be non-zero -- a zero value would mean
    the conditioning path is inert) together with the per-chemistry flow heads.
    """
    out_nmc = model.temperature(points, "nmc")
    out_lfp = model.temperature(points, "lfp")
    return {
        "mean_abs_delta_k": float((out_nmc - out_lfp).abs().mean()),
        "max_abs_delta_k": float((out_nmc - out_lfp).abs().max()),
        "nmc_mean_k": float(out_nmc.mean()),
        "lfp_mean_k": float(out_lfp.mean()),
    }


def build_model(config: PINNConfig = PINN_CONFIG, device="cpu", dtype=torch.float32) -> MicrochannelPINN:
    """Instantiate the PINN and move it to ``device``/``dtype``."""
    torch.manual_seed(config.seed)
    model = MicrochannelPINN(config)
    return model.to(device=device, dtype=dtype)


def load_checkpoint(path, device="cpu") -> MicrochannelPINN:
    """Load a trained PINN from ``path`` (falls back to an untrained model)."""
    model = MicrochannelPINN(PINN_CONFIG)
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
    return model.to(device)


def save_checkpoint(model: MicrochannelPINN, path, metadata: Optional[Mapping] = None) -> None:
    """Persist weights + metadata atomically (write to ``.tmp`` then ``os.replace``).

    ``dataclasses.asdict`` rather than ``vars`` for the config:
    :class:`PINNConfig` is a frozen dataclass declared with ``slots=True``, so
    it carries no ``__dict__`` and ``vars()`` raises ``TypeError: vars()
    argument must have __dict__ attribute``.  That call is the *last* statement
    in a training run, so the failure surfaces after the entire optimisation has
    completed and discards every epoch of work.
    """
    import dataclasses
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": dataclasses.asdict(model.config),
        "n_parameters": model.n_parameters(),
        "metadata": dict(metadata or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)