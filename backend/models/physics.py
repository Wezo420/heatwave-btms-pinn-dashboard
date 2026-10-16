r"""
backend.models.physics
======================

Conjugate-heat-transfer operators for the microchannel cold-plate.

Governing equations
-------------------
The pack is partitioned into three coupled domains:

**(S) solid jellyroll region** -- anisotropic conduction with a volumetric Joule
source:

.. math::
   \rho_s c_{p,s}\,\partial_t T_s
   = \nabla\!\cdot\!(k_s\nabla T_s) + \underbrace{\frac{I^2 R_{\text{int}}(T_s,\text{SOC})}{V_c}}_{\text{Joule}} + \dot q_\text{aux}

**(F) fluid region inside the microchannels** -- advection-diffusion:

.. math::
   \rho_f c_{p,f}\big(\partial_t T_f + \mathbf{u}\!\cdot\!\nabla T_f\big)
   = k_f\nabla^2 T_f

**(M) momentum** -- steady incompressible laminar Navier-Stokes in the channel.
Rather than solving a full 3-D NS system (which needs a dedicated CFD solver at
training time), the code uses a **separable velocity ansatz** built from the exact
plane-Poiseuille profile across the channel width and an exact cosine profile
across the depth:

.. math::
   u(y,z) &= \underbrace{\frac{\hat G}{2\mu}\,y\,(R_y - y)}_{U(y)\ \text{parabolic}}
             \cdot \underbrace{\tfrac12\Bigl[1 - \cos\!\Bigl(\tfrac{2\pi z}{R_z}\Bigr)\Bigr]}_{B(z)}
   \\
   v &= w = 0

``\hat G`` is the *learned* pressure gradient ``-dp/dx``.  Consequences that the
code exploits:

* :math:`\nabla\cdot\mathbf{u} \equiv 0` identically, because the ansatz is
  independent of ``x`` (residuals are zero to machine precision, not merely
  small).
* ``u = 0`` on all four channel walls: ``U(0) = U(R_y) = 0`` and
  ``B(0) = B(R_z) = 0``.
* The flow rate is available in closed form,
  :math:`\bar u = \hat G R_y^2 / (24\mu)`, so a commanded volumetric flow in
  L/min can be inverted to the ``\hat G`` the momentum residual expects
  (:meth:`StreamFunction.dpdx_for_flow_lpm`) and the two representations of the
  coolant state cannot drift apart.

Everything is differentiable: all residuals are produced with ``torch.autograd``
so they can be back-propagated into the trunk network.

Validity range
--------------
The Arrhenius form of :func:`internal_resistance` is saturated at
:data:`~backend.config.ARRHENIUS_CAP_C` (120 degC).  Extrapolating it to the
400 degC needed to bound a runaway would predict resistances ~200x the reference
and Joule loads two to three orders of magnitude above reality, so the relation
is deliberately clipped -- runaway detection is the fail-safe supervisor's job,
not the resistance curve's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..config import (
    ARRHENIUS_CAP_C,
    COLD_PLATE,
    EMISSIVITY,
    K_AIR_300K,
    NU_AIR_300K,
    R_GAS,
    RHO_AIR_300K,
    SIGMA_SB,
    T_REF_K,
    ChemistrySpec,
    kelvin,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Domain:
    """Axis-aligned rectangular domain in metres.

    ``origin`` and ``span`` are broadcastable tensors of shape ``(3,)`` so that
    collocation points can be generated as ``origin + span * unit_cube``.
    """

    origin: Tuple[float, float, float]
    span: Tuple[float, float, float]

    @property
    def length_m(self) -> float:
        """Domain extent along x."""
        return self.span[0]

    @property
    def width_m(self) -> float:
        """Domain extent along y."""
        return self.span[1]

    @property
    def height_m(self) -> float:
        """Domain extent along z."""
        return self.span[2]

    def sample(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw ``n`` uniform interior collocation points, shape ``(n, 3)``.

        Interior sampling is stratified via a 1-D low-discrepancy (golden-ratio)
        sequence broadcast over the three axes, which markedly reduces the
        variance of the PDE residual estimate compared with i.i.d. uniform
        sampling for a fixed collocation budget -- a standard trick in operator
        learning (see Tebaldi et al., *Comput. Math. Appl.* 2019).
        """
        j = torch.arange(n, device=device, dtype=dtype)
        # Van der Corput / golden-ratio additive recurrence.
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        u = torch.remainder(j * phi + 0.5, 1.0)
        if generator is not None:  # optional jitter for stochastic restarts
            u = torch.remainder(u + torch.rand(n, generator=generator, device=device,
                                               dtype=dtype), 1.0)
        pts = u.unsqueeze(1).repeat(1, 3)
        # Decorrelate the axes by a Cranley-Patterson rotation.
        rot = torch.tensor([0.3178, 0.7351, 0.1913], device=device, dtype=dtype)
        pts = torch.remainder(pts + rot.unsqueeze(0), 1.0)
        span = torch.tensor(self.span, device=device, dtype=dtype)
        origin = torch.tensor(self.origin, device=device, dtype=dtype)
        return origin + span * pts

    def grid(self, nx: int, ny: int, nz: int, device="cpu", dtype=torch.float64) -> Tensor:
        """Tensor-product grid of shape ``(nx, ny, nz, 3)`` on the domain."""
        xs = torch.linspace(0, self.length_m, nx, device=device, dtype=dtype)
        ys = torch.linspace(0, self.width_m, ny, device=device, dtype=dtype)
        zs = torch.linspace(0, self.height_m, nz, device=device, dtype=dtype)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1)


PACK_DOMAIN = Domain(
    origin=(0.0, 0.0, 0.0),
    span=(COLD_PLATE.length_m, COLD_PLATE.width_m, COLD_PLATE.thickness_m),
)

FLUID_DOMAIN = Domain(
    origin=(0.0, 0.0, 0.0),
    span=(COLD_PLATE.length_m, COLD_PLATE.width_m, COLD_PLATE.thickness_m),
)

#: Channel-local domain.  The stream-function ansatz is defined in
#: \emph{channel} coordinates -- one channel's cross-section is
#: ``[0, channel_width] x [0, channel_depth]`` -- whereas :data:`PACK_DOMAIN`
#: spans the whole plate.  Sampling the pack faces and evaluating the wall
#: residual against them is therefore incorrect (the pack's ``yl``/``zl`` faces
#: lie far outside the channel cross-section, where the shape function is
#: unbounded).  This domain is the correct frame for every no-slip and momentum
#: residual evaluation.
CHANNEL_DOMAIN = Domain(
    origin=(0.0, 0.0, 0.0),
    span=(COLD_PLATE.length_m, COLD_PLATE.channel_width_m, COLD_PLATE.channel_depth_m),
)


# ---------------------------------------------------------------------------
# Stream-function velocity field (Navier-Stokes ansatz)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamFunction:
    r"""Exact 3-D velocity field for fully-developed laminar microchannel flow.

    The ansatz is a **separable product of the exact plane-Poiseuille profile
    across the channel width and an exact cosine profile across the depth**::

        u(y, z) = \underbrace{\frac{\hat G}{2\mu}\, y\,(R_y - y)}_{U(y)\ \text{(parabolic, in } y)}
                  \cdot \underbrace{\tfrac12\!\left[1 - \cos\!\left(\tfrac{2\pi z}{R_z}\right)\right]}_{B(z)\ \text{(cosine, in } z)}

    with :math:`v = w = 0`.  Three properties are exact rather than
    approximate:

    * **No-slip on all four walls.**  ``U(0) = U(R_y) = 0`` and ``B(0) = B(R_z) = 0``.
    * **Incompressibility by construction.**  ``\mathbf{u}`` does not depend on
      ``x``, so :math:`\nabla\cdot\mathbf{u} \equiv 0` to machine precision.
    * **Analytic flow rate.**  Integrating the profile gives the closed form

      .. math::
        \bar{u} = \frac{1}{R_y R_z}\!\int\!\!\int u\,\mathrm{d}y\,\mathrm{d}z
                = \frac{\hat G R_y^2}{24\,\mu}

      since :math:`\int_0^{R_y} y(R_y - y)\,\mathrm{d}y = R_y^3/6` and
      :math:`\int_0^{R_z} B\,\mathrm{d}z = R_z/2`.

    A stream-function representation is *not* used for ``\psi`` itself, because
    the natural separable candidate :math:`\psi \propto (y/R_y)^2(1-y/R_y)^2`
    has an identically vanishing ``y``-derivative integral and would therefore
    predict **zero net flux** -- a physically wrong channel.  The profile is
    specified directly instead, which is both simpler and correct.

    Parameters
    ----------
    dpdx:
        Learned pressure-gradient magnitude ``\hat G = -dp/dx`` in Pa/m.  This
        single scalar is the neural *inverse problem* unknown: the PINN recovers
        it from the requirement that the temperature field satisfy both the
        fluid advection equation and the convective boundary condition.
    ry, rz:
        Full ``y``-width and full ``z``-depth of one channel cross-section, m.
        The profile is defined for ``y in [0, ry]`` and ``z in [0, rz]`` --
        **channel** coordinates, matching :data:`CHANNEL_DOMAIN`.  Evaluating a
        momentum or no-slip residual against :data:`PACK_DOMAIN` faces would put
        the sample points outside the cross-section, where the profile is
        unbounded and the residual is meaningless.
    """

    dpdx: Tensor
    ry: float = COLD_PLATE.channel_width_m
    rz: float = COLD_PLATE.channel_depth_m

    # -- shape --------------------------------------------------------------
    def poiseuille_y(self, y: Tensor) -> Tensor:
        """Plane-Poiseuille profile ``U(y) = G y (R_y - y) / (2 mu)``.

        Zero at both side walls; parabolic with a maximum at ``y = R_y/2``.
        """
        return (self.dpdx / (2.0 * self._mu)) * y * (self.ry - y)

    def cosine_z(self, z: Tensor) -> Tensor:
        """Cross-depth profile ``B(z) = 0.5 (1 - cos(2 pi z / R_z))``; zero at walls."""
        return 0.5 * (1.0 - torch.cos(2.0 * math.pi * (z / self.rz)))

    def stream_function(self, y: Tensor, z: Tensor) -> Tensor:
        r"""Stream function ``\psi(y, z)`` with ``u = \partial_y \psi``.

        Obtained by integrating the profile in ``y``.  ``\psi`` is *not*
        constant across the section -- ``\psi(R_y) - \psi(0)`` equals the flux
        integral -- which is precisely the non-zero flux condition a channel
        requires.
        """
        return (self.dpdx / (2.0 * self._mu)) * \
            (self.ry * y**2 / 2.0 - y**3 / 3.0) * self.cosine_z(z)

    def shape_factor(self, y: Tensor, z: Tensor) -> Tensor:
        """The velocity profile itself, ``u(y, z)`` (kept for call-site clarity)."""
        return self.poiseuille_y(y) * self.cosine_z(z)

    def velocity(self, xyz: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Analytic velocity ``(u, v, w)`` at points ``xyz`` of shape ``(..., 3)``.

        ``v`` and ``w`` are identically zero, but are constructed as
        ``0 * xyz[..., k]`` rather than ``torch.zeros_like(...)`` so that they
        remain attached to the autograd graph.  Without this, the continuity
        residual raises ``element 0 of tensors does not require grad`` on its
        first ``autograd.grad`` call, because a detached zero has no ``grad_fn``
        -- even though the *mathematical* divergence is zero.  The
        zero-times-input form is exact and differentiable.
        """
        u = self.shape_factor(xyz[..., 1], xyz[..., 2])
        v = torch.zeros_like(xyz[..., 0]) * xyz[..., 0]
        w = torch.zeros_like(xyz[..., 1]) * xyz[..., 1]
        return u, v, w

    # -- flow-rate analytics ------------------------------------------------
    def analytic_mean_velocity(self) -> Tensor:
        r"""Exact mean velocity :math:`\bar u = \hat G R_y^2 / (24 \mu)`, m/s.

        Derived in the class docstring from
        :math:`\int_0^{R_y} y(R_y-y)\,\mathrm{d}y = R_y^3/6` and
        :math:`\int_0^{R_z} B\,\mathrm{d}z = R_z/2`, with the ``1/(2\mu)``
        prefactor of the profile.
        """
        return (self.dpdx * self.ry**2) / (24.0 * self._mu)

    @property
    def _mu(self) -> float:
        """Dynamic viscosity of the glycol coolant, Pa*s."""
        from ..config import CHEMISTRY_REGISTRY

        return CHEMISTRY_REGISTRY["nmc"].coolant_props["mu"]

    @property
    def _rho(self) -> float:
        """Coolant density, kg/m^3."""
        from ..config import CHEMISTRY_REGISTRY

        return CHEMISTRY_REGISTRY["nmc"].coolant_props["density"]

    def channel_area_m2(self) -> float:
        """Cross-sectional area of a single channel, m^2."""
        return self.ry * self.rz

    def mass_flow_kg_s(self, n_channels: int = COLD_PLATE.n_channels) -> Tensor:
        """Total coolant mass flow through the channel bundle, kg/s."""
        return self._rho * self.channel_area_m2() * n_channels * \
            self.analytic_mean_velocity()

    def volume_flow_lpm(self, n_channels: int = COLD_PLATE.n_channels) -> Tensor:
        """Total coolant volumetric flow through the bundle, L/min."""
        area = self.channel_area_m2() * n_channels
        # U [m/s] * A [m^2] = m^3/s ; * 60000 -> L/min ; * 1000 -> L
        return self.analytic_mean_velocity() * area * 6.0e7 / 1000.0

    def reynolds(self, n_channels: int = COLD_PLATE.n_channels) -> Tensor:
        """Bundle Reynolds number ``rho U D_h / mu`` (diagnostic only)."""
        dh = COLD_PLATE.hydraulic_diameter_m
        return self._rho * self.analytic_mean_velocity() * dh / self._mu

    def dpdx_for_flow_lpm(
        self, flow_lpm: float, n_channels: int = COLD_PLATE.n_channels
    ) -> float:
        r"""Invert :meth:`analytic_mean_velocity` for the pressure gradient.

        Used to translate a commanded volumetric flow into the ``\hat G`` the
        momentum residual expects, so the two representations of the coolant
        state cannot drift apart.
        """
        area = self.channel_area_m2() * n_channels
        u_mean = (flow_lpm * 1000.0 / 6.0e7) / max(area, 1e-12)
        return float(u_mean * 24.0 * self._mu / self.ry**2)

    # -- derivative residuals via autodiff ---------------------------------
    def laplacian_u(self, xyz: Tensor) -> Tensor:
        r"""Autodiff :math:`\nabla^2 u` of the analytic velocity field.

        Evaluated through the graph so it participates in the PINN loss: the
        learned ``\hat G`` receives gradient through this term.
        """
        xyz = xyz.requires_grad_(True)
        u, _, _ = self.velocity(xyz)
        grad_u = torch.autograd.grad(u.sum(), xyz, create_graph=True,
                                     retain_graph=True)[0]
        lap = torch.zeros_like(u)
        for i in range(3):
            second = torch.autograd.grad(
                grad_u[..., i].sum(), xyz, create_graph=True, retain_graph=True
            )[0][..., i]
            lap = lap + second
        return lap

    def momentum_residual(self, xyz: Tensor, mu: float | Tensor | None = None) -> Tensor:
        r"""Reduced x-momentum residual, **normalised to a dimensionless form**.

        .. math:: r_u = \frac{\mu\,\nabla^2 u - \hat G}{\max(|\hat G|,\ 1)}

        The convective term vanishes identically (the ansatz is fully
        developed), so the residual is the balance between viscous diffusion
        and the imposed pressure gradient.

        Normalisation matters and is not cosmetic.  ``\hat G`` is O(\num{e2}--\num{e4})
        Pa/m for a microchannel, so the *unnormalised* residual is O(\num{e2}--\num{e4})
        and its square enters the loss at O(\num{e4}--\num{e8}) -- seven orders
        of magnitude above every other term, which would make the gradient-norm
        balancer assign it essentially all the weight and destroy the remaining
        terms.  The normalised residual is O(1) and composes with the energy
        residual on a comparable scale.  For an exactly Poiseuille profile in the
        wide-channel limit this residual goes to zero; the separable cosine in
        ``z`` means it does not vanish exactly, which is reported honestly
        rather than hidden by a smaller weight.
        """
        mu_t = self._mu if mu is None else mu
        raw = mu_t * self.laplacian_u(xyz) - self.dpdx
        scale = torch.clamp(torch.abs(self.dpdx), min=1.0)
        return raw / scale

    def continuity_residual(self, xyz: Tensor) -> Tensor:
        r"""``\nabla\cdot u`` evaluated by autodiff (machine-zero by construction)."""
        xyz = xyz.requires_grad_(True)
        u, v, w = self.velocity(xyz)
        terms = []
        for comp in (u, v, w):
            g = torch.autograd.grad(comp.sum(), xyz, create_graph=True,
                                    retain_graph=True)[0]
            terms.append(g)
        return terms[0][..., 0] + terms[1][..., 1] + terms[2][..., 2]

    def wall_no_slip_residual(self, xyz: Tensor) -> Tensor:
        r"""``u^2 + v^2 + w^2`` on the channel walls (identically zero there)."""
        u, v, w = self.velocity(xyz)
        return u**2 + v**2 + w**2


# ---------------------------------------------------------------------------
# Boundary-condition data generation
# ---------------------------------------------------------------------------


def sample_face(
    domain: Domain,
    face: str,
    n: int,
    device="cpu",
    dtype=torch.float64,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Sample ``n`` points on a named boundary face.

    Returns
    -------
    (points, outward_normal)
        ``points`` has shape ``(n, 3)``; ``normal`` has shape ``(3,)``.
    """
    if face in {"inlet", "x0"}:
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        v = torch.rand(n, generator=generator, device=device, dtype=dtype)
        pts = torch.stack(
            [
                torch.zeros_like(u),
                v * domain.width_m,
                u * domain.height_m,
            ],
            dim=-1,
        )
        nrm = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype)
    elif face in {"outlet", "xl"}:
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        v = torch.rand(n, generator=generator, device=device, dtype=dtype)
        pts = torch.stack(
            [
                torch.full_like(u, domain.length_m),
                v * domain.width_m,
                u * domain.height_m,
            ],
            dim=-1,
        )
        nrm = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    elif face in {"y0", "y_bottom"}:
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        v = torch.rand(n, generator=generator, device=device, dtype=dtype)
        pts = torch.stack(
            [
                u * domain.length_m,
                torch.zeros_like(v),
                v * domain.height_m,
            ],
            dim=-1,
        )
        nrm = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    elif face in {"yl", "y_top"}:
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        v = torch.rand(n, generator=generator, device=device, dtype=dtype)
        pts = torch.stack(
            [
                u * domain.length_m,
                torch.full_like(v, domain.width_m),
                v * domain.height_m,
            ],
            dim=-1,
        )
        nrm = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    elif face in {"z0", "bottom"}:
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        v = torch.rand(n, generator=generator, device=device, dtype=dtype)
        pts = torch.stack(
            [
                u * domain.length_m,
                v * domain.width_m,
                torch.zeros_like(u),
            ],
            dim=-1,
        )
        nrm = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
    elif face in {"zl", "top"}:
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        v = torch.rand(n, generator=generator, device=device, dtype=dtype)
        pts = torch.stack(
            [
                u * domain.length_m,
                v * domain.width_m,
                torch.full_like(u, domain.height_m),
            ],
            dim=-1,
        )
        nrm = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown face {face!r}")
    return pts, nrm


# ---------------------------------------------------------------------------
# Constitutive relations
# ---------------------------------------------------------------------------


def internal_resistance(
    spec: ChemistrySpec,
    temp_c: Tensor | float,
    soc: Tensor | float,
) -> Tensor:
    """Arrhenius + SOC corrected internal resistance (vectorised), in ohm.

    Accepts scalars or tensors of any broadcastable shape.  Scalars are promoted
    to ``float64`` so that a scalar call and a tensor call return the same value
    -- ``torch.clamp`` has no scalar overload, so promoting here rather than at
    every call site keeps the constitutive relations callable from plain Python.
    """
    if torch.is_tensor(temp_c):
        dtype = temp_c.dtype
        t_raw = temp_c + 273.15
    else:
        dtype = torch.float64
        t_raw = torch.tensor(float(temp_c) + 273.15, dtype=torch.float64)
    # Saturate the Arrhenius argument at the fit's validity bound (see
    # ChemistrySpec.r_int); without this the exponential diverges above ~400 degC
    # and the reduced-order model has no bounded equilibrium.
    t_k = torch.clamp(t_raw, min=200.0, max=kelvin(ARRHENIUS_CAP_C))
    arrhenius = torch.exp((spec.ea / R_GAS) * (1.0 / T_REF_K - 1.0 / t_k))
    soc_t = soc if torch.is_tensor(soc) else torch.tensor(soc, dtype=dtype)
    soc_term = 1.0 + 0.35 * (1.0 - 2.0 * torch.clamp(soc_t, 0.0, 1.0)) ** 2
    return spec.r_int_ref * arrhenius * soc_term


def joule_source(
    spec: ChemistrySpec,
    current_a: Tensor | float,
    temp_c: Tensor | float,
    soc: Tensor | float,
) -> Tensor:
    r"""Volumetric Joule heating :math:`\dot q''' = I^2 R_{int} / V_{cell}` [W/m^3]."""
    r_int = internal_resistance(spec, temp_c, soc)
    from ..config import CELL_GEOMETRY

    return (current_a**2) * r_int / CELL_GEOMETRY.volume_m3


def external_convection_coeff(
    temp_c: Tensor | float,
    wind_mps: Tensor | float = 2.5,
    length_scale_m: float = COLD_PLATE.length_m,
) -> Tensor:
    r"""Combined natural + forced convection coefficient of the pack shell, W/(m^2*K).

    Natural: ``Nu = 0.10 Ra^{1/3}`` (laminar vertical plate, Churchill-Chu).
    Forced:  ``Nu = 0.664 Re^{1/2} Pr^{1/3}`` (flat plate, laminar).

    Accepts scalars or tensors of broadcastable shape; scalars are promoted to
    ``float64`` so the constitutive relations stay callable from plain Python
    (``torch.clamp`` has no scalar overload).
    """
    if torch.is_tensor(temp_c):
        t_k = torch.clamp(temp_c + 273.15, min=200.0, max=400.0)
        dtype = temp_c.dtype
    else:
        t_k = torch.tensor(float(temp_c) + 273.15, dtype=torch.float64)
        dtype = torch.float64
    length = torch.as_tensor(float(length_scale_m), dtype=dtype)
    wind = torch.as_tensor(float(wind_mps), dtype=dtype)

    dt = torch.clamp(t_k - 298.15, min=1e-3, max=200.0)
    beta = 1.0 / t_k
    ra = R_GAS * 9.81 * beta * dt * length**3 / (NU_AIR_300K**2)
    nu_nat = 0.10 * ra ** (1.0 / 3.0)
    re = torch.clamp(wind * length / NU_AIR_300K, min=1e-3)
    nu_forced = 0.664 * re**0.5 * 0.71 ** (1.0 / 3.0)
    nu = torch.clamp(nu_nat + nu_forced, min=1e-2, max=1e4)
    return nu * K_AIR_300K / length


def radiation_flux_correction(temp_c: Tensor | float, solar_w_m2: Tensor | float = 800.0) -> Tensor:
    r"""Absorbed solar + net radiative heat flux on the shell, W/m^2.

    ``alpha * G_solar`` is absorbed; the emitted long-wave term is linearised
    about the ambient so that the BC remains a first-order condition (avoiding a
    fourth-order temperature term inside an autodiff graph).
    """
    if torch.is_tensor(temp_c):
        t_k = torch.clamp(temp_c + 273.15, min=200.0, max=450.0)
        dtype = temp_c.dtype
    else:
        t_k = torch.tensor(float(temp_c) + 273.15, dtype=torch.float64)
        dtype = torch.float64
    absorbed = 0.85 * torch.as_tensor(float(solar_w_m2), dtype=dtype)
    emitted = EMISSIVITY * SIGMA_SB * t_k**4
    ambient = EMISSIVITY * SIGMA_SB * (298.15**4)
    return absorbed + 0.6 * (emitted - ambient)


def nusselt_microchannel(
    reynolds: Tensor | float,
    prandtl: float = 13.8,
    aspect_ratio: float = COLD_PLATE.channel_depth_m / COLD_PLATE.channel_width_m,
    entrance_effect: bool = True,
) -> Tensor:
    r"""Hydraulic-diameter Nusselt number for a rectangular duct.

    Structure of the model
    ----------------------
    A rectangular duct has a **fully developed** Nusselt number that depends
    *only* on the aspect ratio, and a **developing** enhancement that depends
    on ``Re`` and ``Pr``.  Both parts are needed and they must not be confused:

    1.  *Fully developed, constant heat flux* (Shah & London 1978).  Written
        with the aspect ratio normalised to :math:`a = \min(W,H)/\max(W,H) \le 1`:

        .. math::
           Nu_{fd}(a) = 8.235\bigl(1 - 2.0421a + 2.0732a^2 - 0.6277a^3
                                  - 1.596a^4 + 1.669a^5\bigr)

        The bracket is 1 at ``a -> 0`` (parallel plates, ``Nu = 8.235``, twice the
        textbook 3.66 for a wide channel because this is the constant-*flux*
        case) and 0.476 at ``a = 1`` (square duct), giving ``Nu = 3.92`` -- the
        accepted value.  ``Nu_{fd}`` carries **no** Reynolds dependence;
        attaching one to it is the single most common error in microchannel
        correlations.

    2.  *Developing (entrance) enhancement*.  A Hausen-type correction with the
        Graetz number ``Gz = Re Pr D_h / L``:

        .. math:: Nu = Nu_{fd}\left[1 + \frac{0.0668\,Gz}{1 + 0.04\,Gz^{2/3}}\right]

        This is what makes ``h`` Reynolds-dependent, and it saturates towards
        ``1 + 0.0668/0.04 = 2.67`` for ``Gz >> 1``.

    Passing ``entrance_effect=False`` returns ``Nu_{fd}`` alone, which is the
    right choice for a long cold plate whose channels are already thermally
    developed over most of their length.

    Consequence for the reduced model: Note that ``a`` is normalised to ``min(W,H)/max(W,H)``, so the correlation is
    symmetric under ``a -> 1/a``: the same duct described either way round gives
    the same Nusselt number.

    At ``Re = 750``, ``Pr = 13.8``, ``D_h = 2.7 mm``, ``L = 360 mm`` one gets
    ``Gz ~ 55`` and a moderate enhancement.  Note that this construction yields
    ``h`` of order \SIrange{1}{3}{e3}{e4}~\si{\watt\per\metre\squared\per\kelvin},
    which is the correct regime for a 2.2 mm microchannel -- a quadratic-in-``Re``
    form would give \num{e5}--\num{e6}, roughly two orders of magnitude too large.
    """
    a = torch.as_tensor(
        float(min(abs(aspect_ratio), 1.0 / max(abs(aspect_ratio), 1e-12))),
        dtype=torch.float64,
    ).clamp(1.0e-3, 1.0)

    # Shah & London, constant heat flux. The polynomial alone is not the Nusselt
    # number -- it must be divided by the aspect ratio. At a = 1 the bracket is
    # 0.4764 and 8.235 * 0.4764 / 1 = 3.92, the accepted square-duct value;
    # omitting the /a gives 3.92 *before* the 8.235 factor and 3.06 after, which
    # is 22% low.
    bracket = (1.0 - 2.0421 * a + 2.0732 * a**2 - 0.6277 * a**3
               - 1.596 * a**4 + 1.669 * a**5).clamp(min=0.05, max=1.0)
    nu_fd = 8.235 * bracket / a

    if not entrance_effect:
        return nu_fd

    re = torch.clamp(torch.as_tensor(reynolds, dtype=torch.float64),
                     min=20.0, max=20000.0)
    length = max(COLD_PLATE.length_m, 1e-3)
    gz = re * prandtl * COLD_PLATE.hydraulic_diameter_m / length
    enhancement = 1.0 + (0.0668 * gz) / (1.0 + 0.04 * torch.clamp(gz, min=1e-6) ** (2.0 / 3.0))
    return nu_fd * enhancement


def heat_transfer_coefficient(flow_lpm: float, coolant: Dict[str, float]) -> float:
    """Cold-plate-side convection coefficient ``h`` in W/(m^2*K) at a given flow."""
    rho, mu, k, pr = (
        coolant["density"], coolant["mu"], coolant["k"], coolant["prandtl"]
    )
    q_m3_s = flow_lpm / 60000.0
    area = COLD_PLATE.fluid_area_m2
    if area <= 0 or q_m3_s <= 0:
        return 0.0
    u_mean = q_m3_s / area
    re = rho * u_mean * COLD_PLATE.hydraulic_diameter_m / mu
    nu = nusselt_microchannel(re, pr)
    return float(nu) * k / COLD_PLATE.hydraulic_diameter_m


# ---------------------------------------------------------------------------
# Differentiable residual operators for the energy equation
# ---------------------------------------------------------------------------


class EnergyResidual:
    r"""Residual factory for the solid and fluid energy equations.

    The temperature field is supplied as a **callable** ``T(xyz) -> Tensor`` so the
    class is agnostic to the network architecture; gradients are taken with
    ``create_graph=True`` so that the residual itself is differentiable w.r.t.
    the network parameters (the defining property of a PINN).

    Parameters
    ----------
    spec:
        Chemistry providing ``rho``, ``cp``, ``k`` and ``R_int``.
    solid_temperature / fluid_temperature:
        Callables for the two fields.  In the full conjugate model both are the
        *same* network evaluated in its two phase-conditional branches; in the
        reduced model only the solid callable is provided.
    dpdx:
        Learned pressure gradient (drives ``u``).
    time_derivative:
        Optional callable ``T_dot(xyz) -> Tensor`` for ``\partial T / \partial t``.
        When ``None`` the quasi-steady formulation ``\partial_t T = 0`` is used.
    """

    def __init__(
        self,
        spec: ChemistrySpec,
        solid_temperature: Callable[[Tensor], Tensor],
        fluid_temperature: Optional[Callable[[Tensor], Tensor]] = None,
        dpdx: Optional[Tensor] = None,
        time_derivative: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.spec = spec
        self.Ts = solid_temperature
        self.Tf = fluid_temperature
        self.dpdx = dpdx
        self.T_dot = time_derivative
        self.cp_s = spec.cp
        self.rho_s = spec.density
        self.k_s = spec.k_solid

    # -- differential operators -------------------------------------------
    @staticmethod
    def _grad(fn: Callable[[Tensor], Tensor], xyz: Tensor, n_out: int = 1) -> Tensor:
        """Return the gradient tensor of ``fn`` at ``xyz``, shape ``(n, 3)``.

        ``grad_outputs`` must be supplied because ``fn`` is evaluated at a *batch*
        of points and therefore returns a vector, not a scalar.  ``autograd.grad``
        only infers the seed for a scalar output; without an explicit seed it
        raises "grad can be implicitly created only for scalar outputs".  An
        all-ones seed is the per-sample derivative: it is equivalent to
        differentiating ``fn(x).sum()`` but keeps the shape of ``val`` intact for
        the second-derivative pass below.
        """
        xyz = xyz.requires_grad_(True)
        val = fn(xyz)
        # A field with no dependence on the coordinates at all (an exactly
        # uniform temperature) produces a value with no ``grad_fn``. Its
        # gradient is identically zero, so short-circuit rather than ask
        # autograd to differentiate a constant.
        if not val.requires_grad:
            return torch.zeros_like(xyz)
        seed = torch.ones_like(val)
        return torch.autograd.grad(
            val, xyz, grad_outputs=seed, create_graph=True, retain_graph=True
        )[0]

    @staticmethod
    def _laplacian(fn: Callable[[Tensor], Tensor], xyz: Tensor) -> Tensor:
        """Return the Laplacian ``\\nabla^2 f`` at ``xyz``, shape ``(n,)``."""
        xyz = xyz.requires_grad_(True)
        val = fn(xyz)
        if not val.requires_grad:
            return torch.zeros_like(val)   # uniform field: exactly zero curvature
        seed = torch.ones_like(val)
        grad = torch.autograd.grad(
            val, xyz, grad_outputs=seed, create_graph=True, retain_graph=True
        )[0]
        lap = torch.zeros_like(val)
        for i in range(3):
            # Differentiate the i-th gradient component again.  Each pass must
            # keep its own graph alive (``retain_graph``) because the next axis
            # reuses the same intermediate.
            #
            # ``allow_unused`` is not optional robustness here.  If the field is
            # *affine* along axis i -- which a linear profile, or any trained PINN
            # whose second derivative underflows to a constant, will be -- then
            # ``grad[..., i]`` has no dependency on ``xyz`` at all and autograd
            # raises "element 0 of tensors does not require grad".  The
            # mathematically correct second derivative is zero, so materialise
            # zeros for the unused axes instead of propagating an exception out
            # of a residual evaluation.
            gi = grad[..., i]
            # ``allow_unused`` only covers the case where autograd *finds* the
            # input but no path exists. When the component is outright constant
            # the tensor carries no ``grad_fn`` at all, and autograd rejects it
            # before consulting ``allow_unused``. The correct second derivative
            # is zero either way.
            if not gi.requires_grad:
                continue
            second = torch.autograd.grad(
                gi, xyz,
                grad_outputs=torch.ones_like(gi),
                create_graph=True, retain_graph=True, allow_unused=True,
            )[0]
            if second is None:
                continue
            lap = lap + second[..., i]
        return lap

    # -- residual terms ----------------------------------------------------
    def solid_residual(
        self,
        xyz: Tensor,
        current_a: Tensor | float,
        soc: Tensor | float,
        aux_heat_w: float = 0.0,
    ) -> Tensor:
        r"""Solid-phase energy residual ``r_s`` [W/m^3].

        .. math::
           r_s = \rho_s c_{p,s}\,\partial_t T_s - \nabla\cdot(k_s\nabla T_s)
                 - \dot q'''_\text{Joule} - \dot q'''_\text{aux}

        ``\dot q'''_aux`` converts the packaged auxiliary load (W) into a
        volumetric term using the cell envelope volume.
        """
        Ts = self.Ts(xyz)
        t_dot = self.T_dot(xyz) if self.T_dot is not None else torch.zeros_like(Ts)
        lap = self._laplacian(self.Ts, xyz)
        q_joule = joule_source(self.spec, current_a, Ts, soc)
        from ..config import CELL_GEOMETRY, COLD_PLATE

        pack_volume = COLD_PLATE.length_m * COLD_PLATE.width_m * (
            COLD_PLATE.thickness_m + 0.028
        )
        q_aux = aux_heat_w / pack_volume
        return self.rho_s * self.cp_s * t_dot - self.k_s * lap - q_joule - q_aux

    def fluid_residual(
        self,
        xyz: Tensor,
        flow_lpm: float,
        coolant: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        r"""Fluid-phase advection-diffusion residual ``r_f`` [W/m^3].

        .. math::
           r_f = \rho_f c_{p,f}\big(\partial_t T_f + \mathbf{u}\cdot\nabla T_f\big)
                 - k_f\nabla^2 T_f
        """
        if self.Tf is None:
            return torch.zeros(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
        coolant = coolant or self.spec.coolant_props
        sf = StreamFunction(self.dpdx if self.dpdx is not None else torch.tensor(0.0,
                             device=xyz.device, dtype=xyz.dtype))
        Tfl = self.Tf(xyz)
        t_dot = self.T_dot(xyz) if self.T_dot is not None else torch.zeros_like(Tfl)
        grad_T = self._grad(self.Tf, xyz)
        lap_T = self._laplacian(self.Tf, xyz)
        u, v, w = sf.velocity(xyz)
        adv = u * grad_T[..., 0] + v * grad_T[..., 1] + w * grad_T[..., 2]
        rho, cp, k = coolant["density"], coolant["cp"], coolant["k"]
        return rho * cp * (t_dot + adv) - k * lap_T

    def interface_residual(
        self,
        xyz_solid: Tensor,
        xyz_fluid: Tensor,
        normal: Tensor,
    ) -> Dict[str, Tensor]:
        r"""Conjugate interface residuals (temperature and flux continuity).

        Returns a dict with ``temperature_jump`` [K] and ``flux_jump`` [W/m^2].
        """
        if self.Tf is None:
            zero = torch.zeros(xyz_solid.shape[0], device=xyz_solid.device,
                               dtype=xyz_solid.dtype)
            return {"temperature_jump": zero, "flux_jump": zero}
        Ts = self.Ts(xyz_solid)
        Tf = self.Tf(xyz_fluid)
        g_s = self._grad(self.Ts, xyz_solid)
        g_f = self._grad(self.Tf, xyz_fluid)
        q_s = -self.k_s * (g_s * normal).sum(-1)
        q_f = -self.spec.coolant_props["k"] * (g_f * normal).sum(-1)
        return {"temperature_jump": Ts - Tf, "flux_jump": q_s - q_f}

    def convective_bc_residual(
        self,
        xyz: Tensor,
        normal: Tensor,
        h: Tensor | float,
        coolant_bulk_c: Tensor | float,
        air_side: bool = False,
        ambient_c: float = 30.0,
        solar_w_m2: float = 800.0,
    ) -> Tensor:
        r"""Convective boundary residual ``-k\,\partial_n T - h (T_s - T_\infty) - q_rad``."""
        T = self.Ts(xyz)
        g = self._grad(self.Ts, xyz)
        dn = (g * normal).sum(-1)
        if air_side:
            h_eff = external_convection_coeff(T, length_scale_m=COLD_PLATE.length_m)
            flux_rad = radiation_flux_correction(T, solar_w_m2)
            bulk = ambient_c + 273.15
            return -self.k_s * dn - h_eff * (T - bulk) - flux_rad
        return -self.k_s * dn - h * (T - (coolant_bulk_c + 273.15))


# ---------------------------------------------------------------------------
# Reduced-order lumped model (used by the telemetry simulator and the MPC)
# ---------------------------------------------------------------------------


def lumped_pack_rhs(
    spec: ChemistrySpec,
    cell_temp_c: float,
    ambient_c: float,
    road_surface_c: float,
    solar_w_m2: float,
    current_a: float,
    soc: float,
    flow_lpm: float,
    coolant_inlet_c: float,
    c_th_per_cell: float | None = None,
    aux_heat_w: float = 0.0,
) -> float:
    r"""``dT_cell/dt`` for one thermal zone of the reduced 12-zone pack, degC/s.

    Energy balance on a zone:

    .. math::
       C_\theta\,\dot T
       = \underbrace{n_s (I/n_p)^2 R_{int}(T,\text{SOC})}_{\text{Joule}}
       - \underbrace{U_\text{in} (T - T_\text{cool, in})}_{\text{microchannel}}
       - \underbrace{U_\text{out} (T - T_\text{amb})}_{\text{shell + road}}
       - \underbrace{\varepsilon\sigma A (T^4 - T_\text{amb}^4)}_{\text{radiation}}

    Three conventions, each of which changes the answer by an order of magnitude
    or more if got wrong, so each is stated explicitly.

    **Current scaling.**  ``current_a`` is the *pack* current.  A zone is
    ``n_s x n_p`` cells (2s4p by default) and the pack current divides among the
    ``n_p`` parallel strings.  Applying the pack current to a single cell's
    resistance over-predicts the Joule load by ``n_p^2`` (16x here) and produces
    an equilibrium that no achievable flow can reach.

    **Thermal mass.**  ``C_\theta`` defaults to the zone's real sensible heat
    capacity ``rho c_p V_zone`` -- about 5.9 kJ/K per zone.  A single bare cell
    is ~365 J/K; using that 16x too small compresses every transient and
    destroys the anticipatory lead time that is the point of the controller.

    **Cooling area.**  ``U_in`` is the wall-film conductance times the bundle's
    *wetted* perimeter x length, allocated per zone.  Using the flow
    cross-section instead understates the cooling area by ~3 orders of magnitude
    and makes the pack temperature insensitive to flow -- the controller's
    actuator would have no authority.

    The microchannel conductance is additionally reduced in parallel by the
    coolant's transport capacity ``\dot m c_p``, which bounds the achievable
    duty cycle.
    """
    from ..config import (
        CELL_GEOMETRY,
        COLD_PLATE,
        ZONE_PARALLEL_STRINGS,
        ZONE_SERIES_CELLS,
    )

    n_s, n_p = ZONE_SERIES_CELLS, ZONE_PARALLEL_STRINGS
    if c_th_per_cell is None:
        c_th_per_cell = CELL_GEOMETRY.zone_volume_m3 * spec.density * spec.cp

    r_int = float(internal_resistance(spec, cell_temp_c, soc))
    i_string = current_a / max(n_p, 1)
    q_joule = (i_string**2) * n_s * r_int
    # Auxiliary load (fast-charge heaters, BMS, cabin conditioning fed from the
    # pack) is distributed across the zones.
    q_aux = aux_heat_w / max(COLD_PLATE.n_cells, 1)

    coolant = spec.coolant_props
    area_flux = COLD_PLATE.fluid_area_m2
    u_mean = (flow_lpm / 60000.0) / area_flux if area_flux > 0 else 0.0
    re = coolant["density"] * u_mean * COLD_PLATE.hydraulic_diameter_m / coolant["mu"]
    h_in = float(nusselt_microchannel(re, coolant["prandtl"])) * coolant["k"] / \
        COLD_PLATE.hydraulic_diameter_m

    # Wetted heat-transfer area of the channel bundle: perimeter x length.
    # This -- not the flow cross-section -- is the area over which the
    # Nusselt-number-derived h acts. Using the cross-section here would
    # understate the cooling area by three orders of magnitude and make the
    # coolant term negligible, which in turn makes the pack temperature
    # insensitive to flow (the controller's actuator would have no authority).
    a_wet = COLD_PLATE.wetted_perimeter_m * COLD_PLATE.length_m

    # Per-cell allocation. The bundle's wetted area is shared by every cell in
    # the pack, so a single-cell energy balance must use a_wet / n_cells.
    # Omitting that division would assign the whole plate's conductance to one
    # cell, making the cooling term ~12x too strong and collapsing the
    # equilibrium onto the coolant inlet temperature (the controller would then
    # have no usable authority over the pack temperature).
    a_wet_cell = a_wet / max(COLD_PLATE.n_cells, 1)

    # Series reduction of two resistances in parallel: the wall film and the
    # bulk coolant's transport capacity. The latter sets a floor on how fast
    # the coolant can absorb heat while flowing through the plate, and is what
    # bounds the achievable cooling duty cycle.
    g_wall = h_in * a_wet_cell
    mcp = coolant["density"] * (flow_lpm / 60000.0) * coolant["cp"]
    g_conv = min(g_wall, mcp * 6.0 / max(COLD_PLATE.n_cells, 1))
    q_cool = g_conv * (cell_temp_c - coolant_inlet_c)

    # Ambient and radiation paths, sized for a whole *zone* (n_s cells), not a
    # single cell.  Using a single cell's envelope against a zone's thermal mass
    # understates the external heat path by the cell count, which makes the pack
    # look far more thermally capable than it is.
    a_shell = CELL_GEOMETRY.width_m * CELL_GEOMETRY.height_m * 0.55 * n_s
    h_out = float(external_convection_coeff(cell_temp_c, 2.5))
    q_amb = h_out * a_shell * (cell_temp_c - ambient_c)
    q_rad = EMISSIVITY * SIGMA_SB * (
        ((cell_temp_c + 273.15) ** 4 - (ambient_c + 273.15) ** 4)
    ) * a_shell * 0.5

    return (q_joule + q_aux - q_cool - q_amb - q_rad) / c_th_per_cell


def required_flow_for_target(
    spec: ChemistrySpec,
    cell_temp_c: float,
    target_c: float,
    coolant_inlet_c: float,
    current_a: float,
    soc: float,
    ambient_c: float,
    solar_w_m2: float = 800.0,
    road_surface_c: float | None = None,
    aux_heat_w: float = 0.0,
) -> float:
    """Invert the lumped model for the flow that holds ``cell_temp_c`` steady.

    Bisection on :func:`lumped_pack_rhs` -- robust, monotone in ``flow_lpm`` for
    all admissible operating points, and far cheaper (and more predictable) than
    a Newton solve inside an online controller.  A road temperature of
    ``ambient + 12`` is assumed when the caller does not supply one.

    Returns
    -------
    float
        The required flow in L/min.  ``0.8`` (the minimum) when the pack is
        already over-cooled, ``14.0`` (the maximum) when the target is
        unreachable -- the caller is expected to interpret the saturations
        rather than treat them as exact solutions, which is why the safety
        supervisor and not this function owns the thermal guarantee.
    """
    lo, hi = 0.8, 14.0
    road = ambient_c + 12.0 if road_surface_c is None else road_surface_c
    target_rhs = 1e-9

    def f(q: float) -> float:
        return lumped_pack_rhs(spec, cell_temp_c, ambient_c, road, solar_w_m2,
                               current_a, soc, q, coolant_inlet_c,
                               aux_heat_w=aux_heat_w)

    if f(lo) < target_rhs:
        return lo          # already over-cooled at the minimum flow
    if f(hi) > target_rhs:
        return hi          # target unreachable at maximum flow

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if f(mid) > target_rhs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def thermal_runaway_margin(spec: ChemistrySpec, cell_temp_c: float | Tensor) -> float | Tensor:
    """Adiabatic exotherm onset margin in K (NMC ~210 degC, LFP ~270 degC)."""
    return spec.runaway_c - cell_temp_c


def reynolds_from_flow(flow_lpm: float, coolant: Dict[str, float]) -> float:
    """Convenience wrapper for the bundle Reynolds number at a given flow."""
    u_mean = (flow_lpm / 60000.0) / COLD_PLATE.fluid_area_m2
    return coolant["density"] * u_mean * COLD_PLATE.hydraulic_diameter_m / coolant["mu"]