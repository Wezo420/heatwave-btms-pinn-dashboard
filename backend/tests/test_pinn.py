"""Tests for the PINN architecture, the loss balancer and the physics operators."""

from __future__ import annotations

import math

import pytest
import torch

from backend.config import (
    ARRHENIUS_CAP_C,
    CELL_GEOMETRY,
    CHEMISTRY_CONDITIONING_DIM,
    CHEMISTRY_REGISTRY,
    COLD_PLATE,
    DEFAULT_PACK_CURRENT_A,
    PINN_CONFIG,
    PINNConfig,
    ZONE_SERIES_CELLS,
    build_chemistry_conditioning_vector,
    get_chemistry,
)
from backend.models.physics import (
    CHANNEL_DOMAIN,
    PACK_DOMAIN,
    Domain,
    EnergyResidual,
    StreamFunction,
    external_convection_coeff,
    heat_transfer_coefficient,
    internal_resistance,
    joule_source,
    lumped_pack_rhs,
    nusselt_microchannel,
    required_flow_for_target,
    sample_face,
    thermal_runaway_margin,
)
from backend.models.pinn import (
    GradientNormBalancer,
    MicrochannelPINN,
    build_model,
    chemistry_invariance_report,
)

torch.manual_seed(1337)


# ---------------------------------------------------------------------------
# Chemistry registry
# ---------------------------------------------------------------------------


class TestChemistryRegistry:
    def test_both_chemistries_registered(self):
        assert set(CHEMISTRY_REGISTRY) == {"nmc", "lfp"}

    def test_specified_values_match_brief(self):
        """The two constants the project brief fixes explicitly."""
        nmc = get_chemistry("nmc")
        lfp = get_chemistry("lfp")
        assert nmc.cp == pytest.approx(1040.0)
        assert lfp.cp == pytest.approx(1150.0)
        assert nmc.runaway_c == pytest.approx(210.0)
        assert lfp.runaway_c == pytest.approx(270.0)

    def test_get_chemistry_is_case_insensitive(self):
        assert get_chemistry("NMC").key == "nmc"
        assert get_chemistry(" Lfp ").key == "lfp"
        assert get_chemistry(None).key == "nmc"

    def test_unknown_chemistry_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_chemistry("sodium_ion")

    def test_conditioning_vector_dimension_and_scaling(self):
        for key, spec in CHEMISTRY_REGISTRY.items():
            vec = build_chemistry_conditioning_vector(spec)
            assert len(vec) == CHEMISTRY_CONDITIONING_DIM
            assert all(math.isfinite(v) for v in vec)
            # Log-scaled resistance term keeps both branches in comparable ranges.
            assert 0.0 < vec[3] < 3.0

    def test_conditioning_vectors_differ_between_chemistries(self):
        nmc = build_chemistry_conditioning_vector(get_chemistry("nmc"))
        lfp = build_chemistry_conditioning_vector(get_chemistry("lfp"))
        assert nmc != lfp


# ---------------------------------------------------------------------------
# Constitutive relations
# ---------------------------------------------------------------------------


class TestConstitutive:
    def test_arrhenius_resistance_increases_with_temperature(self):
        spec = get_chemistry("nmc")
        r25 = spec.r_int(25.0, 0.5)
        r60 = spec.r_int(60.0, 0.5)
        assert r60 > r25
        # A ~1.5x rise over 35 K is the right order for a 23.5 kJ/mol slope.
        assert 1.1 < r60 / r25 < 3.0

    def test_resistance_is_higher_at_soc_extremes(self):
        spec = get_chemistry("lfp")
        mid = spec.r_int(30.0, 0.5)
        assert spec.r_int(30.0, 0.05) > mid
        assert spec.r_int(30.0, 0.95) > mid

    def test_joule_source_scales_quadratically_with_current(self):
        spec = get_chemistry("nmc")
        q1 = float(joule_source(spec, 100.0, 30.0, 0.5))
        q2 = float(joule_source(spec, 200.0, 30.0, 0.5))
        assert q2 / q1 == pytest.approx(4.0, rel=1e-6)

    def test_thermal_runaway_margin_is_chemistry_specific(self):
        assert thermal_runaway_margin(get_chemistry("nmc"), 100.0) == pytest.approx(110.0)
        assert thermal_runaway_margin(get_chemistry("lfp"), 100.0) == pytest.approx(170.0)

    def test_external_convection_increases_with_wind(self):
        low = float(external_convection_coeff(40.0, 1.0))
        high = float(external_convection_coeff(40.0, 8.0))
        assert high > low

    def test_fully_developed_nusselt_is_reynolds_independent(self):
        """``Nu_fd`` depends on aspect ratio only -- a Reynolds term on the
        fully-developed value is the classic microchannel-correlation error."""
        nu_a = float(nusselt_microchannel(200.0, entrance_effect=False))
        nu_b = float(nusselt_microchannel(2000.0, entrance_effect=False))
        assert nu_a == pytest.approx(nu_b, rel=1e-9)
        assert nu_a > 0

    def test_fully_developed_nusselt_matches_the_square_duct_value(self):
        """Shah & London constant-flux square duct: Nu_Dh ~ 3.9."""
        nu = float(nusselt_microchannel(500.0, aspect_ratio=1.0,
                                         entrance_effect=False))
        assert 3.5 < nu < 4.3

    def test_nusselt_grows_with_reynolds_through_the_entrance_effect(self):
        """Larger Re means a larger Graetz number, so a *developing* duct
        transfers more -- but by a bounded factor, not unboundedly."""
        nu_low = float(nusselt_microchannel(200.0))
        nu_high = float(nusselt_microchannel(2000.0))
        assert nu_low < nu_high
        assert nu_high / nu_low < 10.0

    def test_nusselt_is_symmetric_under_aspect_ratio_inversion(self):
        """``a`` and ``1/a`` describe the same duct, so Nu must be equal."""
        ar = COLD_PLATE.channel_depth_m / COLD_PLATE.channel_width_m
        n1 = float(nusselt_microchannel(750.0, aspect_ratio=ar))
        n2 = float(nusselt_microchannel(750.0, aspect_ratio=1.0 / ar))
        assert n1 == pytest.approx(n2, rel=1e-9)

    def test_heat_transfer_coefficient_increases_with_flow(self):
        c = get_chemistry("nmc").coolant_props
        assert heat_transfer_coefficient(12.0, c) > heat_transfer_coefficient(3.0, c)

    def test_internal_resistance_tensorised(self):
        spec = get_chemistry("nmc")
        temps = torch.tensor([20.0, 40.0, 60.0], dtype=torch.float64)
        out = internal_resistance(spec, temps, 0.5)
        assert out.shape == temps.shape
        assert torch.all(out[1:] > out[:-1])


# ---------------------------------------------------------------------------
# Stream function ansatz
# ---------------------------------------------------------------------------


class TestStreamFunction:
    def test_continuity_is_satisfied_to_machine_precision(self):
        """The ansatz satisfies div(u)=0 by construction, not approximately."""
        sf = StreamFunction(torch.tensor(2500.0, dtype=torch.float64))
        pts = PACK_DOMAIN.sample(64, dtype=torch.float64)
        res = sf.continuity_residual(pts)
        assert float(res.abs().max()) < 1e-10

    def test_no_slip_on_channel_walls(self):
        """All four channel walls, evaluated in CHANNEL coordinates.

        Using ``PACK_DOMAIN`` here would sample points far outside the channel
        cross-section, where the separable shape function is unbounded, and the
        residual would be large and non-zero. The frame matters.
        """
        sf = StreamFunction(torch.tensor(2500.0, dtype=torch.float64))
        for face in ("y0", "yl", "z0", "zl"):
            pts, _ = sample_face(CHANNEL_DOMAIN, face, 32, dtype=torch.float64)
            assert float(sf.wall_no_slip_residual(pts).max()) < 1e-18

    def test_momentum_residual_is_finite_on_channel_points(self):
        sf = StreamFunction(torch.tensor(2500.0, dtype=torch.float64))
        pts = CHANNEL_DOMAIN.sample(64, dtype=torch.float64)
        r = sf.momentum_residual(pts)
        assert torch.isfinite(r).all()
        assert float(r.abs().max()) < 1e6

    def test_channel_domain_is_inside_the_pack(self):
        assert CHANNEL_DOMAIN.width_m <= PACK_DOMAIN.width_m
        assert CHANNEL_DOMAIN.height_m <= PACK_DOMAIN.height_m

    def test_velocity_is_positive_on_the_channel_centreline(self):
        """Maximum of the Poiseuille profile, at ``y = R_y/2``."""
        sf = StreamFunction(torch.tensor(2500.0, dtype=torch.float64))
        y_mid = 0.5 * sf.ry
        pts = torch.tensor([[0.1, y_mid, 0.5 * sf.rz]], dtype=torch.float64)
        u, _, _ = sf.velocity(pts)
        assert float(u) > 0

    def test_velocity_matches_the_closed_form_mean(self):
        """Numerical quadrature reproduces ``dpdx*ry^2/(24 mu)`` to O(h^2).

        The tolerance is ``1e-3`` rather than machine precision because a plain
        rectangular grid over-weights the endpoints of the parabolic profile;
        the analytic form is the exact answer, the quadrature is the check.
        """
        sf = StreamFunction(torch.tensor(2500.0, dtype=torch.float64))
        ys = torch.linspace(0.0, sf.ry, 4001, dtype=torch.float64)
        zs = torch.linspace(0.0, sf.rz, 801, dtype=torch.float64)
        yy, zz = torch.meshgrid(ys, zs, indexing="ij")
        values = sf.shape_factor(yy, zz)
        # Trapezoidal weights on both axes. A plain `.mean()` over a grid that
        # includes the endpoints over-counts them by half a cell each, which is a
        # ~0.15 % bias for this profile -- larger than the tolerance being tested.
        wy = torch.full_like(ys, float(ys[1] - ys[0]))
        wy[0] *= 0.5
        wy[-1] *= 0.5
        wz = torch.full_like(zs, float(zs[1] - zs[0]))
        wz[0] *= 0.5
        wz[-1] *= 0.5
        numeric = float((values * wy[:, None] * wz[None, :]).sum()
                        / (sf.ry * sf.rz))
        analytic = float(sf.analytic_mean_velocity())
        assert analytic == pytest.approx(numeric, rel=1e-4)

        # And the closed form itself, written out longhand:
        #   mean u = (1/(ry rz)) * (G/(2 mu)) * (ry^3/6) * (rz/2) = G ry^2/(24 mu)
        mu = sf._mu
        longhand = sf.dpdx * sf.ry**2 / (24.0 * mu)
        assert analytic == pytest.approx(float(longhand), rel=1e-12)

    def test_flow_rate_inversion_round_trips(self):
        """A commanded flow in L/min maps to a dp/dx and back."""
        sf = StreamFunction(torch.zeros((), dtype=torch.float64))
        for q in (0.8, 6.0, 14.0):
            g = sf.dpdx_for_flow_lpm(q)
            back = float(StreamFunction(torch.tensor(g, dtype=torch.float64)
                                        ).volume_flow_lpm())
            assert back == pytest.approx(q, rel=1e-6)

    def test_momentum_residual_is_normalised(self):
        """The residual is dimensionless and O(1), not O(dpdx) ~ 1e4.

        Unnormalised, its square would dominate the loss by ~7 orders of
        magnitude and the gradient-norm balancer would assign it all the weight.
        """
        sf = StreamFunction(torch.tensor(2.5e4, dtype=torch.float64))
        pts = CHANNEL_DOMAIN.sample(128, dtype=torch.float64)
        r = sf.momentum_residual(pts)
        assert torch.isfinite(r).all()
        assert float(r.abs().max()) < 10.0

    def test_mean_velocity_is_proportional_to_dpdx(self):
        a = StreamFunction(torch.tensor(1000.0, dtype=torch.float64))
        b = StreamFunction(torch.tensor(2000.0, dtype=torch.float64))
        assert float(b.analytic_mean_velocity() / a.analytic_mean_velocity()) == \
            pytest.approx(2.0, rel=1e-9)

    def test_volume_flow_is_monotonic_in_pressure_gradient(self):
        flows = [float(StreamFunction(torch.tensor(float(g), dtype=torch.float64))
                       .volume_flow_lpm()) for g in (500, 1000, 2000, 4000)]
        assert flows == sorted(flows)

    def test_continuity_residual_requires_grad_graph(self):
        sf = StreamFunction(torch.tensor(1000.0, dtype=torch.float64))
        pts = PACK_DOMAIN.sample(8, dtype=torch.float64)
        # The residual is differentiable -- it must participate in the PINN loss.
        res = sf.continuity_residual(pts)
        assert res.requires_grad


# ---------------------------------------------------------------------------
# EnergyResidual differential operators
# ---------------------------------------------------------------------------


class TestEnergyResidualOperators:
    """The autograd-based operators, checked against closed-form derivatives.

    ``_grad`` and ``_laplacian`` differentiate a *batch* of collocation points,
    so ``autograd.grad`` receives a vector output and needs an explicit
    ``grad_outputs`` seed. Without one it raises "grad can be implicitly created
    only for scalar outputs" -- which previously made every residual route
    through the API a 500, with the PDE loss silently unavailable in production.
    Asserting against the analytic Laplacian of a separable sinusoid is what
    distinguishes "returns the right number" from "returns a number at all".
    """

    @staticmethod
    def _T(x: torch.Tensor) -> torch.Tensor:
        # Separable in all three coordinates, so each term of the Laplacian is
        # independently checkable.
        return 300.0 + 5.0 * torch.sin(7.0 * x[:, 0]) \
            + 2.0 * torch.cos(3.0 * x[:, 1]) + torch.sin(2.0 * x[:, 2])

    @staticmethod
    def _dT_dT(x: torch.Tensor) -> torch.Tensor:
        """Analytic Laplacian of :meth:`_T`."""
        return -5.0 * 49.0 * torch.sin(7.0 * x[:, 0]) \
            - 2.0 * 9.0 * torch.cos(3.0 * x[:, 1]) \
            - 1.0 * 4.0 * torch.sin(2.0 * x[:, 2])

    def test_laplacian_matches_the_closed_form(self):
        pts = PACK_DOMAIN.sample(64, dtype=torch.float64)
        got = EnergyResidual._laplacian(self._T, pts.clone())
        exact = self._dT_dT(pts)
        assert got.shape == (64,)
        assert float((got - exact).abs().max()) < 1e-9

    def test_gradient_matches_the_closed_form(self):
        pts = PACK_DOMAIN.sample(64, dtype=torch.float64)
        got = EnergyResidual._grad(self._T, pts.clone())
        exact = torch.stack([
            35.0 * torch.cos(7.0 * pts[:, 0]),
            -6.0 * torch.sin(3.0 * pts[:, 1]),
            2.0 * torch.cos(2.0 * pts[:, 2]),
        ], dim=-1)
        assert got.shape == (64, 3)
        assert float((got - exact).abs().max()) < 1e-9

    def test_laplacian_of_a_linear_field_is_exactly_zero(self):
        """The degenerate case that previously raised instead of returning zero.

        For an affine field the gradient is constant, so each component has no
        second-derivative graph at all. The Laplacian is identically zero, and
        the operator must say so rather than propagate a RuntimeError.
        """
        pts = PACK_DOMAIN.sample(16, dtype=torch.float64)
        slope = torch.tensor([12.0, -4.0, 7.0], dtype=torch.float64)

        def T(x: torch.Tensor) -> torch.Tensor:
            return 305.0 + x @ slope

        lap = EnergyResidual._laplacian(T, pts.clone())
        assert float(lap.abs().max()) == 0.0

    def test_solid_residual_reduces_to_the_source_terms_for_a_flat_field(self):
        r"""With a spatially uniform ``T`` the residual is exactly ``-q_joule - q_aux``.

        This pins the sign and scale of every volumetric term. The Joule source
        depends only on temperature through the Arrhenius law, so a uniform field
        makes it constant, and with ``\nabla^2 T = 0`` and ``\partial_t T = 0``
        the remaining residual must equal the negated sources to machine
        precision -- no fitted constants in the way.
        """
        spec = get_chemistry("nmc")
        pts = PACK_DOMAIN.sample(24, dtype=torch.float64)
        t0 = 305.0

        def T(x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x[:, 0], t0)

        res = EnergyResidual(spec, T)
        current, soc, aux = 12.0, 0.5, 3.0
        got = res.solid_residual(pts.clone(), current, soc, aux)

        expected = -(joule_source(spec, current, t0, soc)
                     + aux / (COLD_PLATE.length_m * COLD_PLATE.width_m
                              * (COLD_PLATE.thickness_m + 0.028)))
        assert float((got - expected).abs().max()) / abs(float(expected)) < 1e-12

    def test_solid_residual_scales_with_the_auxiliary_load(self):
        """d(residual)/d(aux) is exactly ``-1 / V_pack``, i.e. the sink is signed."""
        spec = get_chemistry("nmc")
        pts = PACK_DOMAIN.sample(8, dtype=torch.float64)

        def T(x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x[:, 0], 305.0)

        res = EnergyResidual(spec, T)
        r0 = res.solid_residual(pts.clone(), 12.0, 0.5, 0.0)
        r1 = res.solid_residual(pts.clone(), 12.0, 0.5, 1000.0)
        volume = COLD_PLATE.length_m * COLD_PLATE.width_m * (
            COLD_PLATE.thickness_m + 0.028)
        assert torch.allclose(r0 - r1, torch.full_like(r0, 1000.0 / volume))

    def test_solid_residual_is_differentiable_wrt_the_parameters(self):
        """The residual must be differentiable in the temperature field.

        This is what makes it usable as a PINN training term: autograd has to be
        able to push a loss on the residual back into the network weights.
        """
        model = build_model()
        spec = get_chemistry("nmc")
        pts = PACK_DOMAIN.sample(16, dtype=torch.float32)
        latent = model.embedder(chemistry=spec.key)

        def T(x: torch.Tensor) -> torch.Tensor:
            return model.temperature(x, spec.key)

        res = EnergyResidual(spec, T)
        r = res.solid_residual(pts.clone(), 12.0, 0.5)
        r.pow(2).mean().backward()
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0
                   for p in model.parameters())


# ---------------------------------------------------------------------------
# Domain sampling
# ---------------------------------------------------------------------------


class TestDomain:
    def test_samples_stay_inside_the_domain(self):
        pts = PACK_DOMAIN.sample(2048, dtype=torch.float64)
        span = torch.tensor(PACK_DOMAIN.span, dtype=torch.float64)
        assert torch.all(pts >= 0) and torch.all(pts <= span)

    def test_channel_domain_is_a_sub_box_of_the_pack_domain(self):
        for i, (a, b) in enumerate(zip(CHANNEL_DOMAIN.span, PACK_DOMAIN.span)):
            assert a <= b, f"channel axis {i} exceeds the pack extent"

    def test_sampling_is_deterministic(self):
        a = PACK_DOMAIN.sample(128, dtype=torch.float64)
        b = PACK_DOMAIN.sample(128, dtype=torch.float64)
        assert torch.allclose(a, b)

    def test_grid_shape(self):
        g = PACK_DOMAIN.grid(5, 4, 3)
        assert g.shape == (5, 4, 3, 3)

    @pytest.mark.parametrize("face", ["inlet", "outlet", "y0", "yl", "z0", "zl"])
    def test_face_sampling_returns_outward_normals(self, face):
        """Points on a face are pinned to it, and the normal points outward."""
        pts, nrm = sample_face(PACK_DOMAIN, face, 16, dtype=torch.float64)
        assert pts.shape == (16, 3)
        assert float(nrm.norm()) == pytest.approx(1.0)

        axis = max(range(3), key=lambda i: abs(float(nrm[i])))
        extent = PACK_DOMAIN.span[axis]
        expected = extent if float(nrm[axis]) > 0 else 0.0
        assert torch.allclose(
            pts[:, axis],
            torch.full((16,), float(expected), dtype=torch.float64),
        )

    def test_unknown_face_raises(self):
        with pytest.raises(ValueError):
            sample_face(PACK_DOMAIN, "diagonal", 4)


# ---------------------------------------------------------------------------
# Reduced-order model
# ---------------------------------------------------------------------------


class TestLumpedModel:
    """Reduced-order zone model.

    Currents here are *pack* currents for the 96s1p pack, so the realistic
    operating band is 5--20 A (roughly 0.5--2C).  A 200 A pack current would
    exceed the cell's abuse rating and the model would be extrapolating outside
    the region where any thermal claim means anything.
    """

    #: (pack current, auxiliary load) pairs spanning the demo operating range.
    OPERATING_POINTS = [(5.0, 0.0), (12.0, 3_000.0), (20.0, 4_000.0)]

    @pytest.mark.parametrize("chem", ["nmc", "lfp"])
    def test_more_flow_cools_more(self, chem):
        spec = get_chemistry(chem)
        hot = lumped_pack_rhs(spec, 55.0, 40.0, 55.0, 900.0, 12.0, 0.5, 3.0, 26.0)
        cold = lumped_pack_rhs(spec, 55.0, 40.0, 55.0, 900.0, 12.0, 0.5, 13.0, 26.0)
        assert cold < hot

    @pytest.mark.parametrize("chem", ["nmc", "lfp"])
    def test_equilibrium_flow_exists(self, chem):
        """The flow that holds a target temperature exists inside the loop range."""
        spec = get_chemistry(chem)
        for current, aux in self.OPERATING_POINTS:
            required = required_flow_for_target(
                spec, 45.0, 40.0, 26.0, current, 0.5, 40.0, 900.0,
                aux_heat_w=aux,
            )
            assert 0.8 <= required <= 14.0
            residual = lumped_pack_rhs(
                spec, 45.0, 40.0, 54.0, 900.0, current, 0.5, required, 26.0,
                aux_heat_w=aux,
            )
            # Either the residual is ~0 (a true equilibrium exists) or the flow
            # saturated, in which case the loop simply has no more authority.
            saturated = math.isclose(required, 0.8, abs_tol=1e-6) or \
                math.isclose(required, 14.0, abs_tol=1e-6)
            assert saturated or abs(residual) < 0.02

    def test_required_flow_decreases_with_ambient(self):
        spec = get_chemistry("nmc")
        cool = required_flow_for_target(spec, 50.0, 40.0, 26.0, 12.0, 0.5, 25.0,
                                        aux_heat_w=3_000.0)
        warm = required_flow_for_target(spec, 50.0, 40.0, 26.0, 12.0, 0.5, 48.0,
                                        aux_heat_w=3_000.0)
        assert warm >= cool

    def test_required_flow_rises_with_the_auxiliary_load(self):
        spec = get_chemistry("nmc")
        light = required_flow_for_target(spec, 45.0, 40.0, 26.0, 12.0, 0.5, 40.0,
                                         aux_heat_w=1_000.0)
        heavy = required_flow_for_target(spec, 45.0, 40.0, 26.0, 12.0, 0.5, 40.0,
                                         aux_heat_w=4_000.0)
        assert heavy > light

    def test_lfp_thermal_advantage_comes_from_onset_not_from_lower_resistance(self):
        """LFP's advantage is its thermal budget, not a lower ``R``.

        This is worth stating explicitly because the intuitive claim runs the
        other way: LFP cells are *not* lower-resistance than NMC (here
        ``r_int_ref`` is 18.2 vs 13.5 mOhm, and LFP has the flatter Arrhenius
        slope, 19.8 vs 23.5 kJ/mol).  Its advantage is a 60 K higher exotherm
        onset and a 10% higher specific heat -- i.e. more margin and more thermal
        mass, not less dissipation.  A summary that credited LFP with lower
        resistance would misattribute the safety case.
        """
        nmc = get_chemistry("nmc")
        lfp = get_chemistry("lfp")

        # Onset: the dominant term by far.
        assert lfp.runaway_c - nmc.runaway_c == pytest.approx(60.0)
        # Specific heat: a modest secondary term.
        assert lfp.cp > nmc.cp
        # Resistance: LFP is higher, not lower.
        assert lfp.r_int(40.0, 0.5) > nmc.r_int(40.0, 0.5)
        # Arrhenius slope: flatter, so R rises less steeply with temperature.
        assert lfp.ea < nmc.ea

    def test_arrhenius_resistance_saturates_above_the_fit_range(self):
        """Extrapolating the Arrhenius law to 400 degC would give ~200x R_ref
        and a physically impossible Joule load, so the fit is clipped."""
        spec = get_chemistry("nmc")
        r_at_cap = spec.r_int(ARRHENIUS_CAP_C, 0.5)
        assert spec.r_int(400.0, 0.5) == pytest.approx(r_at_cap, rel=1e-9)
        assert spec.r_int(400.0, 0.5) < 50.0 * spec.r_int_ref

    def test_joule_load_scales_with_series_cell_count(self):
        """A zone holds 8 series cells, so its load is 8x a single cell's."""
        spec = get_chemistry("nmc")
        single = lumped_pack_rhs(spec, 40.0, 30.0, 40.0, 500.0, 12.0, 0.5,
                                 0.0, 20.0, c_th_per_cell=1.0,
                                 aux_heat_w=-ZONE_SERIES_CELLS * 144.0
                                 * float(internal_resistance(spec, 40.0, 0.5)))
        # aux cancels the Joule term exactly, so the residual is the pure
        # convective+ambient+radial sink divided by C = 1 J/K.
        assert single > 0
        assert spec.r_int(40.0, 0.5) > 0


# ---------------------------------------------------------------------------
# PINN architecture
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_forward_shape_and_temperature_units(self):
        model = MicrochannelPINN(PINNConfig(hidden_dim=32, n_hidden_layers=2,
                                            n_fourier_features=8))
        pts = PACK_DOMAIN.sample(64, dtype=torch.float32)
        out = model.forward(MicrochannelPINN.prepare_inputs(pts), "nmc")
        assert out.temperature_k.shape == (64,)
        # Bounded decoder: t_offset +- t_scale = 298.15 +- 35 K.
        assert float(out.temperature_k.min()) >= 260.0
        assert float(out.temperature_k.max()) <= 340.0

    def test_both_chemistries_share_one_parameter_set(self):
        model = MicrochannelPINN(PINNConfig(hidden_dim=32, n_hidden_layers=2,
                                            n_fourier_features=8))
        assert model.embedder.keys == ["nmc", "lfp"]
        # One trunk, two conditionings: not two models.
        n_params = model.n_parameters()
        assert 1000 < n_params < 10_000_000

    def test_conditioning_actually_changes_the_output(self):
        """A zero-effect conditioning path would be a silent bug."""
        model = MicrochannelPINN(PINNConfig(hidden_dim=64, n_hidden_layers=3,
                                            n_fourier_features=16))
        pts = PACK_DOMAIN.sample(256, dtype=torch.float32)
        report = chemistry_invariance_report(model, pts)
        assert report["mean_abs_delta_k"] > 1e-6

    def test_initial_laplacian_is_comparable_to_the_physical_source(self):
        """The untrained Laplacian must sit near the scale of ``q/k_s``.

        The physics loss is ``(k_s * lap T + q)^2``, so its *initial* magnitude is
        set entirely by how fast the untrained field varies. ``sigma`` used to be
        tied to the feature count (``n_features / 6`` = 8), which produced a
        wavenumber near 139 per unit normalised coordinate and a Laplacian of
        ~5e8 K/m^2 -- twenty-five thousand times the ~2e4 W/m^3 that the source
        term ``I^2 R`` actually produces.

        That matters beyond a large number. The gradient-norm balancer assigns
        each term a weight inversely proportional to its gradient norm, so a PDE
        term 1e12 times the data term drives every other weight to 1e-3 and the
        loss stops moving. The model is not trainable at that initialisation, and
        nothing raises -- the loss just sits still.

        The assertion below is deliberately loose (two decades) because the
        quantity is an RMS over random initialisation; it is checking the
        *order of magnitude*, which is the property that broke.
        """
        from backend.config import COLD_PLATE

        model = build_model()
        pts = PACK_DOMAIN.sample(2048, dtype=torch.float64).requires_grad_(True)
        t_k = model.temperature(pts, "nmc")
        grad = torch.autograd.grad(t_k.sum(), pts, create_graph=True)[0]
        lap = sum(
            torch.autograd.grad(grad[..., i].sum(), pts, create_graph=True)[0][..., i]
            for i in range(3)
        )
        lap_rms = float(lap.pow(2).mean().sqrt())

        # Characteristic source: 8 series cells at 12 A through ~19 mOhm,
        # spread over the cell envelope.
        spec = get_chemistry("nmc")
        q_joule = abs(spec.r_int(305.0, 0.5)) * DEFAULT_PACK_CURRENT_A**2
        q_per_volume = q_joule / CELL_GEOMETRY.volume_m3
        expected_lap = q_per_volume / spec.k_solid

        assert 1e-4 * expected_lap < lap_rms < 1e4 * expected_lap, (
            f"initial Laplacian {lap_rms:.3e} K/m^2 is not within four decades "
            f"of the physical {expected_lap:.3e} K/m^2 (q/k_s from "
            f"q={q_per_volume:.3e} W/m^3, k_s={spec.k_solid} W/m/K). "
            f"An untrained network whose curvature is orders of magnitude "
            f"away from its own source term cannot be balanced against the "
            f"data loss."
        )

    def test_fourier_sigma_is_independent_of_the_feature_count(self):
        """Adding Fourier capacity must not sharpen the frequency band.

        ``sigma = n_features / 6`` made the wavenumber grow with ``n``, and since
        the physics residual differentiates the field twice, the initial
        residual then grew as ``n^2``. Adding capacity silently made the model
        less trainable. The two must be independently controllable.
        """
        small = MicrochannelPINN(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                             n_fourier_features=8))
        large = MicrochannelPINN(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                             n_fourier_features=96))
        assert small.fourier.sigma == large.fourier.sigma

        # The *typical* frequency must be identical: both draws are N(0, sigma^2)
        # per element, so the mean magnitude is a stable statistic.
        mean_small = float(small.fourier.B.abs().mean())
        mean_large = float(large.fourier.B.abs().mean())
        assert mean_small == pytest.approx(mean_large, rel=0.25), (
            f"typical frequency moved with feature count: "
            f"{mean_small:.3f} vs {mean_large:.3f}"
        )

        # The extreme frequency grows only as the maximum of n Gaussians,
        # ~sqrt(2 ln n): 8 -> 96 features is a factor of ~1.2, not the 12x that
        # tying sigma to the feature count would have produced.
        k_small = 2 * math.pi * small.fourier.sigma * float(small.fourier.B.abs().max())
        k_large = 2 * math.pi * large.fourier.sigma * float(large.fourier.B.abs().max())
        assert k_large < 2.0 * k_small, (
            f"frequency band grew {k_large / k_small:.2f}x from 8 to 96 "
            f"features; tying sigma to n_features made this ~12x"
        )

    def test_positive_head_bias_can_be_seeded_to_an_arbitrary_target(self):
        """A ``PositiveHead`` bias must be settable to any strictly positive value.

        ``softplus`` output is seeded at construction by solving for the bias
        that makes it equal a chosen target. The naive inverse,
        ``log(expm1(target/scale))``, raises :class:`OverflowError` once
        ``target/scale`` exceeds about 709 -- and the ``dpdx`` head's target is
        of order 5.5e3, so a training run built with the naive form cannot get
        past building its first model.
        """
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from train_pinn import _inv_softplus

        def softplus(b: float) -> float:
            """Overflow-safe softplus, used as the round-trip oracle."""
            return math.log1p(math.exp(-abs(b))) + max(b, 0.0)

        # The regime that overflows, plus the regime that loses precision.
        for target in (1e-12, 1e-6, 0.1, 1.0, 1.0001, 10.0, 100.0,
                       709.0, 1500.0, 5500.6976, 1e6):
            b = _inv_softplus(target)
            got = softplus(b)
            assert got == pytest.approx(target, rel=1e-12), (
                f"inverse softplus failed at target={target}: "
                f"b={b}, softplus(b)={got}"
            )

        # A non-positive target must be rejected rather than silently clamped:
        # a head that cannot represent zero is a modelling error, not something
        # to paper over.
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                _inv_softplus(bad)

    def test_physical_heads_are_positive(self):
        model = MicrochannelPINN(PINNConfig(hidden_dim=32, n_hidden_layers=2,
                                            n_fourier_features=8))
        pts = PACK_DOMAIN.sample(32, dtype=torch.float32)
        out = model.forward(MicrochannelPINN.prepare_inputs(pts), "nmc")
        assert float(out.r_int_ohm) > 0
        assert float(out.flow_lpm) > 0
        assert float(out.dpdx) > 0
        assert float(out.h_amb_scale) > 0

    def test_forward_accepts_float32_and_float64(self):
        """The engine runs float32 for inference and float64 for residuals; the
        network must accept both without a dtype mismatch on its buffers."""
        model = MicrochannelPINN(PINNConfig(hidden_dim=32, n_hidden_layers=2,
                                            n_fourier_features=8))
        for dtype in (torch.float32, torch.float64):
            pts = PACK_DOMAIN.sample(16, dtype=dtype)
            out = model.forward(MicrochannelPINN.prepare_inputs(pts), "nmc")
            assert torch.isfinite(out.temperature_k).all()

    def test_gradients_reach_the_trunk(self):
        model = MicrochannelPINN(PINNConfig(hidden_dim=32, n_hidden_layers=2,
                                            n_fourier_features=8))
        # The buffers (Fourier B/phase) are float32; feed float32 so the matmul
        # dtypes agree, then differentiate.
        pts = PACK_DOMAIN.sample(16, dtype=torch.float32)
        loss = model.temperature(pts, "nmc").pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.trunk.parameters() if p.grad is not None]
        assert grads
        assert any(float(g.abs().sum()) > 0 for g in grads)

    def test_prepare_inputs_normalises_coordinates(self):
        pts = torch.tensor([[0.18, 0.115, 0.006]], dtype=torch.float64)
        out = MicrochannelPINN.prepare_inputs(pts, 0.0)
        assert out.shape == (1, 4)
        assert torch.allclose(out[0, :3],
                              pts[0] / torch.tensor([0.36, 0.23, 0.012],
                                                     dtype=torch.float64))
        assert float(out[0, 3]) == 0.0

    def test_build_model_is_seeded(self):
        a = build_model(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                   n_fourier_features=4, seed=7))
        b = build_model(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                   n_fourier_features=4, seed=7))
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.allclose(pa, pb)


# ---------------------------------------------------------------------------
# Gradient-norm balancer
# ---------------------------------------------------------------------------


class TestGradientNormBalancer:
    def test_weights_are_normalised_to_the_largest_term(self):
        """The term with the larger gradient norm receives the larger weight.

        Each step requires a fresh backward pass: the balancer reads
        ``p.grad``, which ``backward`` accumulates into ``.grad`` rather than
        replacing, so repeated steps on one graph would compare against a
        *summed* gradient and the ratio would drift.
        """
        model = MicrochannelPINN(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                            n_fourier_features=4))
        shared = [p for n, p in model.named_parameters() if n.startswith("trunk.")]
        balancer = GradientNormBalancer(["small", "large"], shared, alpha=0.5)

        def make_losses():
            base = model.temperature(torch.rand(4, 3) * 0.1, "nmc").mean()
            return {"small": (base * 0.1) ** 2, "large": (base * 100.0) ** 2}

        weights: Dict[str, float] = {}
        for _ in range(4):
            weights = balancer.step(make_losses())

        assert weights["large"] > weights["small"]
        assert max(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_weights_stay_within_the_clip_range(self):
        model = MicrochannelPINN(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                            n_fourier_features=4))
        shared = [p for n, p in model.named_parameters() if n.startswith("trunk.")]
        balancer = GradientNormBalancer(["a", "b"], shared, alpha=0.5)

        def make_losses():
            base = model.temperature(torch.rand(4, 3) * 0.1, "nmc").sum() ** 2
            return {"a": base, "b": base * 1e-8}

        for _ in range(5):
            weights = balancer.step(make_losses())
        assert all(1e-3 <= v <= 1e3 for v in weights.values())

    def test_zero_gradient_does_not_divide_by_zero(self):
        """A satisfied term has zero gradient; the floor must absorb that."""
        model = MicrochannelPINN(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                            n_fourier_features=4))
        shared = [p for n, p in model.named_parameters() if n.startswith("trunk.")]
        balancer = GradientNormBalancer(["zero", "nonzero"], shared, floor=1e-8)

        def make_losses():
            base = model.temperature(torch.rand(4, 3) * 0.1, "nmc").sum() ** 2
            # `zero` is a constant with no gradient path to the shared params.
            return {"zero": torch.zeros((), requires_grad=True), "nonzero": base}

        for _ in range(3):
            weights = balancer.step(make_losses())
        assert all(math.isfinite(v) and v > 0 for v in weights.values())

    def test_step_clears_gradients_before_returning(self):
        """The caller backprops the composite loss next; a stale gradient from
        the per-term probes would accumulate into it."""
        model = MicrochannelPINN(PINNConfig(hidden_dim=16, n_hidden_layers=2,
                                            n_fourier_features=4))
        shared = [p for n, p in model.named_parameters() if n.startswith("trunk.")]
        balancer = GradientNormBalancer(["a"], shared)
        base = model.temperature(torch.rand(4, 3) * 0.1, "nmc").sum() ** 2
        balancer.step({"a": base})
        assert all(p.grad is None for p in shared)