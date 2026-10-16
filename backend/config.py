r"""
backend.config
==============

Central, immutable configuration for the heat-aware microchannel BTMS backend.

This module owns three things:

1.  **Paths** -- every filesystem location the system touches, resolved relative to the
    repository root so that the backend is location-independent.
2.  **The chemistry registry** -- a data-driven description of each supported cell
    chemistry.  This is deliberately *data* and not code: the PINN consumes the
    registry as a conditioning vector, so adding a chemistry (e.g. LMFP) is a
    dictionary entry rather than a refactor.
3.  **Physics constants and solver defaults** -- dimensional constants and the
    hard safety thresholds that bound both the PINN domain and the control loop.

Units are SI throughout: kelvin internally for thermodynamics, degrees Celsius only
at the API/UI boundary (see ``schemas.py`` for the conversion layer).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, Literal

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

#: ``backend/config.py`` -> ``backend/`` -> repo root
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

BACKEND_DIR: Final[Path] = REPO_ROOT / "backend"
DATA_DIR: Final[Path] = BACKEND_DIR / "data"
ARTIFACT_DIR: Final[Path] = REPO_ROOT / "artifacts"
LOG_DIR: Final[Path] = REPO_ROOT / "logs"

#: Root of the NASA Battery Data Set (PCoE ``BatteryData`` folder of ``.mat`` files).
#: Override with the ``NASA_BATTERY_DATA_ROOT`` environment variable.
NASA_DATA_ROOT: Final[Path] = Path(
    os.getenv("NASA_BATTERY_DATA_ROOT", str(DATA_DIR / "nasa" / "BatteryData"))
)

#: Cells used by default for the NMC calibration sweep and for the LFP profile.
NMC_CELL_IDS: Final[tuple[str, ...]] = ("B0005", "B0006", "B0007", "B0018")
LFP_CELL_IDS: Final[tuple[str, ...]] = ("B0033", "B0034")

CHECKPOINT_PATH: Final[Path] = ARTIFACT_DIR / "pinn_dual_chemistry.pt"
CALIBRATION_PATH: Final[Path] = ARTIFACT_DIR / "nasa_calibration.json"

for _d in (DATA_DIR, ARTIFACT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Physical constants
# ---------------------------------------------------------------------------

#: Stefan-Boltzmann related emissivity for the pack outer shell (W/m^2/K^4).
EMISSIVITY: Final[float] = 0.88
#: Stefan-Boltzmann constant.
SIGMA_SB: Final[float] = 5.670374419e-8
#: Universal gas constant, J/(mol*K).
R_GAS: Final[float] = 8.314462618
#: Faraday constant, C/mol.
FARADAY: Final[float] = 96485.33212
#: Air density at 300 K, kg/m^3 (used for the natural-convection convection coeff.).
RHO_AIR_300K: Final[float] = 1.177
#: Air thermal conductivity at 300 K, W/(m*K).
K_AIR_300K: Final[float] = 0.02624
#: Kinematic viscosity of air at 300 K, m^2/s.
NU_AIR_300K: Final[float] = 1.568e-5
#: Reference temperature for Kirchhoff/Arrhenius corrections, K (25 degC).
T_REF_K: Final[float] = 298.15

#: Upper validity bound of the Arrhenius resistance fit, degC.  Beyond this the
#: relation is saturated -- see :meth:`ChemistrySpec.r_int` for why.
ARRHENIUS_CAP_C: Final[float] = 120.0


def kelvin(celsius: float) -> float:
    """Convert degrees Celsius to kelvin."""
    return celsius + 273.15


def celsius(kelvin_: float) -> float:
    """Convert kelvin to degrees Celsius."""
    return kelvin_ - 273.15


# ---------------------------------------------------------------------------
# 3. Chemistry registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChemistrySpec:
    """A data-driven description of a cell chemistry.

    Attributes
    ----------
    key:
        Short machine identifier (``"nmc"`` / ``"lfp"``).
    label:
        Human-readable name shown in the dashboard toggle.
    cathodes / anode:
        Electrode compositions, quoted in the model card.
    cp:
        Specific heat capacity of the *cell* (canister + contents), J/(kg*K).
    k_solid:
        Effective anisotropic in-plane thermal conductivity of the jellyroll
        assembly, W/(m*K).  See ``thermal.py`` for the anisotropy ratio.
    density:
        Effective cell density, kg/m^3.
    r_int_ref:
        Reference internal resistance at 25 degC and 50% SOC, ohm.  Calibrated
        against the NASA run-to-failure cycles by ``models/nasa_dataset.py``.
    ea:
        Apparent activation energy for R_int(T), J/mol.  Drives the Arrhenius
        temperature correction applied by the PINN.
    entropy_coeff:
        Single parameter of the OCV surface of charge:
        ``OCV(SOC, T) = E0 - A*T*SOC - B*SOC + C*SOC*ln(1-SOC)``.
    runaway_c:
        Adiabatic exotherm onset temperature used as the hard safety boundary,
        degC.  Enforced by the control loop and reported by the PINN monitor.
    nominal_capacity_ah:
        Nameplate capacity, Ah (used to convert pack current to C-rate).
    energy_density_wh_kg:
        Gravimetric energy density, Wh/kg.
    coolant_props:
        Coolant thermophysical properties (50/50 water-glycol unless overridden).
    notes:
        Free-text engineering context surfaced in the dashboard / model card.
    """

    key: str
    label: str
    cathodes: str
    anode: str
    cp: float
    k_solid: float
    density: float
    r_int_ref: float
    ea: float
    entropy_coeff: float
    runaway_c: float
    nominal_capacity_ah: float
    energy_density_wh_kg: float
    coolant_props: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    # -- derived -----------------------------------------------------------
    @property
    def runaway_k(self) -> float:
        """Adiabatic exotherm onset temperature in kelvin."""
        return kelvin(self.runaway_c)

    @property
    def volumetric_heat_capacity(self) -> float:
        """Volumetric heat capacity ``rho * c_p`` in J/(m^3*K)."""
        return self.rho_vol * self.cp

    @property
    def rho_vol(self) -> float:
        """Alias kept for readability at call sites."""
        return self.density

    @property
    def specific_heat_capacity(self) -> float:
        """Alias for :attr:`cp`."""
        return self.cp

    @property
    def thermal_runaway_onset(self) -> float:
        """Alias for :attr:`runaway_c`."""
        return self.runaway_c

    @property
    def internal_resistance_reference(self) -> float:
        """Alias for :attr:`r_int_ref`."""
        return self.r_int_ref

    @property
    def nominal_cell_v(self) -> float:
        """Nominal per-cell terminal voltage, V.

        Derived from the pack rating rather than stored per chemistry: both
        chemistries in this study are packaged 96s1p at 370 V, so the per-cell
        nominal is the same 3.854 V.  Keeping it derived means the pack voltage
        and the per-cell figure can never disagree.
        """
        return PACK_NOMINAL_VOLTAGE_V / PACK_SERIES_CELLS

    def terminal_voltage(self, state_of_charge: float, current_a: float = 0.0) -> float:
        """Per-cell terminal voltage under load, V.

        A linear open-circuit ramp plus an ``I R`` drop, signed so that a
        discharge (positive current, per the project's convention) lowers the
        terminal voltage.

        This is deliberately crude -- the pack energy read-out needs a plausible
        number, not a battery model, and an equivalent-circuit fit would be
        over-claiming at this fidelity.  The ramp does capture the effect that
        matters thermally: an empty pack sits ~0.7 V lower per cell, so the same
        power draws a proportionally larger current.
        """
        soc = min(max(float(state_of_charge), 0.0), 1.0)
        ocv = 3.30 + 0.75 * soc          # 3.30 V empty .. 4.05 V full
        return max(ocv - current_a * self.r_int(ocv + 273.15, soc), 2.5)

    def thermal_diffusivity(self) -> float:
        """Cell thermal diffusivity ``k / (rho * cp)`` in m^2/s."""
        return self.k_solid / (self.density * self.cp)

    def r_int(self, temp_c: float, soc: float) -> float:
        r"""Arrhenius-corrected internal resistance, ohm.

        .. math::
           R(T,\mathrm{SOC}) = R_\text{ref}\,
             \exp\!\Bigl[\frac{E_a}{R_g}\Bigl(\tfrac1{T_\text{ref}}-\tfrac1T\Bigr)\Bigr]\,
             \bigl[1 + 0.35(1-2\,\mathrm{SOC})^2\bigr]

        **Validity range and saturation.**  The Arrhenius form is fitted to
        cell measurements over roughly \SIrange{-20}{120}{\degreeCelsius}.  Above
        \SI{120}{\degreeCelsius} the relation is *saturated* at
        ``ARRHENIUS_CAP_C``: extrapolating it to \SI{400}{\degreeCelsius} would
        predict a resistance ~200x the reference, giving a \SIrange{40}{100}{\kilo\watt}
        Joule load per cell from a single \SI{200}{\ampere} discharge -- two to
        three orders of magnitude above reality.  Physically the electrolyte
        conductivity does not keep degrading exponentially once the cell has
        dried out; further heating is governed by the exotherm, not by ``R``.

        The saturation is therefore enforced here, at the constitutive-relation
        level, rather than at each call site.  Two consequences, both
        intentional: the reduced-order model reaches a bounded equilibrium
        instead of diverging, and the fail-safe supervisor -- not the resistance
        curve -- is what detects an actual thermal runaway, which is the correct
        division of responsibility.
        """
        t_k = max(kelvin(temp_c), 200.0)
        capped_k = min(max(kelvin(temp_c), 200.0), kelvin(ARRHENIUS_CAP_C))
        arrhenius = math.exp(
            (self.ea / R_GAS) * (1.0 / T_REF_K - 1.0 / capped_k)
        )
        soc_term = 1.0 + 0.35 * (1.0 - 2.0 * max(0.0, min(1.0, soc))) ** 2
        return self.r_int_ref * arrhenius * soc_term

    def joule_heat(self, current_a: float, temp_c: float, soc: float) -> float:
        """Volumetric Joule heating ``q_vol = I^2 R / V_cell`` in W/m^3."""
        cell_volume_m3 = self.geometry.volume_m3
        return (current_a**2) * self.r_int(temp_c, soc) / cell_volume_m3

    @property
    def geometry(self) -> "CellGeometry":
        """Default prismatic cell geometry used for the volumetric source term."""
        return CELL_GEOMETRY


#: Pack topology.  96 prismatic cells in series (96s1p), giving roughly a
#: 370 V nominal pack at ~9.7 Ah -- the series count is what makes the pack
#: voltage, and with it the cell count per string, equal to the *whole* cell
#: inventory rather than a small multiple.
#:
#: This fixes the current scale, which is the single most consequential
#: convention in the reduced model: in a series pack the pack current *is* the
#: cell current, so a 15 A pack load gives each cell ``I^2 R = 15^2 * 13.5 mOhm``
#: ~ 3 W and the 96-cell pack dissipates ~290 W of heat.  Dividing the pack
#: current among parallel strings instead would under-predict the Joule load by
#: the square of the string count and make the pack thermally inert.
PACK_SERIES_CELLS: Final[int] = 96
PACK_PARALLEL_STRINGS: Final[int] = 1
#: Nominal pack voltage, V (96 x ~3.85 V).  Reported in the API and used only for
#: the electrical-power read-out; the thermal model works in current terms.
PACK_NOMINAL_VOLTAGE_V: Final[float] = 370.0

#: Cells per thermal zone (96 series cells spread over 12 zones) and the number
#: of parallel strings per zone.  In a series pack the pack current equals the
#: string current, so ``ZONE_PARALLEL_STRINGS = 1`` and a zone's Joule load is
#: ``ZONE_SERIES_CELLS * I_pack^2 * R_cell``.
ZONE_SERIES_CELLS: Final[int] = 8
ZONE_PARALLEL_STRINGS: Final[int] = 1

#: Maximum continuously addressable pack current, A.  Sized at ~2C of the 9.7 Ah
#: pack.  Beyond this the cell C-rate exceeds its abuse rating and the model
#: would be extrapolating outside the region where any thermal claim is
#: meaningful -- the API rejects larger commands rather than extrapolating.
MAX_PACK_CURRENT_A: Final[float] = 20.0

#: Default pack current for the demo scenario, A.  Roughly 1.2C -- a sustained
#: motorway cruise rather than peak acceleration, which is the regime the
#: thermal-management problem actually lives in.
DEFAULT_PACK_CURRENT_A: Final[float] = 12.0

#: Default auxiliary thermal load for the demo scenario, W (fast charging plus
#: cabin conditioning).  At 1.2C the ohmic load is only ~15 W per zone, which a
#: microchannel plate removes at well under 1 K of rise, so the fast-charge term
#: is what actually exercises the cooling loop.  See
#: :data:`backend.services.heatwave.DEFAULT_AUX_HEAT_W`.
DEFAULT_AUX_HEAT_W: Final[float] = 3_000.0

# --- Demo-scenario operating point ----------------------------------------
#: Coolant pump flow the BMS commands during normal driving, L/min.
#:
#: This is the *baseline* the anticipatory planner is scored against, and its
#: value decides whether the planner has any work to do. At 2 L/min the cold
#: plate holds the pack ~16 K above the coolant, so a pack that arrives
#: pre-warmed from a fast charge enters the heatwave with the 42 degC ceiling
#: under pressure and a reactive controller would wait until it was already
#: breached. At 6 L/min the plate has ~4.7 kW of rejection at the same
#: temperature rise -- several times this pack's own heat load -- so the two
#: schedules coincide and the anticipatory benefit is identically zero. Using
#: the wide-open pump as the baseline would be measuring the controller in the
#: one regime where it provably cannot help.
BASELINE_CRUISE_FLOW_LPM: Final[float] = 2.0

#: Pack temperature at the start of a heatwave scenario, degC.
#:
#: A pack has just spent 20 minutes on a 150 kW DC charger. Arrival at ~38 degC
#: (NMC) with the ceiling at 42 degC is the state the heatwave actually
#: confronts: starting from a cold 30 degC leaves too much headroom for the
#: anticipatory controller to matter, which is the same error as the
#: wide-open-pump baseline but in the other variable.
PRE_FAST_CHARGE_TEMP_C: Final[float] = 38.0


@dataclass(frozen=True, slots=True)
class CellGeometry:
    """Prismatic pouch-cell geometry (Tesla M2170-class, scaled)."""

    length_m: float = 0.104
    width_m: float = 0.070
    height_m: float = 0.020

    @property
    def volume_m3(self) -> float:
        """Single cell volume in m^3."""
        return self.length_m * self.width_m * self.height_m

    @property
    def zone_volume_m3(self) -> float:
        """Enclosed volume of one thermal zone (2s4p), m^3."""
        return self.volume_m3 * ZONE_SERIES_CELLS * ZONE_PARALLEL_STRINGS

    @property
    def energy_wh(self) -> float:
        """Cell energy content in Wh, derived from the chemistry's nameplate."""
        return self.volume_m3 * self.density * 0.42  # Wh/m^3 * m^3, coarse


#: Frozen geometry shared by all chemistries (chemistry changes material
#: properties, not the mechanical envelope).
CELL_GEOMETRY: Final[CellGeometry] = CellGeometry()

#: 50/50 ethylene-glycol / water coolant at ~40 degC.
_GLYCOL_COOLANT: Final[Dict[str, float]] = {
    "density": 1051.0,       # kg/m^3
    "cp": 3400.0,            # J/(kg*K)
    "k": 0.505,              # W/(m*K)
    "mu": 0.00205,           # Pa*s
    "prandtl": 13.8,
}

CHEMISTRY_REGISTRY: Final[Dict[str, ChemistrySpec]] = {
    "nmc": ChemistrySpec(
        key="nmc",
        label="NMC (Nickel Manganese Cobalt)",
        cathodes="Li(Ni_x Mn_y Co_z)O_2, x:y:z ~ 8:1:1",
        anode="Graphite (C_6)",
        cp=1040.0,               # J/(kg*K) -- specified by the project brief
        k_solid=1.10,            # W/(m*K) in-plane jellyroll
        density=2410.0,          # kg/m^3
        r_int_ref=0.0135,        # ohm @ 25 degC, 50% SOC
        ea=23500.0,              # J/mol
        entropy_coeff=0.0,       # placeholder; see ``ocv_entropy_a/b``
        runaway_c=210.0,         # degC adiabatic exotherm onset (project brief)
        nominal_capacity_ah=4.85,
        energy_density_wh_kg=255.0,
        coolant_props=dict(_GLYCOL_COOLANT),
        notes=(
            "High energy density, steep OCV slope in the 3.4-3.9 V window, and the "
            "earliest onset of thermal runaway of the two supported chemistries. Requires "
            "the higher coolant-flow set-point and the more aggressive thermal ramp."
        ),
    ),
    "lfp": ChemistrySpec(
        key="lfp",
        label="LFP (Lithium Iron Phosphate)",
        cathodes="LiFePO_4 (olivine)",
        anode="Graphite (C_6)",
        cp=1150.0,               # J/(kg*K) -- specified by the project brief
        k_solid=1.35,            # less graphitic content -> higher conductivity
        density=2280.0,          # kg/m^3
        r_int_ref=0.0182,        # ohm @ 25 degC, 50% SOC
        ea=19800.0,              # J/mol -- flatter Arrhenius rise
        entropy_coeff=0.0,
        runaway_c=270.0,         # degC adiabatic exotherm onset (project brief)
        nominal_capacity_ah=5.20,
        energy_density_wh_kg=165.0,
        coolant_props=dict(_GLYCOL_COOLANT),
        notes=(
            "Flatter OCV curve and higher thermal-runaway onset grant a substantially "
            "larger thermal budget, at the cost of ~35% lower gravimetric energy density "
            "and higher ohmic resistance at high SOC."
        ),
    ),
}

DEFAULT_CHEMISTRY: Final[str] = "nmc"
SUPPORTED_CHEMISTRIES: Final[tuple[str, ...]] = tuple(CHEMISTRY_REGISTRY.keys())

ChemistryKey = Literal["nmc", "lfp"]


def get_chemistry(key: str | None) -> ChemistrySpec:
    """Resolve a chemistry key to its :class:`ChemistrySpec`.

    Raises
    ------
    KeyError
        If ``key`` is not a registered chemistry.  Surfaced by the FastAPI
        error handler as HTTP 422.
    """
    resolved = (key or DEFAULT_CHEMISTRY).lower().strip()
    if resolved not in CHEMISTRY_REGISTRY:
        raise KeyError(
            f"Unknown chemistry {key!r}. Supported: {sorted(CHEMISTRY_REGISTRY)}"
        )
    return CHEMISTRY_REGISTRY[resolved]


# ---------------------------------------------------------------------------
# 4. Microchannel cold-plate geometry and solver defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColdPlateSpec:
    """Geometry of the serpentine aluminium cold plate under the pack.

    ``n_channels`` parallel microchannels of width ``channel_width_m`` and depth
    ``channel_depth_m`` traverse the pack in x.  Heat is conducted from the cells
    through a 2 mm bonded interface into the plate, then convected into the
    coolant.
    """

    length_m: float = 0.360
    width_m: float = 0.230
    thickness_m: float = 0.012
    n_channels: int = 24
    channel_width_m: float = 0.0022
    channel_depth_m: float = 0.0035
    interface_thickness_m: float = 0.0020
    n_cells: int = 12
    rows: int = 3

    @property
    def fluid_area_m2(self) -> float:
        """Total microchannel cross-sectional flow area, m^2."""
        return self.n_channels * self.channel_width_m * self.channel_depth_m

    @property
    def wetted_perimeter_m(self) -> float:
        """Total wetted perimeter of all channels, m."""
        return self.n_channels * 2.0 * (self.channel_width_m + self.channel_depth_m)

    @property
    def hydraulic_diameter_m(self) -> float:
        """Hydraulic diameter ``4A/P`` of the channel bundle, m."""
        return 4.0 * self.fluid_area_m2 / self.wetted_perimeter_m

    @property
    def conductance_m2k(self) -> float:
        """Cell-to-coolant footprint area, m^2 (the convective reference area)."""
        return self.length_m * self.width_m

    @property
    def wet_footprint(self) -> tuple[float, float, float]:
        """Return ``(L, W, H)`` of the fluid domain in metres."""
        return self.length_m, self.width_m, self.thickness_m


COLD_PLATE: Final[ColdPlateSpec] = ColdPlateSpec()

#: Maximum continuously addressable coolant mass flow, L/min.
MAX_COOLANT_FLOW_LPM: Final[float] = 14.0
#: Minimum (bypass) coolant mass flow, L/min.
MIN_COOLANT_FLOW_LPM: Final[float] = 0.8
#: Thermal expansion coefficient of the 50/50 glycol coolant, 1/K.
COOLANT_BETA: Final[float] = 3.4e-4
#: Heat capacity of the pump/heatsink mass seen by the loop, J/K.
LOOP_HEAT_CAPACITY_J_PER_K: Final[float] = 48_000.0

#: Default cooling-loop temperature set-point, degC.
COOLANT_SETPOINT_C: Final[float] = 28.0
#: Upper bound on cell temperature set-point used by the controller, degC.
CELL_TEMPERATURE_CEILING_C: Final[float] = 45.0
#: Hard safety trip used by the fail-safe monitor (min over chemistries).
CRITICAL_TEMPERATURE_C: Final[float] = 210.0

#: Inference latency budget enforced in the model card, milliseconds.
INFERENCE_LATENCY_BUDGET_MS: Final[float] = 800.0

#: Anticipatory heatwave look-ahead horizon, seconds.
HEATWAVE_LOOKAHEAD_S: Final[float] = 240.0
#: Lead time at which the anticipatory controller begins to ramp, seconds.
HEATWAVE_RAMP_LEAD_S: Final[float] = 180.0


# ---------------------------------------------------------------------------
# 5. PINN / solver defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PINNConfig:
    """Hyper-parameters of the dual-chemistry temperature-field PINN."""

    #: Trunk network width (Fourier-feature-augmented SIREN).
    hidden_dim: int = 128
    #: Depth of the trunk MLP (number of hidden layers).
    n_hidden_layers: int = 5
    #: Width of the chemistry-conditioning branch.
    chem_dim: int = 32
    #: Number of random Fourier features. Each contributes a sine and a cosine,
    #: so the trunk input width is ``2 * n_fourier_features`` -- see
    #: :attr:`trunk_input_dim`.
    n_fourier_features: int = 48
    #: Gaussian frequency scale of the Fourier draw, ``B ~ N(0, sigma^2)``.
    #:
    #: This governs the *frequency band*, and it is the single most
    #: consequential number in the network after the trunk depth. The Fourier
    #: term is ``sin(2*pi*sigma*x@B)``, so the network's second derivative --
    #: which is precisely what the physics residual differentiates -- scales as
    #: ``k^2`` with ``k = 2*pi*sigma*b``. Doubling ``sigma`` quadruples the
    #: Laplacian, and squares the residual the loss reports, by sixteen.
    #:
    #: It must be **independent of** :attr:`n_fourier_features`. Tying the two
    #: together (``sigma = n_features / 6``, which is what this did) means every
    #: capacity increase also sharpens the frequency band, so the initial
    #: residual grows as ``n^2`` instead of the model gaining resolution. At 48
    #: features that put the initial Laplacian at ~5e8 K/m^2 -- twenty-five
    #: thousand times the ~2e4 W/m^3 physical source term -- and the
    #: gradient-norm balancer consequently assigned the PDE a weight of 1.0 and
    #: every other term 1e-3, after which the loss could not move.
    #:
    #: ``sigma = 1.0`` spans the unit cube with a handful of cycles, the right
    #: band for a plate whose longest dimension is 0.36 m. More features then
    #: sample that same band more densely, which is the intended meaning of
    #: adding capacity.
    fourier_sigma: float = 1.0
    #: Input dimensionality of the trunk: ``(x, y, z, t)``.
    condition_dim: int = 4
    #: SIREN first-layer w0.
    w0: float = 12.0
    #: SIREN hidden-layer w0.
    hidden_w0: float = 4.5
    #: Number of collocation points per boundary condition face.
    n_boundary_samples: int = 512
    #: Number of interior PDE collocation points per sample.
    n_collocation_samples: int = 2048
    #: LR for the Adam stage.
    lr: float = 3.0e-4
    #: Exponential decay rate applied to ``lr`` at every epoch.
    lr_gamma: float = 1.0e-5
    #: EMA decay for the adaptive loss weights.
    weight_ema: float = 0.999
    #: Gradient-norm clipping threshold.
    grad_clip: float = 5.0
    #: Total epochs for the Adam stage.
    epochs: int = 4000
    #: Epochs for the L-BFGS-B refinement stage.
    lbfgs_epochs: int = 320
    #: Loss weights of the operator (multiplicative prior to adaptation).
    w_physics_init: float = 1.0
    w_boundary: float = 10.0
    w_initial: float = 10.0
    w_data: float = 25.0
    #: Adam betas.
    beta1: float = 0.9
    beta2: float = 0.999
    #: Seeds for reproducibility of the research runs.
    seed: int = 1337

    @property
    def trunk_input_dim(self) -> int:
        """Width of the Fourier-expanded trunk input, ``2 * n_fourier_features``.

        ``RandomFourierFeatures`` emits a sine *and* a cosine per frequency, so
        the factor of two is required.  Note the input axes are already mixed by
        the Gaussian draw ``B`` before the trigonometric embedding, so the width
        does not scale with ``condition_dim``.
        """
        return 2 * self.n_fourier_features

    @property
    def n_trunk_params_upper_bound(self) -> int:
        """Analytic parameter count used for the edge-deployment budget check."""
        d_in = self.trunk_input_dim
        per_layer = d_in * self.hidden_dim + self.hidden_dim          # first layer
        rest = (self.n_hidden_layers - 1) * (self.hidden_dim**2 + self.hidden_dim)
        chem = self.chem_dim * 2 * self.hidden_dim + 2 * self.chem_dim   # FiLM
        head = (2 * self.hidden_dim + self.chem_dim) * 1 + 1
        return per_layer + rest + chem + head


PINN_CONFIG: Final[PINNConfig] = PINNConfig()


# ---------------------------------------------------------------------------
# 6. API / CORS
# ---------------------------------------------------------------------------

API_TITLE: Final[str] = "Heatwave-Aware Microchannel BTMS API"
API_VERSION: Final[str] = "1.0.0"
CORS_ORIGINS: Final[tuple[str, ...]] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
)
API_HOST: Final[str] = os.getenv("API_HOST", "0.0.0.0")
API_PORT: Final[int] = int(os.getenv("API_PORT", "8000"))


def build_chemistry_conditioning_vector(spec: ChemistrySpec) -> list[float]:
    """Return the normalised conditioning vector fed to the PINN's chem branch.

    The vector is deliberately log-scaled for the resistance term so that NMC and
    LFP occupy comparable regions of conditioning space; without this, the LFP
    branch is starved of gradient early in training.

    Returns
    -------
    list of float
        12-dimensional vector, order documented in :mod:`backend.models.pinn`.
    """
    import math as _m

    return [
        spec.cp / 1200.0,
        spec.k_solid / 1.5,
        spec.density / 2600.0,
        _m.log10(spec.r_int_ref / 0.01),
        spec.ea / 30000.0,
        spec.runaway_c / 300.0,
        spec.nominal_capacity_ah / 6.0,
        spec.energy_density_wh_kg / 300.0,
        spec.coolant_props["cp"] / 4000.0,
        spec.coolant_props["k"] / 0.6,
        spec.coolant_props["mu"] / 0.004,
        spec.coolant_props["density"] / 1100.0,
    ]


CHEMISTRY_CONDITIONING_DIM: Final[int] = 12