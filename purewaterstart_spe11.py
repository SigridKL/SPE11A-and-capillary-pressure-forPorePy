"""Two-phase (H2O/CO2) flow with capillary pressure on the SPE11 case A domain.

This script is a copy of ``purewaterstart.py`` where the synthetic two-sand-layer
domain (``TwoSandLayers``) is replaced by the SPE11 case A geometry and materials:

- Geometry: ``Modifiedgeometry`` + ``InactiveCellsMixin`` from ``SPE11A_geometry``
  (mesh from ``spe11_data/spe11a.geo``, facies 7 deactivated).
- Porosity/permeability: ``porositymixin``/``permeabilitymixin`` from
  ``SPE11A_materials`` (cellwise values per facies from ``case_A``).
- Entry pressure: per facies from ``case_A["ENTRY_PRESSURE"]`` (``SPE11CapillaryPressure``
  below), instead of the lower/upper layer values used in ``purewaterstart.py``.
- Wells: positions and size from ``case_A``.

Everything else (fluid, viscosity, boundary conditions, constitutive laws, solver
settings and plotting) is kept as close to ``purewaterstart.py`` as possible.
"""

from __future__ import annotations

import pathlib

# import shutil
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Sequence, cast

import matplotlib.pyplot as plt
import meshio
import numpy as np
from capillary_pressure import CapillaryPressure
from DarcysCapillaryPressure import (
    AdvectiveFluxCapillaryPressure,
    DarcysLawCapillaryPressure,
)
from Fluid_and_component_flux import (
    CapillaryComponentMassBalanceEquations,
    CapillaryFluidMassBalanceEquations,
)
from frapeter_source import Modifiedsource
from frapeter_source import well_cell_id
from matplotlib.collections import PolyCollection
from phase_potential_upwinding import PhasePotentialUpwinding
from porepy.compositional.base import Phase
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from porepy.viz.data_saving_model_mixin import IterationExporting
from SPE11A_caseAdata import case_A
from SPE11A_geometry import InactiveCellsMixin, Modifiedgeometry
from SPE11A_materials import permeabilitymixin, porositymixin

import porepy as pp

if TYPE_CHECKING:
    from porepy.compositional.base import DomainFunctionType

dirname = pathlib.Path(__file__).parent.resolve()
output_root = dirname / "visualization_spe11a_capillary_runs"

# Facies tags present in the SPE11 mesh (facies 7 is inactive, but the tag exists).
FACIES_NAMES = [f"Facies {i}" for i in range(1, 8)]


# region Fluid components and phases, and their properties.
class MyFluid(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        return [
            # Some custom values for a later section.
            pp.FluidComponent(
                name="H2O",
                viscosity=float(self.params["viscosity_h2o"]),
                density=float(self.params["density_h2o"]),
            ),
            pp.FluidComponent(
                name="CO2",
                viscosity=float(self.params["viscosity_co2"]),
                density=float(self.params["density_co2"]),
            ),
        ]

    def z_co2_from_gas_saturation(
        self, s_gas: float | np.ndarray
    ) -> float | np.ndarray:
        """Overall CO2 mass fraction for a gas saturation, with pure phases.

        Inverse of :meth:`MyCompositionalFlowModel.saturation` for zero solubilities.

        """
        rho_w = float(self.params["density_h2o"])
        rho_n = float(self.params["density_co2"])
        return s_gas * rho_n / (s_gas * rho_n + (1.0 - s_gas) * rho_w)

    def set_components_in_phases(
        self, components: Sequence[pp.Component], phases: Sequence[pp.Phase]
    ) -> None:
        h2o, co2 = components
        aqu, gas = phases

        aqu.components = [h2o, co2]
        gas.components = [h2o, co2]

    def get_phase_configuration(
        self, components: Sequence[pp.compositional.ComponentLike]
    ) -> Sequence[tuple[pp.compositional.PhysicalState, str]]:
        # No EoS objects: phase properties come from constitutive laws.
        return [
            (pp.compositional.PhysicalState.liquid, "aqueous"),
            (pp.compositional.PhysicalState.gas, "gaseous"),
        ]


class MyViscosity(pp.PorePyModel):
    def viscosity_of_phase(
        self, phase: pp.Phase[pp.FluidComponent]
    ) -> pp.compositional.compositional_mixins.ExtendedDomainFunctionType:
        aqu, gas = self.fluid.phases
        h2o, co2 = self.fluid.components

        def mu(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            # Per SPE11: mixture properties equal pure-phase values.
            if phase == aqu:
                return pp.ad.Scalar(h2o.viscosity)
            elif phase == gas:
                return pp.ad.Scalar(co2.viscosity)
            raise ValueError(f"Unknown phase: {phase.name}")

        return mu

    def density_of_phase(
        self, phase: pp.Phase[pp.FluidComponent]
    ) -> pp.compositional.compositional_mixins.ExtendedDomainFunctionType:
        aqu, gas = self.fluid.phases
        h2o, co2 = self.fluid.components

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            # Per SPE11: mixture properties equal pure-phase values.
            if phase == aqu:
                return pp.ad.Scalar(h2o.density)
            elif phase == gas:
                return pp.ad.Scalar(co2.density)
            raise ValueError(f"Unknown phase: {phase.name}")

        return rho


# endregion


# region SPE11 facies helpers and entry pressure
def facies_values(
    domain: pp.Grid | pp.BoundaryGrid, values_per_facies: dict[str, float]
) -> np.ndarray:
    """Cellwise array built from a ``{"Facies i": value}`` dictionary and the facies
    tags of the grid.

    For a boundary grid (which has no tags), each boundary cell gets the value of
    the subdomain cell adjacent to the corresponding boundary face.
    """
    if isinstance(domain, pp.BoundaryGrid):
        sd = domain.parent
        cell_values = facies_values(sd, values_per_facies)
        # Boundary cell -> parent face (one face per row of the projection).
        faces = domain.projection().tocsr().indices
        # Boundary face -> the single adjacent cell.
        cells = sd.cell_faces.tocsr()[faces].indices
        return cell_values[cells]

    values = np.zeros(domain.num_cells)
    for facies, value in values_per_facies.items():
        if facies in domain.tags:
            values[domain.tags[facies]] = float(value)
    return values


def facies_index(sd: pp.Grid) -> np.ndarray:
    """Cellwise facies number (1-7) from the facies tags of the grid."""
    index = np.zeros(sd.num_cells, dtype=int)
    for number, facies in enumerate(FACIES_NAMES, start=1):
        if facies in sd.tags:
            index[sd.tags[facies]] = number
    return index


class SPE11CapillaryPressure(CapillaryPressure):
    """Brooks-Corey capillary pressure with facies-wise entry pressure.

    ``CapillaryPressure.entry_pressure`` assumes the two-sand-layer domain
    (``øvre_lag`` and ``p_entry_lower``/``p_entry_upper``); here the entry pressure
    is instead read per facies from ``params["entry_pressure"]`` (default
    ``case_A["ENTRY_PRESSURE"]``). ``capillary_pressure`` and ``data_to_export`` are
    inherited unchanged.
    """

    def entry_pressure(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        size = sum(domain.num_cells for domain in domains)
        entry_per_facies = self.params.get("entry_pressure", case_A["ENTRY_PRESSURE"])

        values = np.zeros(size)
        start = 0
        for domain in domains:
            values[start : start + domain.num_cells] = facies_values(
                domain, entry_per_facies
            )
            start += domain.num_cells

        return pp.wrap_as_dense_ad_array(values, size, name="entry_pressure")


# endregion


# region Boundary conditions
class ModifiedBoundaryConditions(BoundaryConditionsSinglePhaseFlow):
    """Dirichlet condition at the top (pressure and overall fraction), no-flow on the
    other external boundaries.

    The phase-potential fluxes use the same boundary types; on the Dirichlet faces the
    water pressure is evaluated as pressure minus the capillary pressure of the
    boundary state.
    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet at the top, Neumann elsewhere."""
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.north, "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition for the advective phase fluxes."""
        return self.bc_type_darcy_flux(sd)

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero flux on the Neumann boundaries."""
        return np.zeros(bg.num_cells)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Homogeneous pressure values at the top Dirichlet boundary; equals initial
        pressure.

        """
        values = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        p_top = float(self.params["initial_pressure"])
        values[sides.north] = self.units.convert_units(p_top, "Pa")
        return values

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        assert component.name == "CO2"
        x_aq = self.solubility_co2_in_water()
        x_gas = 1.0 - self.solubility_h2o_in_gas()
        target = self.z_co2_from_gas_saturation(float(self.params["s_gas_bc"]))
        values = np.full(bg.num_cells, target, dtype=float)

        # The overall CO2 fraction must remain in the physical interval implied by
        # the fixed phase compositions; otherwise the derived saturations become
        # invalid when inverted from z_CO2 = s_aq x_aq + s_gas x_gas.
        values = np.clip(values, x_aq, x_gas)
        return values


# endregion


class ConstitutiveLaws(  # type: ignore[misc]
    # General constitutive laws required for flow & transport.
    pp.compositional_flow.ConstitutiveLawsSolidSkeletonCF,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.GravityForce,  # skrur på tyngdekraft
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.DimensionReduction,
    # pp.constitutive_laws.AdvectiveFlux, bytter ut med min:
    AdvectiveFluxCapillaryPressure,
    pp.constitutive_laws.DisplacementJump,  # chat la til denne med begrunnelse: lagt til kun for å stilne en tidligere abstrakt-typeklage. For en ren flowmodell er den ofte unødvendig fysisk, men den skader ikke nødvendigvis kjøring.
    # pp.constitutive_laws.DarcysLaw, bytter ut med min:
    DarcysLawCapillaryPressure,
    pp.constitutive_laws.PeacemanWellFlux,
    # NOTE IMPORTANT: Due to the EoS being used, we must override even those properties
    # with constitutive laws, which are not formally used by the model.
    # This tells the model to disregard the computations of phase properties by the
    # EoS, for which there already are mixed in constitutive laws.
):
    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Brooks-Corey relative permeability with immobile saturation correction."""
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        sat = phase.saturation(domains)

        if phase.name == "aqueous":
            s_imm = float(self.params["sw_imm"])
            c_exp = float(self.params["c_w1"])
        elif phase.name == "gaseous":
            s_imm = float(self.params["sn_imm"])
            c_exp = float(self.params["c_n1"])
        else:
            raise ValueError(f"Unknown phase for relative permeability: {phase.name}")

        sw_imm = float(self.params["sw_imm"])
        sn_imm = float(self.params["sn_imm"])
        mobile_saturation_interval = max(1.0 - sw_imm - sn_imm, 1e-12)
        sat_norm = (sat - pp.ad.Scalar(s_imm)) / pp.ad.Scalar(
            mobile_saturation_interval
        )
        sat_eff = f_max(sat_norm, pp.ad.Scalar(0.0))

        kr = sat_eff ** pp.ad.Scalar(c_exp)
        kr.set_name(f"kr_{phase.name}")
        maximum_func = pp.ad.Function(
            partial(
                pp.ad.functions.maximum,
                # If we set this too high, the phases retain mobility at immobile
                # saturation and may continue draining lower.
                var_1=0.0,
            ),
            "max",
        )
        kr = pp.ad.Scalar(1.0) - maximum_func(pp.ad.Scalar(1.0) - maximum_func(kr))
        return kr


class MyCompositionalFlowModel(  # type: ignore[misc]
    IterationExporting,
    # Phase-potential upwinding of the phase weights. Must come before the solution
    # strategy (extends update_flux_values and set_nonlinear_discretizations).
    PhasePotentialUpwinding,
    Modifiedgeometry,
    ModifiedBoundaryConditions,
    Modifiedsource,
    # SPE11 materials and geometry (replaces TwoSandLayers).
    porositymixin,
    permeabilitymixin,
    InactiveCellsMixin,
    SPE11CapillaryPressure,
    # Required constitutive laws.
    ConstitutiveLaws,
    # MyViscosity above MyVerySpecialFluid.
    MyViscosity,
    MyFluid,
    # Variables for flow & transport.
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.compositional.CompositionalVariables,
    # Equations for flow & transport.
    CapillaryComponentMassBalanceEquations,
    CapillaryFluidMassBalanceEquations,
    # CapillaryMassicPressureEquations,
    # For BC, IC and solution strategy, we need the mixins for phase properties, which
    # are represented by SurrogateFactories and EoS. We also need those for fractions in
    # the multicomponent setting, and the ones for single phase flow, which is takes
    # care of anything pressure related.
    # pp.compositional_flow.BoundaryConditionsPhaseProperties,
    pp.compositional_flow.BoundaryConditionsMulticomponent,
    # pp.compositional_flow.InitialConditionsPhaseProperties,
    pp.compositional_flow.InitialConditionsFractions,
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    # pp.compositional_flow.SolutionStrategyPhaseProperties,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    def prepare_simulation(self) -> None:
        """Saves well cell indices in self.well_cells, as these are needed in both
        fluid_source and component_source (see frapeter_source.Modifiedsource).

        The wells are the small squares from SPE11 case A (``case_A["WELL_SIZE"]``,
        ``case_A["WELL_1_POS"]`` and ``case_A["WELL_2_POS"]``) unless overridden by
        ``params["well_size"]`` / ``params["well_positions"]``.
        """
        self.set_geometry()
        sd = self.mdg.subdomains()[0]
        well_size = float(self.params["well_size"])

        well_cells: list[int] = []
        for well_pos_x, well_pos_y in self.params["well_positions"]:
            corners = np.array(
                [
                    [well_pos_x - well_size / 2, well_pos_y - well_size / 2],
                    [well_pos_x + well_size / 2, well_pos_y + well_size / 2],
                ]
            )
            ids = well_cell_id(sd, corners).flatten().tolist()
            if len(ids) == 0:
                raise ValueError(
                    f"No cells found inside the well at ({well_pos_x}, {well_pos_y}) "
                    f"with size {well_size}. Increase params['well_size']."
                )
            print(f"   Well at ({well_pos_x}, {well_pos_y}): {len(ids)} cells")
            well_cells.extend(ids)
        self.well_cells = well_cells

        super().prepare_simulation()

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
        """TPFA instead of PorePy's default MPFA for all Darcy and phase-potential
        fluxes."""
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

    def has_independent_partial_fraction(
        self, component: pp.Component, phase: pp.Phase
    ) -> bool:
        """Partial fractions are dependent (derived from solubilities)."""
        return False

    def has_independent_saturation(self, phase: pp.Phase) -> bool:
        """Saturation is derived from overall CO2 fraction for both phases."""
        return False

    def saturation(self, phase: Phase) -> DomainFunctionType:
        """Return a callable giving phase saturation from the overall CO2 fraction.

        PorePy expects `saturation(phase)` to return a function taking `domains`.
        The returned callable inverts

            z_CO2 = (s_gas rho_gas x_CO2_gas + s_aq rho_aq x_CO2_aq) / (s_gas rho_gas +
            s_aq rho_aq)

        using the fixed phase compositions implied by the solubilities.
        """
        h2o, co2 = self.fluid.components
        x_co2_aq = float(self.solubility_co2_in_water())
        x_co2_gas = float(1.0 - self.solubility_h2o_in_gas())
        rho_aq = float(h2o.density)
        rho_gas = float(co2.density)

        def s_func(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            z_co2 = co2.fraction(domains)
            # Mass of each phase per unit total mass, up to the common denominator.
            aq_part = pp.ad.Scalar(rho_gas) * (pp.ad.Scalar(x_co2_gas) - z_co2)
            gas_part = pp.ad.Scalar(rho_aq) * (z_co2 - pp.ad.Scalar(x_co2_aq))
            nevn = aq_part + gas_part + pp.ad.Scalar(1e-12)

            if phase.name == "aqueous":
                s = aq_part / nevn
            elif phase.name == "gaseous":
                s = gas_part / nevn
            else:
                raise ValueError(f"Unknown phase: {phase.name}")

            max0 = pp.ad.Function(partial(pp.ad.functions.maximum, var_1=0.0), "max0")
            max1 = pp.ad.Function(partial(pp.ad.functions.maximum, var_1=0.0), "max1")
            s = max0(s)
            s = pp.ad.Scalar(1.0) - max1(pp.ad.Scalar(1.0) - s)
            return s

        return s_func

    def partial_fraction(
        self, component: pp.Component, phase: pp.Phase
    ) -> pp.ad.Operator:
        """Return dependent partial fractions from solubilities.

        The aqueous phase is pure water and the gaseous phase is pure CO2.
        """
        if phase.name == "aqueous":
            if component.name == "CO2":
                return lambda domains: pp.ad.Scalar(0.0)
            else:
                return lambda domains: pp.ad.Scalar(1.0)
        else:
            if component.name == "H2O":
                return lambda domains: pp.ad.Scalar(0.0)
            else:
                return lambda domains: pp.ad.Scalar(1.0)

    def solubility_co2_in_water(self, pressure: pp.ad.Operator | None = None) -> float:
        """Return the CO2 fraction in the aqueous phase."""
        return 0.0

    def solubility_h2o_in_gas(self, pressure: pp.ad.Operator | None = None) -> float:
        """Return the H2O fraction in the gaseous phase."""
        return 0.0

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Hydrostatic initial pressure, consistent with the initial CO2 distribution.

        Uses the mixture density ``rho = s_gas rho_gas + s_aq rho_aq`` of the uniform
        initial gas saturation ``s_gas_init`` (the SPE11 domain has no lower/upper
        layer split, so a single value is used for the whole domain).

        Note:
            This is still only density-consistent, not capillary-pressure-consistent:
            it does not offset ``p_n``/``p_w`` by the initial ``p_cap(s_gas)``.
        """
        p_top = self.units.convert_units(
            float(self.params["initial_pressure"]),
            "Pa",
        )
        y_max = float(self.params["domain_height"])
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")

        h2o, co2 = self.fluid.components
        rho_w = float(h2o.density)
        rho_n = float(co2.density)

        s_init = float(self.params["s_gas_init"])
        rho_mix = (1.0 - s_init) * rho_w + s_init * rho_n

        y = sd.cell_centers[1, :]
        return p_top + rho_mix * g * (y_max - y)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        assert component.name == "CO2"

        z_init = self.z_co2_from_gas_saturation(float(self.params["s_gas_init"]))

        x_aq = self.solubility_co2_in_water()
        x_gas = 1.0 - self.solubility_h2o_in_gas()

        z = np.full(sd.num_cells, z_init, dtype=float)
        z = np.clip(z, x_aq, x_gas)

        return z

    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()
        """Set initial values for the independent phase-composition variables."""

        for sd in self.mdg.subdomains():
            # NOTE This doesn't do anything currently (no independent partial
            # fractions) and could cause confusion.
            x_co2_aq = np.zeros(sd.num_cells)
            x_co2_gas = np.ones(sd.num_cells)

            for phase in self.fluid.phases:
                for component in phase:
                    if self.has_independent_partial_fraction(component, phase):
                        init_value = x_co2_aq if phase.name == "aqueous" else x_co2_gas
                        self.equation_system.set_variable_values(
                            init_value,
                            [
                                cast(
                                    pp.ad.Variable,
                                    phase.partial_fraction_of[component]([sd]),
                                )
                            ],
                            iterate_index=0,
                        )

    def after_nonlinear_iteration(
        self, nonlinear_increment: np.ndarray, updated_variables=None
    ) -> None:
        """Clip the overall CO2 fraction to the physical interval implied by the fixed
        phase compositions and residual saturations.

        """
        super().after_nonlinear_iteration(nonlinear_increment)

        if not self.params.get("enable_capillary_pressure", False):
            return

        # Clip z_CO2 by variable identity, not by assuming a global unknown ordering.
        z_variables = self.equation_system.get_variables(["z_CO2"])
        z_values = self.equation_system.get_variable_values(
            z_variables, iterate_index=0
        )
        clipped_values = np.clip(
            z_values,
            self.solubility_co2_in_water(),
            1.0 - self.solubility_h2o_in_gas(),
        )
        if not np.array_equal(z_values, clipped_values):
            self.equation_system.set_variable_values(
                clipped_values,
                z_variables,
                iterate_index=0,
            )
            self.update_derived_quantities()

    def data_to_export(self):
        data = super().data_to_export()

        for sd in self.mdg.subdomains():
            porosity = facies_values(sd, case_A["POROSITY"])
            permeability = facies_values(sd, case_A["PERMEABILITY"])
            facies = facies_index(sd)
            co2_well_source = np.zeros(sd.num_cells)
            co2_well_source[self.well_cells] = (
                float(self.params["injection_rate_kg"])
                * float(self.params["injection_co2_fraction"])
                / len(self.well_cells)
            )

            data.append((sd, "porosity", porosity))
            data.append((sd, "permeability", permeability))
            data.append((sd, "facies", facies))
            data.append((sd, "co2_well_source", co2_well_source))

        return data


time_manager = pp.TimeManager(
    schedule=[0.0, 300 * pp.SECOND],
    dt_init=3.125 * pp.SECOND,
    dt_min_max=(10e-3 * pp.SECOND, 5000000.0 * pp.SECOND),
    constant_dt=False,
    iter_max=50,
)

# shutil.rmtree(output_folder, ignore_errors=True)
# output_folder.mkdir(parents=True, exist_ok=True)


params = {
    # Use the pypardiso linear solver to speed up the solution of the linear system.
    "linear_solver": "pypardiso",
    # Geometry parameters. The mesh is read from spe11_data/spe11a.geo by
    # InactiveCellsMixin, so no meshing_arguments are needed.
    "domain_width": case_A["WIDTH"],
    "domain_height": case_A["HEIGHT"],
    "initial_pressure": case_A["INITIAL_PRESSURE"],
    # Solid and fluid parameters.
    "material_constants": {},
    # Fluid properties from NIST: density in kg/m^3, viscosity in Pa*s.
    "density_h2o": 998.21,
    "viscosity_h2o": 0.0010016,
    "density_co2": 1.8152,
    "viscosity_co2": 1.4675e-05,
    # Porosity and permeability per facies are taken from case_A in
    # SPE11A_materials (porositymixin/permeabilitymixin).
    # Wells: both SPE11 case A wells inject from t=0 (in the benchmark well 2 starts
    # later). Remove an entry to use a single well. injection_rate_kg is the total
    # rate, split evenly over all well cells (see frapeter_source.Modifiedsource).
    "well_size": case_A["WELL_SIZE"],
    "well_positions": [case_A["WELL_1_POS"], case_A["WELL_2_POS"]],
    "injection_rate_kg": 1e-5,  # SPE11 case A: 1.7e-7 kg/s per well
    "injection_co2_fraction": 1.0,
    # If true, the water phase is driven by pressure - capillary pressure; if false,
    # both phases are driven by the pressure variable.
    "enable_capillary_pressure": True,
    "fractional_flow": False,
    # Gas saturations; converted to the overall CO2 mass fraction z_CO2 in the model.
    "s_gas_bc": 0.2,
    "s_gas_init": 0.2,  # Water-rich domain at start (uniform)
    "x_co2_aq": 0.0,  #  CO2 in water (SPE11 value)
    "x_h2o_gas": 0.0,  #  H2O mole fraction in gas (SPE11 value)
    "sw_imm": 0.2,  #  Immobile water saturation (residual)
    "sn_imm": 0.2,  #  Immobile gas saturation (residual)
    # SPE11CapillaryPressure bruker disse verdiene til å lage cellwise
    # inngangstrykk per facies. Kopier og endre for å teste andre verdier.
    "entry_pressure": dict(case_A["ENTRY_PRESSURE"]),
    "c2": 2.0,
    "pcap_max": case_A["MAX_CAP_PRESS"],
    "c_w1": 2.0,  # Aqueous-phase relative-permeability exponent
    "c_n1": 2.0,  # Gaseous-phase relative-permeability exponent
    # Simulation control parameters.
    "time_manager": time_manager,
    # "folder_name": output_folder,
    # "solver_statistics_file_name": "solver_statistics_no_inj_simple.json",
    "progressbars": True,
}

# Active facies (1-6) span the entry-pressure range used in the run label.
active_entry_pressures = [
    float(params["entry_pressure"][facies]) for facies in FACIES_NAMES[:6]
]
pressure_label = (
    f"spe11a_pe_min_{min(active_entry_pressures):g}"
    f"_max_{max(active_entry_pressures):g}"
)
run_label = f"{pressure_label}__{datetime.now():%Y%m%d_%H%M%S}"
output_folder = output_root / run_label
output_folder.mkdir(parents=True, exist_ok=False)
params["folder_name"] = output_folder
params["solver_statistics_file_name"] = "solver_statistics.json"

model = MyCompositionalFlowModel(params)  # type: ignore[abstract]


# Visualization
def create_result_figure() -> None:
    """Create a summary figure from the latest VTU result."""
    vtu_files = sorted(output_folder.glob("data_2_*.vtu"))
    if not vtu_files:
        print("No VTU files found for plotting.")
        return

    latest_vtu = vtu_files[-1]
    output_name = output_folder / "tpf_results_overview.png"

    # Edit these field names to change which variables are visualized.
    fields = [
        "pressure",
        "s_gaseous",
        "x_CO2_aqueous",
        "z_CO2",
    ]  # disse variablen er henholdsvis trykk, metning av gassfase, CO2 partial fraction i væskefase, og overall mass fraction av CO2. Sjekk at disse navnene stemmer med det som faktisk finnes i cell_data_dict i vtu-filene, ellers må de endres til det som faktisk finnes der for å få plottet noe.
    cmap = "viridis"

    mesh = meshio.read(str(latest_vtu))
    pts = mesh.points[:, :2]
    # The SPE11 mesh from gmsh is (mostly) triangles; also accept quads.
    cell_types = [cells.type for cells in mesh.cells if cells.type in ("triangle", "quad")]
    if not cell_types:
        print(f"No 2d cells in {latest_vtu.name}; skipping figure.")
        return

    polys = [
        pts[cell]
        for cells in mesh.cells
        if cells.type in cell_types
        for cell in cells.data
    ]
    available = set(mesh.cell_data_dict.keys())
    selected = [f for f in fields if f in available]
    if not selected:
        print(
            f"No requested fields found in {latest_vtu.name}. Available: {sorted(available)}"
        )
        return

    n = len(selected)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 4.5 * nrows), constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    for ax, name in zip(axes, selected):
        vals = np.concatenate(
            [np.asarray(mesh.cell_data_dict[name][cell_type]) for cell_type in cell_types]
        )
        # z_CO2 should be in [0, 1]. Clipping avoids outlier-dominated color scales
        # and makes spatial transport patterns easier to inspect.
        if name == "z_CO2":
            vals = np.clip(vals, 0.0, 1.0)
        coll = PolyCollection(
            polys,
            array=vals,
            cmap=cmap,
            edgecolors="k",
            linewidths=0.05,
        )
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(coll, ax=ax, shrink=0.88)

    for ax in axes[len(selected) :]:
        ax.axis("off")

    fig.suptitle(f"TPF results at last saved step ({latest_vtu.name})")
    fig.savefig(output_name, dpi=220)
    plt.close(fig)
    print(f"Updated figure: {output_name}")


print("\n1. Preparing simulation...")
model.prepare_simulation()
print("   -> prepare_simulation() completed successfully!")

print("\n2. Creating equations and checking setup...")
sds = model.mdg.subdomains()
print(
    "Overall fraction is variable:",
    [
        (comp.name, isinstance(comp.fraction(sds), pp.ad.Variable))
        for comp in model.fluid.components
    ],
)
print(
    "Saturation is variable:",
    [
        (phase.name, isinstance(phase.saturation(sds), pp.ad.Variable))
        for phase in model.fluid.phases
    ],
)

for phase in model.fluid.phases:
    print("---")
    print(f"{phase.name} phase:")
    print(
        "Partial fraction is variable:",
        [
            (
                comp.name,
                isinstance(phase.partial_fraction_of[comp](sds), pp.ad.Variable),
            )
            for comp in phase
        ],
    )

print("\n3. Running simulation...")
print(
    "   Phase fluxes with phase-potential upwinding,",
    "capillary pressure enabled"
    if model.params.get("enable_capillary_pressure", False)
    else "capillary pressure disabled",
)
try:
    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            # "solver_statistics_file_name": "statistics",
            "nl_max_iterations": 20,
            "nl_convergence_inc_atol": 2e-7,
            "nl_convergence_inc_rtol": 1e-5,
            # "nl_convergence_res_atol": 1e-10,
            "nl_convergence_res_rtol": np.inf,
        },
    )  # prepare_simulation() er allerede kjørt ovenfor.
    # Inne i run_time_dependent_model er default at den også kjører model.prepare_simulation(),
    # (som vi allerede har kalt over her sjøl )hvis prepare_simulation ikke settes til False (se run_models.py:68).
    # kan evt fjerne mode.prepare_simulation() og la run_time_dependent_model gjøre alt sjøl.
    print("   -> Simulation completed successfully!")
except Exception as e:
    print(f"   -> ERROR during simulation: {e}")
    import traceback

    traceback.print_exc()

# Always regenerate a figure from the latest available result file.
create_result_figure()

import plot_solver_statistics

# Point the statistics plots at the SPE11 run folder instead of the two-layer runs.
plot_solver_statistics.RUNS_FOLDER = output_root
plot_solver_statistics.plot_statistics()
