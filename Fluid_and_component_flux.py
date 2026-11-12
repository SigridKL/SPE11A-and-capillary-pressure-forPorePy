"""Advective fluxes as sums of phase fluxes with phase-potential upwinding.

Each phase j is advected with its own phase-potential flux
``v_j = -K (grad p_j - rho_j g)`` (``darcy_flux(..., flow_type=phase.name)``), and its
weight is upwinded on the sign of the same flux
(``PhasePotentialUpwinding.phase_mobility_discretization``). Gravity and capillary
pressure are contained in ``v_j``, so there is no separate buoyancy or capillary term.

"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np

import porepy as pp


class CapillaryFluidMassBalanceEquations(
    pp.fluid_mass_balance.FluidMassBalanceEquations,
):
    """PorePy's total mass balance with the total flux as sum of phase fluxes."""

    def fluid_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
    ) -> pp.ad.Operator:
        """Return the total mass flux ``sum_j rho_j lambda_j v_j``."""
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return pp.ad.DenseArray(
                np.zeros(sum(d.num_cells for d in domains)),
                name="bc_values_fluid_flux_zero",
            )

        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)
        advective_flux = cast(Callable[..., pp.ad.Operator], self.advective_flux)

        flux = pp.ad.sum_operator_list(
            [
                advective_flux(
                    domains,
                    self.phase_mass_mobility(phase, domains),
                    self.phase_mobility_discretization(phase, domains),
                    self.boundary_phase_mass_mobility(phase, domains),
                    None,
                    flow_type=phase.name,
                )
                for phase in self.fluid.phases
            ]
        )
        flux.set_name("fluid_flux")
        return flux


class CapillaryComponentMassBalanceEquations(
    pp.compositional_flow.ComponentMassBalanceEquations,
):
    """PorePy's component balance with the component flux as sum of phase fluxes."""

    def component_flux(
        self,
        component: pp.Component,
        domains: pp.SubdomainsOrBoundaries,
    ) -> pp.ad.Operator:
        """Return the component mass flux ``sum_j chi_ij rho_j lambda_j v_j``."""
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return self.advection_weight_component_mass_balance(
                component, domains
            ) * self.darcy_flux(domains)

        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("Domains must consist entirely of subdomains.")
        domains = cast(list[pp.Grid], domains)
        advective_flux = cast(Callable[..., pp.ad.Operator], self.advective_flux)

        # NOTE phase_mass_mobility is weighted by the partial fraction of the component.
        # If the component is not present in a phase, it will evaluate to zero.
        flux = pp.ad.sum_operator_list(
            [
                advective_flux(
                    domains,
                    self.phase_mass_mobility(phase, domains, component),
                    self.phase_mobility_discretization(phase, domains),
                    self.boundary_phase_mass_mobility(phase, domains, component),
                    None,
                    flow_type=phase.name,
                )
                for phase in self.fluid.phases
            ]
        )
        flux.set_name(f"component_flux_{component.name}")
        return flux
