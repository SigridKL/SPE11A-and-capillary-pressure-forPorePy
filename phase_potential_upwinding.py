r"""Phase-potential upwinding (PPU).

The weight of phase j is upwinded on the sign of its phase-potential flux

.. math::

    v_j = -K (\nabla p_j - \rho_j g), \quad p_g = \texttt{pressure}, \quad p_w =
    \texttt{pressure} - p_c,

which is ``darcy_flux(..., flow_type=phase.name)``. The fluxes in
``Fluid_and_component_flux`` multiply each upwinded weight with the same ``v_j``, so a
change of upwind direction happens where the phase flux is zero, and the discrete fluxes
are continuous.

"""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence, cast

import numpy as np

import porepy as pp


class PhasePotentialUpwinding(pp.PorePyModel):
    """Upwind discretizations for each phase, based on the phase potentials."""

    darcy_flux: Callable[..., pp.ad.Operator]
    phase_mobility: Callable[[pp.Phase, pp.SubdomainsOrBoundaries], pp.ad.Operator]
    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]

    def phase_mobility_keyword(self, phase: pp.Phase) -> str:
        """Keyword for the phase-potential upwind discretization of a phase."""
        return f"ppu_{phase.name}"

    def phase_mobility_discretization(
        self, phase: pp.Phase, subdomains: list[pp.Grid]
    ) -> pp.ad.UpwindAd:
        """Upwind discretization of the weights of a phase, upwinded on the sign of
        the phase-potential flux."""
        return pp.ad.UpwindAd(self.phase_mobility_keyword(phase), subdomains)

    def phase_mass_mobility(
        self,
        phase: pp.Phase,
        domains: pp.SubdomainsOrBoundaries,
        component: pp.Component | None = None,
    ) -> pp.ad.Operator:
        r"""Mass mobility of a phase.

        .. math::
            \rho_j \lambda_j, \quad \text{or} \quad \chi_{ij} \rho_j \lambda_j

        if a component is given.

        """
        weight = phase.density(domains) * self.phase_mobility(phase, domains)
        if component is not None:
            weight = phase.partial_fraction_of[component](domains) * weight
        return weight

    def boundary_phase_mass_mobility(
        self,
        phase: pp.Phase,
        domains: Sequence[pp.Grid],
        component: pp.Component | None = None,
    ) -> pp.ad.Operator:
        """Boundary data for the advected phase weight.

        Dirichlet faces get :meth:`phase_mass_mobility` evaluated on the boundary
        (used for inflow). Neumann faces get zero, i.e. no prescribed advective
        in/outflux.

        """
        suffix = phase.name if component is None else f"{component.name}_{phase.name}"
        name = f"bc_values_phase_mass_mobility_{suffix}"

        def zero(boundary_grids: Sequence[pp.BoundaryGrid]) -> pp.ad.Operator:
            return pp.ad.DenseArray(
                np.zeros(sum(bg.num_cells for bg in boundary_grids)),
                name=name + "_zero",
            )

        return self._combine_boundary_operators(
            subdomains=domains,
            dirichlet_operator=cast(
                Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
                partial(self.phase_mass_mobility, phase, component=component),
            ),
            neumann_operator=zero,
            robin_operator=None,
            bc_type=self.bc_type_fluid_flux,
            name=name,
        )

    def update_discretization_parameters(self) -> None:
        """Set boundary types (and initial zero fluxes) for the PPU discretizations."""
        super().update_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            for phase in self.fluid.phases:
                key = self.phase_mobility_keyword(phase)
                params = {"bc": self.bc_type_fluid_flux(sd)}
                # Keep stored phase fluxes; only initialize them once.
                if "darcy_flux" not in data[pp.PARAMETERS].get(key, {}):
                    params["darcy_flux"] = np.zeros(sd.num_faces)
                pp.initialize_data(data, key, params)

    def update_flux_values(self) -> None:
        """Store the phase-potential fluxes for the PPU discretizations."""
        super().update_flux_values()

        # Evaluate each phase's flux for all subdomains jointly, then distribute the
        # face values to the individual subdomains.
        subdomains = self.mdg.subdomains()
        offsets = np.cumsum([0] + [sd.num_faces for sd in subdomains])
        for phase in self.fluid.phases:
            key = self.phase_mobility_keyword(phase)
            vals = self.equation_system.evaluate(
                self.darcy_flux(subdomains, phase.name)
            )
            for i, sd in enumerate(subdomains):
                data = self.mdg.subdomain_data(sd)
                data[pp.PARAMETERS][key].update(
                    {"darcy_flux": vals[offsets[i] : offsets[i + 1]]}
                )

    def set_nonlinear_discretizations(self) -> None:
        """Rediscretize the PPU upwinding every nonlinear iteration."""
        super().set_nonlinear_discretizations()
        for phase in self.fluid.phases:
            self.add_nonlinear_discretization(
                self.phase_mobility_discretization(
                    phase, self.mdg.subdomains()
                ).upwind(),
            )
