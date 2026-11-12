"""Darcy and advective fluxes for phase-potential upwinding.

``DarcysLawCapillaryPressure.darcy_flux`` takes a ``flow_type``:

- ``"nonwetting"`` (default): Darcy flux of the pressure variable with the fluid
  density in the gravity term, as in PorePy's ``DarcysLaw``. PorePy uses it internally
  (e.g. for the stored mobility fluxes and the pressure trace); it does not enter the
  balance equations of this model.
- a phase name (``"aqueous"``, ``"gaseous"``): the phase-potential flux
  ``-K (grad p_j - rho_j g)``, with ``p_gaseous = pressure`` and
  ``p_aqueous = pressure - capillary_pressure``.

``AdvectiveFluxCapillaryPressure.advective_flux`` passes ``flow_type`` on to
``darcy_flux``, so that each phase is advected with its own phase-potential flux.

"""

from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union, cast

import porepy as pp

"""Modifiserte konstitutive lover som skal funke for kapillærtrykk.

Sammenligna med den opprinnelige constitutive_laws.py-fila har denne
en utvida darcyflux og advectiveflux: I de to metodene er det lagt til en parameter som heter
flow_type. denne kan være enten nonwetting eller capillary. og har
nonwetting som standardverdi. modellene oppfører seg derfor som flr hvis det ikke oppgis neen annen flowtype.

PLAN:
Hvis flow_type=nonwetting brukes, beregnes Darcy-fluxen fra det vanlige trykket
gjennom self.pressure.
Hvis flowtype=capillary, bergnes fluksem fra cappress isteden gjennom funksjonen
self.capillary_pressure, som finnes i en egen fil capillary_pressure.py.

Det er 2 darcy-flux implementasjoner: 
DarcysLaw.darcy_flux og DarcysLawAd.darcy_flux. flowtypen er lagt til i begge disse.
Den er også lagt til i AdvectiveFlux.advective_flux. Denne metoden sender den valgte flow typen videre
til darcy_flux, slik at advective flux bruker riktig trykkpotensial. Selve
upwind-diskretiseringa og interface-fluxer er ellers uendra.

Ingen modeller bruker automatisk kapillartrykk. Modeller som ikke oppgir
flow_type, bruker fortsatt den vanlige nonwetting-trykkformuleringen. For å aktivere
kapillartrykk må fluxen kalles eksplisitt med flow_type="capillary".

Denne fila endrer derfor kun hvordan flowtype velges i Darcy- og
advective fluxene. Selve mobilitetsformuleringen for wetting og nonwetting phase må evt
implementeres videre i modellen som bruker disse fluxene og da må kildekoden lokaliseres først.


Det øvrige er imidlertid ikke testa da alle disse endringene er laga i ei ny fil (denne),
og ikke har blitt forsøkt brukt istedenfor den originale fila constitutive_laws.py.
Det er fordi jeg ikke veit om jeg kan flytte kildekoden inn i github enda, og fordi
jeg har to mapper med src (src og src2), så veit ikke hvilken jeg skal endre.
SPØR OM DETTE.
"""

Scalar = pp.ad.Scalar


class DarcysLawCapillaryPressure(pp.PorePyModel):
    """This class could be refactored to reuse for other diffusive fluxes, such as
    heat conduction. It's somewhat cumbersome, though, since potential, discretization,
    and boundary conditions all need to be passed around.

    I denne er interface_darcy_flux fra porepys originale constitutive_laws, DarcysLaw, bytta ut med capillary_pressure
    Ellers er metoden lik."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    capillary_pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Capillary-pressure potential, provided by a capillary-pressure mixin."""
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    normal_permeability: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Normal permeability. Normally defined in a mixin instance of :
    class:`~porepy.models.constitutive_laws.ConstantPermeability`.

    """
    darcy_keyword: str
    """Keyword used to identify the Darcy flux discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    bc_data_darcy_flux_key: str
    """See
    :attr:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow.
    bc_data_darcy_flux_key`.

    """
    bc_type_darcy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the Darcy flux. Normally
    provided by a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.

    """

    gravity_force: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]], Literal["fluid", "solid"]],
        pp.ad.Operator,
    ]
    """Gravity force. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.GravityForce` or
    :class:`~porepy.models.constitutive_laws.ZeroGravityForce`.

    """

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )
        # We know p is variable, since called on subdomains, not boundaries.
        p = cast(pp.ad.MixedDimensionalVariable, self.pressure(subdomains))

        boundary_operator = self.combine_boundary_operators_darcy_flux(
            subdomains=subdomains
        )

        pressure_trace = (
            discr.bound_pressure_cell() @ p
            + discr.bound_pressure_face()
            @ (
                projection.mortar_to_primary_int()
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.bound_pressure_face() @ boundary_operator
            + discr.bound_pressure_vector_source()
            @ self.vector_source_darcy_flux(subdomains)
        )
        return pressure_trace

    @pp.ad.cached_method
    def darcy_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
        flow_type: str = "nonwetting",
    ) -> pp.ad.Operator:
        """Discretization of Darcy's law.

        ``flow_type`` is ``"nonwetting"`` (Darcy flux of the pressure variable with the
        fluid density) or the name of a phase (phase-potential flux of that phase, see
        :meth:`phase_potential_flux`).

        Note:
            The fluid mobility is not included in the Darcy flux. This is because we
            discretize it with an upstream scheme. This means that the fluid mobility
            may have to be included when using the flux in a transport equation. The
            units of the Darcy flux are [m^2 Pa / s].

        Parameters:
            domains: List of domains where the Darcy flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """

        phase_names = [phase.name for phase in self.fluid.phases]
        if flow_type not in ["nonwetting"] + phase_names:
            raise ValueError(
                f"flow_type must be 'nonwetting' or a phase name {phase_names}."
            )

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(
                name=self.bc_data_darcy_flux_key,
                domains=cast(Sequence[pp.BoundaryGrid], domains),
            )

        # Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument `domains` should either be a list of grids or a list of
                boundary grids."""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        if flow_type != "nonwetting":
            phase = self.fluid.phases[phase_names.index(flow_type)]
            return self.phase_potential_flux(phase, domains)

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            domains
        )
        boundary_operator = self.combine_boundary_operators_darcy_flux(
            subdomains=domains
        )
        flux = (
            discr.flux() @ self.pressure(domains)
            + discr.bound_flux()
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int()
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source() @ self.vector_source_darcy_flux(domains)
        )

        flux.set_name("Darcy_flux")
        return flux

    def phase_pressure(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Pressure of a phase: ``pressure`` for the gas, ``pressure - capillary
        pressure`` for the aqueous (wetting) phase if capillary pressure is enabled."""
        is_aqueous = phase == self.fluid.phases[0]
        if is_aqueous and self.params.get("enable_capillary_pressure", False):
            return self.pressure(domains) - self.capillary_pressure(domains)
        return self.pressure(domains)

    def phase_potential_flux(
        self, phase: pp.Phase, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Phase-potential flux

        .. math::

            v_j = -K (\nabla p_j - \rho_j g).

        It uses the Darcy flux discretization and its boundary types. On Dirichlet
        faces the phase pressure is evaluated from the boundary values; on Neumann
        faces the Darcy flux boundary values are used.

        """
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        intf_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        discr = self.darcy_flux_discretization(subdomains)
        boundary_operator = self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=partial(self.phase_pressure, phase),
            neumann_operator=self.darcy_flux,
            robin_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name=f"bc_values_phase_potential_flux_{phase.name}",
        )

        # Gravity force with the density of the phase, along the last coordinate.
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        size = sum(sd.num_cells for sd in subdomains)
        gravity = pp.wrap_as_dense_ad_array(g, size=size, name="gravity")
        e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
        gravity_force = Scalar(-1) * e_n @ (phase.density(subdomains) * gravity)

        flux = (
            discr.flux() @ self.phase_pressure(phase, subdomains)
            + discr.bound_flux()
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int()
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source() @ gravity_force
        )
        flux.set_name(f"phase_potential_flux_{phase.name}")
        return flux

    def combine_boundary_operators_darcy_flux(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Combine Darcy flux boundary operators.

        Note that the default Robin operator is the same as that of Neumann. Override
        this method to define and assign another boundary operator of your choice. The
        new operator should then be passed as an argument to the
        _combine_boundary_operators method, just like self.darcy_flux is passed to
        robin_operator in the default model.

        Parameters:
            subdomains: List of the subdomains whose boundary operators are to be
                combined.

        Returns:
            The combined Darcy flux boundary operator.

        """
        op = self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.pressure,
            neumann_operator=self.darcy_flux,
            robin_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name="bc_values_" + self.bc_data_darcy_flux_key,
        )
        return op

    def interface_darcy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Darcy flux on interfaces.

        The units of the Darcy flux are [m^2 Pa / s], see note in :meth:`darcy_flux`.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Darcy flux equation on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        # Gradient operator in the normal direction. The collapsed distance is
        # :math:`\frac{a}{2}` on either side of the fracture.
        # We assume here that :meth:`aperture` is implemented to give a meaningful value
        # also for subdomains of co-dimension > 1.
        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg()
            @ self.aperture(subdomains) ** Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        # Project the two pressures to the interface and multiply with the normal
        # diffusivity.
        pressure_l = projection.secondary_to_mortar_avg() @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg() @ self.pressure_trace(
            subdomains
        )
        eq = self.interface_darcy_flux(interfaces) - self.volume_integral(
            self.normal_permeability(interfaces)
            * (
                normal_gradient * (pressure_h - pressure_l)
                + self.interface_vector_source_darcy_flux(interfaces)
            ),
            interfaces,
            1,
        )
        eq.set_name("interface_darcy_flux_equation")
        return eq

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source_darcy_flux(
        self, grids: Union[list[pp.Grid], list[pp.MortarGrid]]
    ) -> pp.ad.Operator:
        """Vector source term. Represents gravity effects.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.

        Returns:
            Cell-wise nd-vector source term operator.

        """
        return self.gravity_force(grids, "fluid")

    def interface_vector_source_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface vector source term.

        The term is the dot product of unit normals and vector source values.
        Normalization is needed to balance the integration done in the interface flux
        law.

        Parameters:
            interfaces: List of interfaces where the vector source is defined.

        Returns:
            Cell-wise vector source term.

        """
        # Account for sign of boundary face normals.
        # No scaling with interface cell volumes.
        normals = self.outwards_internal_boundary_normals(interfaces, unitary=True)
        # Project vector source from lower-dimensional neighbors to the interfaces.
        # This allows including pressure and temperature dependent density, which would
        # not be defined on the interface.
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )
        # int? @EK
        vector_source = (
            projection.secondary_to_mortar_avg()
            @ self.vector_source_darcy_flux(subdomain_neighbors)
        )
        # Make dot product with vector source in two steps. First multiply the vector
        # source with a matrix (though the formal mypy type is Operator, the matrix is
        # composed by summation).
        normals_times_source = normals * vector_source
        # Then sum over the nd dimensions. The result will in effect be a matrix.
        nd_to_scalar_sum = pp.ad.sum_projection_list(
            [e.T for e in self.basis(interfaces, dim=self.nd)]
        )
        # Finally, the dot product between normal vectors and the vector source. This
        # must be implemented as a matrix-vector product (yes, this is confusing).
        dot_product = nd_to_scalar_sum @ normals_times_source
        dot_product.set_name("Interface vector source term")
        return dot_product


class AdvectiveFluxCapillaryPressure(pp.PorePyModel):
    """Mixin class for discretizing advective fluxes."""

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """Well flux variables on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """

    def advective_flux(
        self,
        subdomains: list[pp.Grid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindAd,
        bc_values: pp.ad.Operator,
        interface_flux: Optional[
            Callable[[list[pp.MortarGrid]], pp.ad.Operator]
        ] = None,
        flow_type: str = "nonwetting",
    ) -> pp.ad.Operator:
        """An operator representing the advective flux on subdomains.

        .. note::
            The implementation assumes that the advective flux is discretized using a
            standard upwind discretization. Other discretizations may be possible, but
            this has not been considered.

        Parameters:
            subdomains: List of subdomains.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.
            bc_values: Boundary conditions for the advective flux.
            interface_flux: Interface flux operator/variable. If subdomains have no
                neighboring interfaces, this argument can be omitted.
            flow_type: Passed on to :meth:`darcy_flux`; ``"nonwetting"`` or a phase
                name.

        Returns:
            Operator representing the advective flux.

        """
        darcy_flux = self.darcy_flux(subdomains, flow_type)
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        flux: pp.ad.Operator = darcy_flux * (discr.upwind() @ advected_entity)

        flux += (
            discr.bound_transport_dir() @ (darcy_flux * bc_values)
            # Advective flux coming from lower-dimensional subdomains
            + discr.bound_transport_neu() @ bc_values
        )

        if interface_flux is not None:
            flux += (
                discr.bound_transport_neu()
                @ mortar_projection.mortar_to_primary_int()
                @ interface_flux(interfaces)
            )
        else:
            assert len(interfaces) == 0
        return flux

    def interface_advective_flux(
        self,
        interfaces: list[pp.MortarGrid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindCouplingAd,
    ) -> pp.ad.Operator:
        """An operator represetning the advective flux on interfaces.

        .. note::
            The implementation here is tailored for discretization using an upwind
            discretization for the advective interface flux. Other discretizaitons may
            be possible, but this has not been considered.

        Parameters:
            interfaces: List of interface grids.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.

        Returns:
            Operator representing the advective flux on the interfaces.

        """
        # If no interfaces are given, make sure to proceed with a non-empty subdomain
        # list if relevant.
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        trace = pp.ad.Trace(subdomains)
        # Project the two advected entities to the interface and multiply with upstream
        # weights and the interface Darcy flux.
        interface_flux: pp.ad.Operator = self.interface_darcy_flux(interfaces) * (
            discr.upwind_primary()
            @ mortar_projection.primary_to_mortar_avg()
            @ trace.trace
            @ advected_entity
            + discr.upwind_secondary()
            @ mortar_projection.secondary_to_mortar_avg()
            @ advected_entity
        )
        return interface_flux

    def well_advective_flux(
        self,
        interfaces: list[pp.MortarGrid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindCouplingAd,
    ) -> pp.ad.Operator:
        """An operator represetning the advective flux on interfaces.

        .. note::
            The implementation here is tailored for discretization using an upwind
            discretization for the advective interface flux. Other discretizaitons may
            be possible, but this has not been considered.

        Parameters:
            interfaces: List of interface grids.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.

        Returns:
            Operator representing the advective flux on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        # Project the two advected entities to the interface and multiply with upstream
        # weights and the interface Darcy flux.
        interface_flux: pp.ad.Operator = self.well_flux(interfaces) * (
            discr.upwind_primary()
            @ mortar_projection.primary_to_mortar_avg()
            @ advected_entity
            + discr.upwind_secondary()
            @ mortar_projection.secondary_to_mortar_avg()
            @ advected_entity
        )

        return interface_flux
