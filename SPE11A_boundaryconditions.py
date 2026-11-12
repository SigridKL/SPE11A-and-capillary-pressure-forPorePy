import porepy as pp
import numpy as np
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow



class ModifiedBC(BoundaryConditionsSinglePhaseFlow): 
    """Denne klassen følger denne tutorialen: https://github.com/pmgbergen/porepy/blob/develop/tutorials/single_phase_flow.ipynb.
    Kopierer metodene bc_type_darcy_flux og bc_values_pressure fra porepy,
    og justerer slik at kun den øvre randen er Dirichlet og at trykkverdien på denne randen er lik null.
    """
    #Kilde: Porepy.
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the north boundary. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.north, "dir")
        return bc

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Select whether advective flow is allowed through the top boundary.

        The lateral boundaries are always no-flow. The top boundary is also no-flow
        unless ``enable_top_boundary_flow`` is enabled in the model parameters.
        """
        domain_sides = self.domain_boundary_sides(sd)
        if self.params.get("enable_top_boundary_flow", False):
            return pp.BoundaryCondition(sd, domain_sides.north, "dir")
        return pp.BoundaryCondition(sd, domain_sides.all_bf, "neu")
    #Kilde: Porepy.
    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on all sides(neumann by default). Så da har det egt ingen effekt å eksplisitt sette trykket til 0, men gjør det fordet."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        # See section on scaling for explanation of the conversion.
        values[domain_sides.north] = self.units.convert_units(0, "Pa")
        return values  
