import porepy as pp
import numpy as np
from SPE11A_caseAdata import case_A
from porepy.models.constitutive_laws import ConstantPermeability, ConstantPorosity


class porositymixin(ConstantPorosity):
    """This class overrides porepy's original def porosity()."""

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cellevis porøsitet fra case_A facies, basert på oppskrift fra peter."""
        size = sum(
            sd.num_cells for sd in subdomains
        )  # Vi har bare ett sd, men det funker ikke å skrive subdomains.num_cells, så gjør heller dette.

        # Initialiseringer:
        porosity_p = np.zeros(size)
        offset = 0

        for sd in subdomains:
            for facies, por in case_A["POROSITY"].items():
                porosity_p[sd.tags[facies]] = por
            offset += sd.num_cells

        # Wrapper som pp.ad.operator istedenfor det porepy bruker:
        porosity = pp.wrap_as_dense_ad_array(porosity_p, size, name="porosity")

        return porosity


class permeabilitymixin(ConstantPermeability):
    """This class overrides Porepy's def permeability()."""

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cellevis permeability fra case_A facies, basert på oppskrift fra Peter."""

        size = sum(sd.num_cells for sd in subdomains)

        # Initialisering:
        permeability_p = np.zeros(size)
        offset = 0

        for sd in subdomains:
            for facies, perm in case_A["PERMEABILITY"].items():
                permeability_p[sd.tags[facies]] = perm

            offset += sd.num_cells
            print(
                "Min permeability:", permeability_p.min()
            )  # TODO: Hvorfor printes denne så mange ganger?
            print("Max permeability:", permeability_p.max())

        # Wrap as pp.ad.operator:
        permeability = pp.wrap_as_dense_ad_array(
            permeability_p, size, name="permeability"
        )

        return self.isotropic_second_order_tensor(subdomains, permeability)
