import porepy as pp
import numpy as np


# Finne well indekser
# Kilde: Peter Schultzendorff Github:
# https://github.com/pschultzendorff/tpf_PorePy/blob/dev/src/tpf/derived_models/utils.py#L139


def well_cell_id(
    g: pp.Grid,
    well_boundaries: np.ndarray,
) -> np.ndarray:
    """Identify the cells inside a well boundary.

    Assumes that a grid cell is either fully inside or outside the well. Thus it
    suffices, to check only cell centers to find whether a cell is inside a well.

    Parameters:
        g: Grid.
        well_boundaries ``shape=(2, nd)``: Lower left and upper right corner of the
            square well.

    Returns:
        corners: Indices of the cells inside the well.

    """
    # Ignore z-values of the grid.
    cell_centers: np.ndarray = g.cell_centers[:2, :]
    # Find indices of cells inside the well.
    indices = np.argwhere(
        np.logical_and(
            np.logical_and(
                cell_centers[0] >= well_boundaries[0, 0],
                cell_centers[0] <= well_boundaries[1, 0],
            ),
            np.logical_and(
                cell_centers[1] >= well_boundaries[0, 1],
                cell_centers[1] <= well_boundaries[1, 1],
            ),
        )
    )
    return indices


class Modifiedsource:
    def fluid_source(self, subdomains):
        """Set source term = rate (injeksjonsrate) for one cell and 0 for all other cells.
        For å gjøre det, trenger vi hjørnekoordinatene til brønnene,
        som er gitt ved: # wellcorners = well center +- wellsize/2 i begge retninger.
        """
        other_sources = super().fluid_source(
            subdomains
        )  # fracture source terms, ikke nødvendig her, "best practice"
        size = sum(sd.num_cells for sd in subdomains)
        q = np.zeros(size)
        injection_rate = self.params["Injection_rate_kg"]
        start = 0
        for sd in subdomains:
            for cell in self.well_cells:
                q[start + cell] = injection_rate
            start += sd.num_cells

        # Sette alt sammen:
        # new_source_np = np.hstack(alle_verdier)    # hstack combines multiple arrays into a single, wider array along the second axis (axis 1).
        new_source = pp.ad.DenseArray(
            q
        )  # PorePy trenger ikke numpy arrays, men pp.ad.Operator. Her bruker vi DenseArray, som er en subclass av operator
        total = other_sources + new_source
        return total

    def component_source(self, component: pp.Component, subdomains):
        """Component-wise well source consistent with fluid_source.

        The total injected mass is split between H2O and CO2 using the optional
        parameter ``injection_co2_fraction``. If the parameter is missing, the
        default is pure CO2 injection.
        """
        base_component_source = getattr(super(), "component_source", None)
        if callable(base_component_source):
            other_sources = base_component_source(component, subdomains)
        else:
            other_sources = pp.ad.DenseArray(np.zeros(sum(sd.num_cells for sd in subdomains)))
        size = sum(sd.num_cells for sd in subdomains)
        q = np.zeros(size)
        params = getattr(self, "params", {})
        injection_rate = float(params["Injection_rate_kg"])
        co2_fraction = float(params.get("injection_co2_fraction", 1.0))

        if not hasattr(self, "well_cells"):
            raise AttributeError(
                "Missing attribute 'well_cells'. Ensure prepare_simulation() sets "
                "self.well_cells before source evaluation."
            )
        well_cells = self.well_cells
        if len(well_cells) == 0:
            raise ValueError(
                "well_cells is empty. No injection cells were identified for the mesh."
            )

        if component.name == "CO2":
            rate = injection_rate * co2_fraction
        elif component.name == "H2O":
            rate = injection_rate * (1.0 - co2_fraction)
        else:
            rate = 0.0

        start = 0
        for sd in subdomains:
            for cell in well_cells:
                q[start + cell] = rate
            start += sd.num_cells

        new_source = pp.ad.DenseArray(q)
        return other_sources + new_source
