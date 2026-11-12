import copy
import pathlib

import numpy as np
from porepy.fracs.fracture_importer import dfm_from_gmsh
from porepy.grids.partition import extract_subgrid
from SPE11A_caseAdata import case_A

import porepy as pp

""" Her settes cellene i facies 7 til å være inaktive. dette skjer i inactivecellsmixin.
"""


DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve() / "spe11_data"
ZIP_FILENAME: str = "por_perm_case2a.zip"
URL_CASE_A: str = "https://raw.githubusercontent.com/Simulation-Benchmarks/11thSPE-CSP/refs/heads/main/geometries/spe11a.geo"
GEO_FILE_CASE_A: str = "spe11a.geo"


class Modifiedgeometry:
    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmax": case_A["WIDTH"],  # 2.8m
                "ymax": case_A["HEIGHT"],  # 1.2m
            }
        )


class InactiveCellsMixin:
    """Denne klassen gjør cellene uten porøsitet (cellene i facies 7) inaktive.
    Kilde: Peter, men jeg har justert den litt, slik at den passer denne koden."""

    def create_mdg(self) -> None:

        # Set full domain including cells to be set inactive later.
        # Load from mesh file, generate from fracture network, etc.

        geo_path = DATA_DIR / GEO_FILE_CASE_A  # spe11a.geo
        self.extended_domain = dfm_from_gmsh(
            geo_path, dim=2
        )  # bruke riktig path for denne koden, ikke denne #self.extended_domain = dfm_from_gmsh("extended_domain.msh")
        extended_domain_sd = self.extended_domain.subdomains()[0]

        # Extract subgrid corresponding to the active cells. Here we define active cells
        # as those with non-zero porosity.
        # Før hadde vi de cellene der poro er 0, men vi har ikke poro enda, så for å unngå å hente poro to gangeer, sorterer vi heller inactive cells på
        # cellene i facies 7, siden vi uansett veit at poro er 0 her.

        facies_7 = extended_domain_sd.tags["Facies 7"]
        self.active_cells = np.nonzero(~facies_7)[0]  # (poro>0)[0]
        active_sd, _, _ = extract_subgrid(extended_domain_sd, self.active_cells)

        # Copy any user-defined tags from extended domain to active subdomain.
        # peter hadde liten forbokstav på facies, og her var det mye styr med sd.tags.
        for key, value in extended_domain_sd.tags.items():
            if key.startswith("Facies"):
                active_sd.tags[key] = value[self.active_cells]

        # Set subgrid as domain.
        self.mdg = copy.deepcopy(self.extended_domain)
        self.mdg.remove_subdomain(self.mdg.subdomains()[0])
        self.mdg.add_subdomains(active_sd)
        self.mdg.set_boundary_grid_projections()
        self.mdg.compute_geometry()
        self.nd: int = self.mdg.dim_max()

    # TODO: Spør peter om denne: Skal den følgende funksjonen med? Isåfall, fiks.
    # def data_to_export(
    #     self, time_step_index: int | None = None
    # ) -> list[DataInput]:
    #     #peter hadde:data = super().data_to_export(time_step_index=time_step_index)
    #     #men vi må kalle super uten argumentet (se porepy sin data_to_export her:
    #     # https://github.com/pmgbergen/porepy/blob/develop/src/porepy/viz/data_saving_model_mixin.py
    #     data = super().data_to_export() #kalle på super uten argument

    #     # Add zero values on inactive cells for proper visualization.
    #     extended_domain_nd = self.extended_domain.subdomains()[0]
    #     #Problem med loopen under er:
    #     #feilmelding: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    #     # for grid, name, array in data:
    #     #     full_array = np.zeros(extended_domain_nd.num_cells)
    #     #     full_array[self.active_cells] = array
    #     #     array = full_array
    #     #     # Update data in place.
    #     #     data[data.index((grid, name, array))] = (
    #     #         extended_domain_nd,
    #     #         name,
    #     #         array,
    #     #     )
    #     #løsning: iterere ved å bruke enumerate isteden og oppdatere i lista data direkte:
    #     #ny:
    #     for idx, (grid, name, array) in enumerate(data): #må legge til index som vi skal loope over
    #         full_array = np.zeros(extended_domain_nd.num_cells) #samme som peter
    #         full_array[self.active_cells] = array #samme som peter
    #         data[idx] = (extended_domain_nd,name,full_array) #egt samme metode bare at vi itererer over indexen også, har det noen effekt? jaa nå funker det:)
    #     print('data_to_export er brukt')
    #     return data
