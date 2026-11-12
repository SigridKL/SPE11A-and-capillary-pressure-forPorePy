"""*********Simulering av Single-phase flow for  SPE11 caseA.**********


Jeg har gjort følgende:

- Dokumentet SPF_SPE11A.py er restrukturert s.a geometri, data, materialer, run osv. har sine egne filer:

- Klassen mix er gjort arvbar slik at det ikke er noen dobbelarv.
Klassen SPE11Singlephaseflow(singlephaseflow) som mix tidligere arva fra er fjerna, slik at kun klassen mix
(den vi bruker i modellen nå) arver fra Singlephaseflow. Dermed er det mulig å arve fra mix når man skal sette på tracer også ved å arve fra
tracerflowmodel istedenfor singlephaseflow i fila SPE11A_tracer_run.py.


######## FILBESKRIVELSER ######################
- SPE11A_data.py:
Materialparametre og brønner.

- SPE11A_materials.py:
Permeabilitet og porøsitet for hver facie lastes inn og er visualisert i paraview, sammen med trykket langs en linje.

- SPE11A_source.py:
Source term legges til i class ModifiedSource.
Funksjonen well_cell_id defineres også her (laget av Peter).

- SPE11A_geometry.py:
Geometrien settes (create_mdg og set_domain er her)
InactiveCellsMixin lagt til for å deaktivere facies 7.
InactiveCellsMixin er redigert slik at inaktive celler er definert ved at de er i facies 7,
istedenfor at de der definert ved at de har poro=0. Dermed trengs ikke porøsiteten før mdg er laget. Den gamle poro_np er derfor fjerna.
Dette er testa i Paraview, og alt eksporteres slik vi vil, inkl. at facies 7 uteblir fra visualiseringa.

- SPE11A_boundaryconditions.py:
Randbetingelser fiksa. Dirichlet på toppen, og trykket satt til 0 Pa.

- SPE11A_run.py:
Væske (water) og TimeManager er lagt til i params.

- SPE11A_tracer_run.py:
Tracer er lagt til og plotter tracer etter 1t. Lurt å avbryte plotting etter første plott,
det andre tar superlang tid å få opp. evt juster tidspunkt for plotting.
#######################################################


- self.well_cells er lagra i prepare_simulation() da de trengs i både component_source(tracer) og
fluid_source, i.e. frigjort brønnindeksene fra fluidsource slik at klassene kan brukes i andre modeller også, ikke bare i SPF.
Dette er testa i Paraview og fungerer som det skal med forventa fordeling av trykk, porosity blir med og perm blir med.


TODO:               Hva er disse to forno?:
                    - data_to_export. sjekk om skal med, isåfall fikse.
                    - self.exporter i add_constant_spe11_data(), denne må med.
                    
"""

import logging
import porepy as pp
from porepy.viz.exporter import DataInput
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.viz.data_saving_model_mixin import IterationExporting

# Etter restructure:
from SPE11A_geometry import Modifiedgeometry, InactiveCellsMixin, GEO_FILE_CASE_A
from SPE11A_materials import porositymixin, permeabilitymixin
from SPE11A_source import Modifiedsource
from SPE11A_boundaryconditions import ModifiedBC

# Etter fikling med brønnindekser, måtte ha med disse og:
from SPE11A_caseAdata import case_A
import numpy as np
from SPE11A_source import well_cell_id


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class mix(
    IterationExporting,
    Modifiedgeometry,
    ModifiedBC,
    porositymixin,
    permeabilitymixin,
    InactiveCellsMixin,
    Modifiedsource,
    SinglePhaseFlow,
):
    def prepare_simulation(self) -> None:
        """Saves well cell indices in self.well_cells as these are needed in both
        fluid_source in class Modifiesource and in component_source in this class."""
        self.set_geometry()
        sd = self.mdg.subdomains()[
            0
        ]  # nb må lage cellene før super.prepare simulation.
        wellsize = case_A["WELL_SIZE"]

        self.well_cells = []

        for pos in [case_A["WELL_1_POS"], case_A["WELL_2_POS"]]:
            # "finner koordinatene til nedre venstre hjørne frst, deretter øvre høyre hjørne. brønnene er små kvadrater med sidelende 0.02m"
            corners = np.array(
                [
                    [pos[0] - wellsize / 2, pos[1] - wellsize / 2],
                    [pos[0] + wellsize / 2, pos[1] + wellsize / 2],
                ]
            )

            ids = well_cell_id(sd, corners)
            self.well_cells.extend(ids.flatten())
        super().prepare_simulation()  # type: ignore
        self.add_constant_spe11_data()

    def add_constant_spe11_data(self) -> None:
        """Save the SPE11 data to the exporter."""

        data: list[DataInput] = []

        # Porøsitet:
        for sd in self.mdg.subdomains():
            por_ad = self.porosity([sd])
            por_np = por_ad.value(self.equation_system)
            data.append((sd, "porosity", por_np))

        # Permeabilitet
        for sd in self.mdg.subdomains():
            perm_ad = self.permeability([sd])
            perm_np = perm_ad.value(self.equation_system)
            data.append((sd, "permeability", perm_np))  # TODO: er dette ok?
        self.exporter.add_constant_data(data)

        # TODO: Denne er gjort litt annerledes enn peter sin add_constant_spe11_data(), men er det ok?


# Lage en væske (vann) som skal legges til i params:
water = pp.FluidComponent(
    **pp.fluid_values.water
)  # objekt som kan brukes av self.fluid i porepy sin SPF.
material_constants = {
    "fluid": water
}  # her kunne man ha lagt til solid også, men det har jo ikke vi


# Lage en time manager som legges til i params:
time_manager = pp.TimeManager(
    schedule=[0, 5 * pp.DAY],
    dt_init=1 / 24 * pp.DAY,
    constant_dt=False,
    iter_max=10,
    print_info=True,
)
# Schedule:  [0,5*pp.Day] (simuleringen tar 5 dager)
# For å bruke ikke for mange tidssteg burde dt_init økes. Kanskje til 1/24*pp.Day?
# Hvis simuleringen krasjer: Sette constant_dt til False
# -> tidssteg tilpasses automatisk slik at den ikkelineære løseren funker (måtte gjøre dette).


# Sette parametre:
params = {
    "progressbars": False,
    "grid_file_name": str(GEO_FILE_CASE_A),
    "max_iterations": 30,
    "folder_name": "Newton_debug",
    "nl_convergence_tol": 1e-1,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "Injection_rate_kg": 1.0,
}  # husk at vi har endret disse


# Kjøre selve modellen:
model = mix(params)
# print(model._is_nonlinear_problem())
pp.run_time_dependent_model(model, params)
