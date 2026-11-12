"""Denne fila er den samme som SPE11A_run, bare at det er lagt til tracer som mix arver fra istedenfor
at mix arver fra singlephaseflow.

Modellen produserer det tracerplottet vi forventer, mog arver slik vi vil.

Tidssteget for plotting av tracer er så lite at etter det første plottet kommer
opp er det bare  avbryte i terminal, fordi d tar så lang tid.

"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import porepy as pp
from porepy.viz.exporter import DataInput
import matplotlib.pyplot as plt

# Etter restructure:
from SPE11A_geometry import Modifiedgeometry, InactiveCellsMixin, GEO_FILE_CASE_A
from SPE11A_materials import porositymixin, permeabilitymixin
from SPE11A_source import Modifiedsource
from SPE11A_boundaryconditions import ModifiedBC
from SPE11A_caseAdata import case_A
from porepy.examples.tracer_flow import TracerFlowModel
import numpy as np
from SPE11A_source import well_cell_id


logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler('app.log')  # Specify the log file name
logging.basicConfig(level=logging.DEBUG)


class TracerMIX(
    InactiveCellsMixin,
    Modifiedgeometry,
    ModifiedBC,
    porositymixin,
    permeabilitymixin,
    Modifiedsource,
    TracerFlowModel,
):
    # ny:
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
            self.well_cells.extend(ids.flatten())  # gjør om til 1D.
        super().prepare_simulation()
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
        # TODO: samme som for spe11a_run-py: ^^Denne er gjort litt annerledes enn peter sin add_constant_spe11_data(), men er det ok?

    def ic_values_overall_fraction(self, component, sd):

        if component.name != "tracer":
            return super().ic_values_overall_fraction(component, sd) #hvis komponenten ikke heter tracer(feks vann, ikke gjør noe med den)

        return np.zeros(sd.num_cells)

    def component_source(self, component, subdomains):
        """dette er tracerinjeksjon. husk at det ikke er samme som fluid_source, som er væskeinjeksjon.
        ligning for tracer for ukjent z er:
        ∂(phi*dens*z)/∂t + div(u)dens*z*trykkgradient = 0. fra fluid mass balance.
        """

        size = sum(sd.num_cells for sd in subdomains)
        q = np.zeros(size)
        tracer_rate = self.params["Tracer_injection_rate_kg"]
        start = 0
        for sd in subdomains:
            for cell in self.well_cells:
                q[start + cell] = tracer_rate
            start += sd.num_cells
        return q

    # kilde porepy, men endra på:
    """Showcasing tracer flow with some plots at scheduled times. override porepy sin"""

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        plot_schedule = self.params.get("plot_schedule", [])
        if not plot_schedule:
            return

        t = plot_schedule[0]
        if self.time_manager.time >= t:
            plot_schedule.pop(0)

            minutes = int(self.time_manager.time / 60)
            title = f"Tracer after {minutes} minutes"

            pp.plot_grid(
                self.mdg,
                "z_tracer",  # setter på tracer
                clim=(0, 1),
                figsize=(10, 8),
                linewidth=0.2,
                title=title,
                plot_2d=True,
            )
            out_path = Path(__file__).with_name('tracer_plot.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f'Saved plot to {out_path}')


# Var tidligere i mix: Iterationexporting,

# Dette fjernes pga vi har lagt inn tracerflowmodel:
# Lage en væske (vann) som skal legges til i params:
# water = pp.FluidComponent(**pp.fluid_values.water)  #objekt som kan brukes av self.fluid i porepy sin SPF.
# tracer = TracerFluid() #Tracer
# material_constants = {"fluid": water,
# "tracer_fluid": tracer,}               #her kunne man ha lagt til solid også, men det har jo ikke vi


# Tid:
dt_init = (
    1 / 24 * pp.DAY
)  # Initial time step: 1gang/dag. Satt stor for ikke å bruke for mange tidssteg
T_end = 5 * pp.DAY  # Sluttid er etter 5 dager
# dt_min_max=(  ,  )     #min max time step respectively, spørsmål: trenger vi dette?
max_iterations = 10  # Max iterations spørsmål: hvilken max iter brukes? denne eller 30?
newton_tol = 1e-8  # toleranse for newton solver


# Lage en time manager som legges til i params:
time_manager = pp.TimeManager(
    schedule=[0, T_end],
    dt_init=dt_init,
    constant_dt=False,
    iter_max=max_iterations,
    print_info=True,
)

# Hvis simuleringen krasjer: Sette constant_dt til False.
# Da vil tidssteg tilpasses automatisk slik at den ikkelineære løseren funker (måtte gjøre dette).
# Schedule:  [0,5*pp.Day] (simuleringen tar 5 dager)


# Sette parametre:
params = {
    "progressbars": False,
    "grid_file_name": str(GEO_FILE_CASE_A),
    "max_iterations": 30,
    "folder_name": "Newton_debug",
    "nl_convergence_tol": newton_tol,
    "Tracer_injection_rate_kg": 0.1,
    "Injection_rate_kg": 0.1,
    "plot_schedule":[0.002*pp.DAY], #[0.002 * pp.DAY],  # [1*pp.DAY, 3*pp.DAY],
    "time_manager": time_manager,
}  # husk at vi har endret disse, fjerner: "material_constants": material_constants,


# Kjøre selve modellen:
model = TracerMIX(params)  # mix(params)
print(model._is_nonlinear_problem())
pp.run_time_dependent_model(model, params)
