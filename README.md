# Capillary pressure and SPE11A simulations with PorePy

This repository contains the simulation and plotting code for the capillary
pressure studies and SPE11A-related experiments in the thesis. The only
included data directory is `spe11_data/`, containing the SPE11A geometry and
mesh. 

The active-capillary SPE11A simulation is a diagnostic pilot, not a completed
benchmark. The original run was stopped at 77.413336 s of a 300 s target.
Both wells injected from time zero at a combined rate of 1e-5 kg/s, unlike
the benchmark schedule. A new run will not necessarily reproduce the
interrupted state exactly.

## Files

- purewaterstart.py is the runfile for two-phase flow simulations with the capillary pressure function. You can enable or disable this function in this file's params.
- `spe11_data/`: SPE11A geometry and mesh.
- `spe11a_capillary_simulation.py`: active-capillary SPE11A pilot.
- `two_layer_capillary_simulation.py`: two-layer capillary-pressure studies.
- `SPE11A_*.py`: SPE11A geometry, material properties, sources, boundary
  conditions and additional simulation scripts.
- `capillary_pressure.py` and `DarcysCapillaryPressure.py`: capillary-pressure
  relations and flow discretization.
- `Fluid_and_component_flux.py`, `phase_potential_upwinding.py` and
  `well_sources.py`: flow and well-source components required by the models.
- `plot_*.py` and `results_common.py`: plotting scripts and shared helpers.
- `requirements-simulation.txt` and `requirements-plotting.txt`: dependencies.

## Environment

The original runs used Python 3.14.6, PorePy 1.13.0 at commit
`f743bcb0b3112c8c171baa11f40f179a1f8883ac`, NumPy 2.4.6, Matplotlib 3.10.9,
Meshio 5.3.5, Gmsh 4.15.2 and PyPardiso 0.4.7. On Linux, create an environment
and install the simulation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-simulation.txt
```

The simulations also need the external Gmsh/PorePy dependencies.
`requirements-plotting.txt` provides the smaller plotting-only environment.
The scripts were updated after the original runs to reflect renamed files, so
their current file hashes differ from the original source hashes.

## Run the simulations

Run commands from the repository root. Each run creates a timestamped output
folder locally. The two-layer cases use an injection rate of 1e-4 kg/s and a
lower-layer entry pressure of 50 Pa:

```bash
for upper in 100 500 1000 2000 50000 60000 62500 65000; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
    python two_layer_capillary_simulation.py --upper "$upper" --final-time 300 \
    --output-root visualization_capillary_runs/thesis_short --no-postprocess
done

for upper in 200 500 3870.63; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
    python two_layer_capillary_simulation.py --upper "$upper" --final-time 3000 \
    --max-dt 50 --output-root visualization_capillary_runs/thesis_long --no-postprocess
done
```

To start the SPE11A pilot:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  python spe11a_capillary_simulation.py
```

It targets 300 s and writes under `visualization_spe11a_capillary_runs/`.
These exploratory cases are not the SPE11 benchmark schedule.

## Generate figures

The setup plots can be generated without simulation output. The other plots
require suitable saved output from the corresponding simulations; the
original run data is not included.

```bash
export MPLBACKEND=Agg
export POREPY_RESULTS_FIGURES="$PWD/figurer_results"
python plot_results_setup.py

export POREPY_RESULTS_RUNS="$PWD/visualization_capillary_runs/thesis_short"
python plot_results_entry_pressure_study.py --lower 50 --cases 100 500 1000 2000
python plot_results_solver_cost.py --vary upper --fixed 50 \
  --cases 100 500 1000 2000 50000 60000 62500 65000

export POREPY_RESULTS_RUNS="$PWD/visualization_capillary_runs/thesis_long"
python plot_results_long_injection.py --lower 50 --upper 500 \
  --snapshot-times 300 1000 2000 3000
python plot_results_long_injection.py --lower 50 --upper 200 --suffix _200Pa \
  --snapshot-times 300 1000 2000 3000
python plot_results_long_injection.py --lower 50 --upper 3870.63 --suffix _3870Pa \
  --snapshot-times 300 1000 2000 3000
```

For SPE11A figures, pass the path to a suitable newly generated run:

```bash
python plot_spe11a_active_capillary_results.py \
  --run-folder /path/to/spe11a/run --figure-folder "$PWD/figurer_results"
```

Plotting scripts can create PNG images and LaTeX tables locally; these
generated files are not included in the repository.

## Sources and licensing

- **SPE11 benchmark:** [11th SPE Comparative Solution Project](https://spe.org/csp/spe11/)
  and its [benchmark repository](https://github.com/Simulation-Benchmarks/11thSPE-CSP),
  including the geometry files.
- **PorePy:** [PorePy repository](https://github.com/pmgbergen/porepy);
  the simulations used
  [commit `f743bcb0`](https://github.com/pmgbergen/porepy/tree/f743bcb0b3112c8c171baa11f40f179a1f8883ac).
  PorePy is an external dependency, distributed under its own license.
- **Well-cell selection:** the attribution and source link are retained in
  `well_sources.py`.
