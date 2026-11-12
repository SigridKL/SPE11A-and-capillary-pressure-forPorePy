# Results: files for reproduction

This folder is a staging copy for the public repository that will accompany the
thesis. Nothing has been pushed to GitHub.

The package contains the Results LaTeX section, its plotted figures and tables,
the model/post-processing source files, the SPE11A mesh inputs, and the saved
accepted-time-step outputs and solver statistics for the two-layer studies and
the SPE11A active-capillary pilot. The pilot itself reached 77.413336 s of a
300 s target; it is not a completed benchmark run.

## Important outstanding files

The supplied `thesis_results_export0311/results.tex` references these
single-phase pressure figures, but neither file was present in the accessible
workspace or in the supplied export archive:

- `figurer/pressure_med_line.png`
- `figurer/pressuregraf.png`

Consequently, a thesis build that includes the whole Results section will
still need these exact figures, or the corresponding figure blocks in
`results.tex` must be revised. They have deliberately not been replaced by
unrelated pressure images. The tracer image referenced by the section is
included as `figurer/tracer_SPF_well1og2_60min.png`; its available source file
was named `tracer_plot.png`.

The original per-Newton-iteration VTU exports are not included. They are not
read by any figure-generation script or cited result: the Newton-attempt and
failure-history figures use `solver_statistics.json`, while physical plots
use the accepted-time-step VTU files. Omitting these debug snapshots reduces
the package by about 1 GB. All accepted-time-step run data needed by the
included plots is retained.

## Repository layout

- `thesis_results_export0311/results.tex` — Results section. Its
  `\input`/`\includegraphics` paths assume this folder is copied to the
  directory containing the thesis `main.tex`.
- `thesis_results_export0311/figurer_results/` — all figures, tables,
  mass-balance summaries and breakthrough notes used by the Results section.
- `visualization_capillary_runs/` — accepted-time-step output and solver
  statistics for the entry-pressure and long-injection cases.
- `visualization_spe11a_active_capillary_20261106/` — saved SPE11A pilot data.
- `spe11_data/` — SPE11A `.geo` input and the corresponding `.msh` mesh.
- Python files at the root — model mixins, run scripts and figure-generation
  scripts.

The two-layer simulation source and output are retained separately from the
SPE11A pilot. All data was copied from the original run directories without
re-running a simulation. Run metadata contains source hashes and parameters.

## Environment

The saved runs were produced with Python 3.14.6, PorePy 1.13.0 at commit
`f743bcb0b3112c8c171baa11f40f179a1f8883ac`, NumPy 2.4.6, Matplotlib 3.10.9,
Meshio 5.3.5, Gmsh 4.15.2 and PyPardiso 0.4.7. The PorePy source checkout used
for the run was clean. Install the plotting environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plotting.txt
```

To re-run the simulations, install PorePy and its pinned dependencies as well:

```bash
python -m pip install -r requirements-simulation.txt
```

The simulation runs use PyPardiso and the SPE11A geometry reader. Run them on
Linux with the external Gmsh/PorePy dependencies installed. PorePy is an
external dependency and is not copied into this repository.

## Recreate the figures

Run commands from this folder. They regenerate plots from the included runs
and write the figures/tables into the thesis export directory:

```bash
export MPLBACKEND=Agg
export POREPY_RESULTS_FIGURES="$PWD/thesis_results_export0311/figurer_results"

export POREPY_RESULTS_RUNS="$PWD/visualization_capillary_runs/thesis_short_20261103"
python plot_results_setup.py
python plot_results_entry_pressure_study.py --lower 50 --cases 100 500 1000 2000
python plot_results_solver_cost.py --vary upper --fixed 50 \
  --cases 100 500 1000 2000 50000 60000 62500 65000

export POREPY_RESULTS_RUNS="$PWD/visualization_capillary_runs/thesis_long_20261103"
python plot_results_long_injection.py --lower 50 --upper 500 \
  --snapshot-times 300 1000 2000 3000
python plot_results_long_injection.py --lower 50 --upper 200 --suffix _200Pa \
  --snapshot-times 300 1000 2000 3000

export POREPY_RESULTS_RUNS="$PWD/visualization_capillary_runs/thesis_long_20261104"
python plot_results_long_injection.py --lower 50 --upper 3870.63 \
  --suffix _3870Pa --snapshot-times 300 1000 2000 3000

python plot_spe11a_active_capillary_results.py \
  --run-folder "$PWD/visualization_spe11a_active_capillary_20261106/spe11a_pe_min_1_max_1500__20261106_095814" \
  --figure-folder "$PWD/thesis_results_export0311/figurer_results"
```

These commands reproduce the figures and tables from the saved simulation
outputs; they do not re-run the flow simulations. The simulation launch
instructions and case list are in `visualization_capillary_runs/README.md`.
The SPE11A run settings and its early stop are described in
`visualization_spe11a_active_capillary_20261106/README.md`.

## Full simulation runs

To create new simulation outputs rather than use the included runs, see
`visualization_capillary_runs/README.md` and
`visualization_spe11a_active_capillary_20261106/README.md`. The saved runs
include high injection-rate exploratory configurations, not the SPE11A
benchmark schedule. Do not describe the SPE11A pilot as a benchmark result.

## Public release checklist

- Add the missing pressure figures or revise/remove their figure references.
- Decide and add an appropriate license for this project code before inviting
  reuse. The included `spe11_data/spe11a.geo` has its upstream MIT SPDX header;
  retain that attribution and verify the mesh redistribution terms.
- Keep PorePy as an external, version-pinned dependency; do not copy its source
  into this package.
- The raw saved runs are about 191 MB (before Git object packing). The omitted
  per-iteration exports are substantially larger.
