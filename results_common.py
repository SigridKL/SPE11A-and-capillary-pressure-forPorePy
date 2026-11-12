"""Shared helpers for the result figures of the two-layer capillary-barrier study.

The module mirrors the *physical* parameters of ``purewaterstart.py`` so that the
figures can be produced without importing (and thereby executing) the run script.
If a parameter is changed in ``purewaterstart.py`` it must be changed here as well;
``check_parameters_against_run_script`` warns if the two disagree.

All figures are written to ``FIG_FOLDER`` (``figurer_results`` next to this file).
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parent
RUNS_FOLDER = Path(os.environ.get("POREPY_RESULTS_RUNS", ROOT / "visualization_capillary_runs"))
FIG_FOLDER = Path(os.environ.get("POREPY_RESULTS_FIGURES", ROOT / "figurer_results"))
RUN_SCRIPT = ROOT / "purewaterstart.py"

PRESSURE_PATTERN = re.compile(r"pe_lower_([^_]+)_upper_([^_]+)__(\d{8}_\d{6})")

# Colour-blind safe palette (Okabe-Ito), used consistently across all figures.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]

# ---------------------------------------------------------------------------
# Physical setup, mirrored from purewaterstart.py.
# ---------------------------------------------------------------------------
PARAMS: dict[str, Any] = {
    "domain_width": 2.8,
    "domain_height": 1.2,
    "y_split": 0.5,
    "buehøyde": 0.25,
    "initial_pressure": 1.0,
    "cell_size": 0.05,
    "density_h2o": 998.21,
    "viscosity_h2o": 0.0010016,
    "density_co2": 1.8152,
    "viscosity_co2": 1.4675e-05,
    "porosity_lower": 0.2,
    "porosity_upper": 0.15,
    "permeability_lower": 1e-5,
    "permeability_upper": 5e-7,
    "well_size": 0.2,
    "well_pos_x": 0.9,
    "well_pos_y": 0.3,
    "injection_rate_kg": 1e-4,
    "injection_co2_fraction": 1.0,
    "s_gas_bc": 0.2,
    "s_gas_init_lower": 0.2,
    "s_gas_init_upper": 0.2,
    "sw_imm": 0.2,
    "sn_imm": 0.2,
    "c2": 2.0,
    "pcap_max": 9.5e4,
    "c_w1": 2.0,
    "c_n1": 2.0,
    "dt_init": 3.125,
    "dt_min": 1e-2,
    "t_final": 300.0,
    "nl_max_iterations": 20,
    "nl_convergence_inc_atol": 2e-7,
    "nl_convergence_inc_rtol": 1e-5,
    "nl_convergence_res_atol": 1e-10,
}
GRAVITY = 9.81


def check_parameters_against_run_script() -> None:
    """Warn if the mirrored parameters disagree with ``purewaterstart.py``."""
    if not RUN_SCRIPT.exists():
        return
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^params\s*=\s*\{", source, flags=re.MULTILINE)
    if match is None:
        return
    # Extract the dictionary literal by brace matching.
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    else:
        return
    literal = source[start:end]
    dictionary = ast.parse(literal, mode="eval").body
    if not isinstance(dictionary, ast.Dict):
        raise ValueError("Expected the run-script parameters to be a dictionary.")
    script_params = {}
    for key, value in zip(dictionary.keys, dictionary.values):
        if key is None:
            continue
        try:
            name, number = ast.literal_eval(key), ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        script_params[name] = number
    for key, value in script_params.items():
        if key in PARAMS and isinstance(value, (int, float)):
            if not np.isclose(float(value), float(PARAMS[key])):
                warnings.warn(
                    f"results_common.PARAMS[{key!r}]={PARAMS[key]} differs from "
                    f"purewaterstart.py ({value}). Update results_common.py.",
                    stacklevel=2,
                )


# ---------------------------------------------------------------------------
# Constitutive relations (same formulas as capillary_pressure.py and the
# relative_permeability method in purewaterstart.py).
# ---------------------------------------------------------------------------
def gas_saturation_from_z(z: np.ndarray) -> np.ndarray:
    """Invert z_CO2 -> s_gas for immiscible pure phases."""
    rho_w, rho_g = PARAMS["density_h2o"], PARAMS["density_co2"]
    z = np.asarray(z, dtype=float)
    return np.clip(rho_w * z / (rho_w * z + rho_g * (1.0 - z)), 0.0, 1.0)


def z_from_gas_saturation(s_gas: float | np.ndarray) -> float | np.ndarray:
    rho_w, rho_g = PARAMS["density_h2o"], PARAMS["density_co2"]
    return s_gas * rho_g / (s_gas * rho_g + (1.0 - s_gas) * rho_w)


def capillary_pressure(s_w: np.ndarray, p_entry: float) -> np.ndarray:
    """Brooks-Corey capillary pressure with the cap used in the model."""
    sw_imm = PARAMS["sw_imm"]
    s_eff = np.clip((np.asarray(s_w) - sw_imm) / (1.0 - sw_imm), 1e-12, None)
    pc = p_entry * s_eff ** (-1.0 / PARAMS["c2"])
    return np.minimum(pc, PARAMS["pcap_max"])


def relative_permeability(s: np.ndarray, phase: str) -> np.ndarray:
    sw_imm, sn_imm = PARAMS["sw_imm"], PARAMS["sn_imm"]
    s_imm = sw_imm if phase == "aqueous" else sn_imm
    exponent = PARAMS["c_w1"] if phase == "aqueous" else PARAMS["c_n1"]
    s_eff = np.clip((np.asarray(s) - s_imm) / (1.0 - sw_imm - sn_imm), 0.0, 1.0)
    return s_eff**exponent


def entry_pressure_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g} kPa"
    return f"{value:g} Pa"


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------
def interface_height(x: np.ndarray) -> np.ndarray:
    """y-coordinate of the sinusoidal layer interface (same as TwoSandLayers)."""
    return PARAMS["y_split"] + PARAMS["buehøyde"] * np.sin(
        np.pi * np.asarray(x) / PARAMS["domain_width"]
    )


def upper_layer_mask(cell_centers: np.ndarray) -> np.ndarray:
    return cell_centers[:, 1] > interface_height(cell_centers[:, 0])


def well_cell_mask(cell_centers: np.ndarray) -> np.ndarray:
    half = PARAMS["well_size"] / 2
    x0, y0 = PARAMS["well_pos_x"], PARAMS["well_pos_y"]
    return (
        (cell_centers[:, 0] >= x0 - half)
        & (cell_centers[:, 0] <= x0 + half)
        & (cell_centers[:, 1] >= y0 - half)
        & (cell_centers[:, 1] <= y0 + half)
    )


def cells_below_interface_mask(cell_centers: np.ndarray, n_rows: int = 1) -> np.ndarray:
    """Lower-layer cells within ``n_rows`` cell heights below the interface."""
    y_if = interface_height(cell_centers[:, 0])
    dy = PARAMS["cell_size"]
    return (cell_centers[:, 1] <= y_if) & (cell_centers[:, 1] > y_if - n_rows * dy)


# ---------------------------------------------------------------------------
# Run discovery and I/O.
# ---------------------------------------------------------------------------
@dataclass
class RunInfo:
    folder: Path
    pe_lower: float
    pe_upper: float
    timestamp: str
    final_time: float = np.nan
    completed: bool = False
    n_outputs: int = 0

    @property
    def label(self) -> str:
        return (
            f"$p_{{e,\\mathrm{{low}}}}$={entry_pressure_label(self.pe_lower)}, "
            f"$p_{{e,\\mathrm{{up}}}}$={entry_pressure_label(self.pe_upper)}"
        )


def read_pvd(folder: Path) -> list[tuple[float, Path]]:
    pvd = folder / "data.pvd"
    if not pvd.exists():
        return []
    tree = ET.parse(pvd)
    entries = []
    for dataset in tree.getroot().iter("DataSet"):
        entries.append((float(dataset.get("timestep")), folder / dataset.get("file")))
    entries.sort(key=lambda item: item[0])
    return entries


def find_runs(
    runs_folder: Path = RUNS_FOLDER,
    pe_lower: float | None = None,
    pe_upper: float | None = None,
    latest_only: bool = True,
) -> list[RunInfo]:
    """Discover run folders, optionally filtered on entry pressures.

    With ``latest_only`` only one run per (pe_lower, pe_upper) pair is returned:
    the one that reached the latest simulated time, ties broken by timestamp.
    """
    runs: list[RunInfo] = []
    for folder in sorted(runs_folder.iterdir()):
        if not folder.is_dir() or folder.name.endswith("_iterations"):
            continue
        match = PRESSURE_PATTERN.match(folder.name)
        if match is None:
            continue
        info = RunInfo(folder, float(match[1]), float(match[2]), match[3])
        if pe_lower is not None and not np.isclose(info.pe_lower, pe_lower):
            continue
        if pe_upper is not None and not np.isclose(info.pe_upper, pe_upper):
            continue
        outputs = read_pvd(folder)
        info.n_outputs = len(outputs)
        if outputs:
            info.final_time = outputs[-1][0]
        stats = read_solver_statistics(folder)
        if stats is not None:
            info.completed = bool(stats["global"].get("final_time_reached", 0))
        runs.append(info)

    if latest_only:
        best: dict[tuple[float, float], RunInfo] = {}
        for info in runs:
            key = (info.pe_lower, info.pe_upper)
            current = best.get(key)
            if current is None or (
                (np.nan_to_num(info.final_time), info.timestamp)
                > (np.nan_to_num(current.final_time), current.timestamp)
            ):
                best[key] = info
        runs = sorted(best.values(), key=lambda r: (r.pe_lower, r.pe_upper))
    return runs


def read_solver_statistics(folder: Path) -> dict[str, Any] | None:
    path = folder / "solver_statistics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


@dataclass
class Attempt:
    index: int
    time: float
    dt: float
    iterations: int
    success: bool
    inc_abs: list[float] = field(default_factory=list)
    res_abs: list[float] = field(default_factory=list)


def solver_attempts(stats: dict[str, Any]) -> list[Attempt]:
    """Chronological list of time-step attempts from a statistics dictionary."""
    attempts = []
    for key, record in stats.items():
        if key == "global" or not isinstance(record, dict):
            continue
        try:
            index = int(key)
        except ValueError:
            continue
        info = record.get("convergence_info", {})
        attempts.append(
            Attempt(
                index=index,
                time=float(record.get("time", np.nan)),
                dt=float(record.get("dt", np.nan)),
                iterations=int(record.get("num_iterations", 0)),
                success=record.get("solver_status") == "successful",
                inc_abs=[float(v) for v in info.get("inc_abs", []) or []],
                res_abs=[float(v) for v in info.get("res_abs", []) or []],
            )
        )
    attempts.sort(key=lambda a: a.index)
    return attempts


# ---------------------------------------------------------------------------
# Mesh and field handling.
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    time: float
    points: np.ndarray  # (n_points, 2)
    quads: np.ndarray  # (n_cells, 4)
    centers: np.ndarray  # (n_cells, 2)
    areas: np.ndarray  # (n_cells,)
    fields: dict[str, np.ndarray]

    @property
    def polygons(self) -> list[np.ndarray]:
        return [self.points[q] for q in self.quads]


def read_snapshot(path: Path, time: float = np.nan) -> Snapshot:
    mesh = meshio.read(str(path))
    points = mesh.points[:, :2]
    quads = mesh.cells_dict["quad"]
    corners = points[quads]
    x, y = corners[:, :, 0], corners[:, :, 1]
    areas = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    )
    fields = {name: np.asarray(data["quad"]) for name, data in mesh.cell_data_dict.items()}
    return Snapshot(time, points, quads, corners.mean(axis=1), areas, fields)


def layer_porosity(centers: np.ndarray) -> np.ndarray:
    """Porosity from the geometry (the exported field uses a flat split)."""
    return np.where(
        upper_layer_mask(centers), PARAMS["porosity_upper"], PARAMS["porosity_lower"]
    )


def co2_mass_per_cell(snap: Snapshot) -> np.ndarray:
    """CO2 mass [kg] per cell for unit depth, pure phases assumed."""
    s_gas = gas_saturation_from_z(snap.fields["z_CO2"])
    return layer_porosity(snap.centers) * snap.areas * s_gas * PARAMS["density_co2"]


def nearest_cell(centers: np.ndarray, x: float, y: float) -> int:
    return int(np.argmin((centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2))


MONITOR_POINTS: dict[str, tuple[float, float]] = {
    # Below the crest of the interface arch (the capillary trap).
    "trap": (PARAMS["domain_width"] / 2, PARAMS["y_split"] + PARAMS["buehøyde"] - 0.075),
    # Just above the crest, in the upper layer.
    "above trap": (PARAMS["domain_width"] / 2, PARAMS["y_split"] + PARAMS["buehøyde"] + 0.075),
    "well": (PARAMS["well_pos_x"], PARAMS["well_pos_y"]),
}


def time_series(run: RunInfo) -> dict[str, np.ndarray]:
    """Scalar diagnostics as functions of time for one run."""
    outputs = read_pvd(run.folder)
    rows: dict[str, list[float]] = {
        "time": [],
        "injected": [],
        "mass_total": [],
        "mass_lower": [],
        "mass_upper": [],
        "sg_max_lower": [],
        "sg_max_upper": [],
        "pc_max_below_interface": [],
        "p_trap": [],
        "p_above_trap": [],
        "p_well": [],
        "pw_trap": [],
    }
    monitor_ids: dict[str, int] | None = None
    for time, path in outputs:
        snap = read_snapshot(path, time)
        if monitor_ids is None:
            monitor_ids = {
                name: nearest_cell(snap.centers, *xy) for name, xy in MONITOR_POINTS.items()
            }
        upper = upper_layer_mask(snap.centers)
        below = cells_below_interface_mask(snap.centers)
        mass = co2_mass_per_cell(snap)
        s_gas = gas_saturation_from_z(snap.fields["z_CO2"])
        p = snap.fields["pressure"]
        pc = snap.fields.get("capillary_pressure", np.zeros_like(p))
        rows["time"].append(time)
        rows["injected"].append(
            PARAMS["injection_rate_kg"] * PARAMS["injection_co2_fraction"] * time
        )
        rows["mass_total"].append(mass.sum())
        rows["mass_lower"].append(mass[~upper].sum())
        rows["mass_upper"].append(mass[upper].sum())
        rows["sg_max_lower"].append(s_gas[~upper].max())
        rows["sg_max_upper"].append(s_gas[upper].max())
        rows["pc_max_below_interface"].append(pc[below].max())
        rows["p_trap"].append(p[monitor_ids["trap"]])
        rows["p_above_trap"].append(p[monitor_ids["above trap"]])
        rows["p_well"].append(p[monitor_ids["well"]])
        rows["pw_trap"].append(p[monitor_ids["trap"]] - pc[monitor_ids["trap"]])
    return {key: np.asarray(values) for key, values in rows.items()}


# ---------------------------------------------------------------------------
# Plotting helpers.
# ---------------------------------------------------------------------------
def draw_field(
    ax: plt.Axes,
    snap: Snapshot,
    values: np.ndarray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    show_well: bool = True,
    edgecolor: str = "none",
) -> PolyCollection:
    coll = PolyCollection(
        snap.polygons, array=values, cmap=cmap, edgecolors=edgecolor, linewidths=0.1
    )
    coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    draw_interface(ax)
    if show_well:
        draw_well(ax)
    ax.set_xlim(0, PARAMS["domain_width"])
    ax.set_ylim(0, PARAMS["domain_height"])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    return coll


def draw_interface(ax: plt.Axes, **kwargs: Any) -> None:
    x = np.linspace(0, PARAMS["domain_width"], 400)
    style = {"color": "white", "lw": 1.2, "ls": "--"}
    style.update(kwargs)
    ax.plot(x, interface_height(x), **style)


def draw_well(ax: plt.Axes, **kwargs: Any) -> None:
    half = PARAMS["well_size"] / 2
    x0, y0 = PARAMS["well_pos_x"], PARAMS["well_pos_y"]
    style = {"fill": False, "edgecolor": "red", "lw": 1.2}
    style.update(kwargs)
    ax.add_patch(plt.Rectangle((x0 - half, y0 - half), 2 * half, 2 * half, **style))


def save_figure(fig: plt.Figure, stem: str, also_pdf: bool = True) -> Path:
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)
    png = FIG_FOLDER / f"{stem}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    if also_pdf:
        fig.savefig(FIG_FOLDER / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    return png


def write_latex_table(
    stem: str, header: Iterable[str], rows: Iterable[Iterable[str]], caption: str, label: str
) -> Path:
    """Write a booktabs table as a standalone .tex file for \\input."""
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)
    header = list(header)
    rows = [list(r) for r in rows]
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}",
        "    \\toprule",
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    lines += ["    " + " & ".join(r) + " \\\\" for r in rows]
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    path = FIG_FOLDER / f"{stem}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")
    return path


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
