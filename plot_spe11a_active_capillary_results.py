from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

from SPE11A_caseAdata import case_A


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = (
    ROOT
    / "visualization_spe11a_active_capillary_20261106"
    / "spe11a_pe_min_1_max_1500__20261106_095814"
)
DEFAULT_FIGURES = ROOT / "figurer_results"
FIGURE_STEMS = (
    "fig_spe11a_active_capillary_state",
    "fig_spe11a_newton_attempts",
    "fig_spe11a_newton_failure_histories",
)


def _read_outputs(run_folder: Path) -> tuple[list[tuple[float, Path]], dict]:
    pvd_path = run_folder / "data.pvd"
    statistics_path = run_folder / "solver_statistics.json"
    if not pvd_path.exists():
        raise FileNotFoundError(f"Missing PVD output: {pvd_path}")
    if not statistics_path.exists():
        raise FileNotFoundError(f"Missing solver statistics: {statistics_path}")

    entries = [
        (float(item.get("timestep")), run_folder / item.get("file"))
        for item in ET.parse(pvd_path).getroot().iter("DataSet")
    ]
    if not entries:
        raise ValueError(f"No saved time steps in {pvd_path}")

    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    return entries, statistics


def _save_figure(fig: plt.Figure, figure_folder: Path, stem: str) -> None:
    fig.savefig(figure_folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_folder / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_state(
    figure_folder: Path,
    entries: list[tuple[float, Path]],
) -> None:
    final_time, final_path = entries[-1]
    mesh = meshio.read(str(final_path))
    triangles = mesh.cells_dict["triangle"]
    polygons = mesh.points[triangles, :2]
    data = mesh.cell_data_dict

    rho_w = 998.21
    rho_g = 1.8152
    initial_sg = 0.2
    z_co2 = np.asarray(data["z_CO2"]["triangle"]).ravel()
    s_gas = rho_w * z_co2 / (rho_w * z_co2 + rho_g * (1.0 - z_co2))
    delta_sg = s_gas - initial_sg
    capillary_pressure = np.asarray(data["capillary_pressure"]["triangle"]).ravel()
    entry_pressure = np.asarray(data["entry_pressure"]["triangle"]).ravel()

    plot_data = (
        (
            delta_sg,
            "Gas-saturation increase $S_n-S_n^0$",
            "viridis",
            None,
            (0.0, 0.6),
        ),
        (
            capillary_pressure,
            "Capillary pressure $p_c$ (Pa)",
            "magma",
            LogNorm(vmin=1.0, vmax=2000.0),
            None,
        ),
        (
            entry_pressure,
            "Facies entry pressure $p_e$ (Pa)",
            "cividis",
            LogNorm(vmin=1.0, vmax=1500.0),
            None,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    wells = (case_A["WELL_1_POS"], case_A["WELL_2_POS"])
    for ax, (values, title, cmap, norm, limits) in zip(axes, plot_data):
        collection = PolyCollection(
            polygons,
            array=values,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            linewidths=0,
        )
        if limits is not None:
            collection.set_clim(*limits)
        ax.add_collection(collection)
        ax.set_xlim(0.0, float(case_A["WIDTH"]))
        ax.set_ylim(0.0, float(case_A["HEIGHT"]))
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_title(title)
        fig.colorbar(collection, ax=ax, shrink=0.82, pad=0.02)
        ax.plot(
            [position[0] for position in wells],
            [position[1] for position in wells],
            marker="*",
            color="red",
            linestyle="none",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    fig.suptitle(
        f"SPE11A pilot with active capillary pressure, $t={final_time:.1f}$ s",
        fontsize=13,
    )
    _save_figure(fig, figure_folder, FIGURE_STEMS[0])


def _plot_attempts(
    figure_folder: Path,
    entries: list[tuple[float, Path]],
    statistics: dict,
) -> None:
    trials = sorted(
        (
            (int(key), value)
            for key, value in statistics.items()
            if key != "global" and key.isdigit()
        ),
        key=lambda item: item[0],
    )
    x = np.array([index + 1 for index, _ in trials])
    iterations = np.array([value["num_iterations"] for _, value in trials])
    dt = np.array([value["dt"] for _, value in trials], dtype=float)
    successful = np.array(
        [value["solver_status"] == "successful" for _, value in trials]
    )
    final_time = entries[-1][0]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    for ax, values, ylabel in zip(
        axes,
        (iterations, dt),
        ("Newton iterations", "Attempted time step $\\Delta t$ (s)"),
    ):
        ax.plot(x, values, color="0.65", linewidth=0.9, zorder=1)
        ax.scatter(
            x[successful],
            values[successful],
            color="#0072B2",
            marker="o",
            s=25,
            label="Accepted",
            zorder=3,
        )
        ax.scatter(
            x[~successful],
            values[~successful],
            color="#D55E00",
            marker="x",
            s=65,
            linewidths=1.8,
            label="Failed / retried",
            zorder=4,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", frameon=True)
    axes[0].axhline(20, color="black", linestyle="--", linewidth=1)
    axes[0].annotate(
        "20-iteration limit",
        xy=(0.99, 20),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Chronological time-step attempt")
    fig.suptitle(
        f"Newton cost and adaptive time steps "
        f"(reached $t={final_time:.1f}$ s; target 300 s)",
        fontsize=12,
    )
    _save_figure(fig, figure_folder, FIGURE_STEMS[1])


def _plot_failed_histories(figure_folder: Path, statistics: dict) -> None:
    trials = sorted(
        (
            (int(key), value)
            for key, value in statistics.items()
            if key != "global" and key.isdigit()
        ),
        key=lambda item: item[0],
    )
    failed = [
        (index, record)
        for index, record in trials
        if record["solver_status"] != "successful"
    ]
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for (index, record), color in zip(failed, colors):
        label = f"attempt {index + 1}, $t^*={record['time']:.2f}$ s"
        info = record["convergence_info"]
        metrics = (
            (
                axes[0],
                np.asarray(info.get("res_abs", []), dtype=float),
                "$\\|F\\|$",
                1e-10,
            ),
            (
                axes[1],
                np.asarray(info.get("inc_abs", []), dtype=float),
                "$\\|\\delta x\\|$",
                2e-7,
            ),
        )
        for ax, values, ylabel, tolerance in metrics:
            valid = np.isfinite(values) & (values > 0)
            ax.plot(
                np.arange(1, len(values) + 1)[valid],
                values[valid],
                marker=".",
                markersize=3,
                linewidth=1.2,
                color=color,
                label=label,
            )
            ax.axhline(tolerance, color="black", linestyle=":", linewidth=1)
            ax.set_yscale("log")
            ax.set_xlabel("Newton iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)

    axes[0].set_title("Absolute residual; tolerance $10^{-10}$")
    axes[1].set_title("Absolute increment; tolerance $2\\times10^{-7}$")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Convergence histories in the failed time-step attempts", fontsize=12)
    _save_figure(fig, figure_folder, FIGURE_STEMS[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot physical and Newton diagnostics for an SPE11A capillary run."
    )
    parser.add_argument("--run-folder", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--figure-folder", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()

    run_folder = args.run_folder.resolve()
    figure_folder = args.figure_folder.resolve()
    figure_folder.mkdir(parents=True, exist_ok=True)
    entries, statistics = _read_outputs(run_folder)
    _plot_state(figure_folder, entries)
    _plot_attempts(figure_folder, entries, statistics)
    _plot_failed_histories(figure_folder, statistics)

    for stem in FIGURE_STEMS:
        print(f"Wrote {figure_folder / (stem + '.pdf')}")
        print(f"Wrote {figure_folder / (stem + '.png')}")


if __name__ == "__main__":
    main()
