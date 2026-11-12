from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RUNS_FOLDER = ROOT / "visualization_capillary_runs"
LEGACY_STATISTICS = ROOT / "visualization_capillary" / "solver_statistics_no_inj_simple.json"


def _last_number(values: Any) -> float:
    """Return the last finite scalar in a per-iteration statistics value."""
    if not isinstance(values, list):
        values = [values]
    for value in reversed(values):
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return float("nan")


def _case_label(path: Path) -> str:
    if path == LEGACY_STATISTICS:
        return "Legacy run (entry pressures not recorded)"
    label = path.parent.name.split("__", maxsplit=1)[0]
    label = label.replace("pe_lower_", "lower ").replace("_upper_", ", upper ")
    return f"{label.replace('_', ' ')} Pa"


def _read_case(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    entries: list[dict[str, Any]] = []
    for key, record in data.items():
        if key == "global" or not isinstance(record, dict):
            continue
        try:
            int(key)
        except ValueError:
            continue

        convergence_info = record.get("convergence_info", {})
        metrics = {
            name: _last_number(convergence_info.get(name))
            for name in ("inc_abs", "inc_rel", "res_abs", "res_rel")
        }
        if metrics["res_rel"] == 0.0:
            residual_history = convergence_info.get("res_abs", [])
            if isinstance(residual_history, list):
                residuals = [
                    float(value)
                    for value in residual_history
                    if isinstance(value, (int, float))
                    and np.isfinite(value)
                    and value > 0
                ]
                if residuals:
                    metrics["res_rel"] = residuals[-1] / residuals[0]

        entries.append(
            {
                "trial": int(key),
                "time": float(record.get("time", record.get("time_index", 0.0))),
                "iterations": int(record.get("num_iterations", 0)),
                "success": record.get("solver_status") == "successful",
                "metrics": metrics,
            }
        )

    entries.sort(key=lambda entry: int(entry["trial"]))
    return {"label": _case_label(path), "entries": entries}


def _statistics_files() -> list[Path]:
    files = sorted(RUNS_FOLDER.glob("*/solver_statistics.json"))
    if LEGACY_STATISTICS.exists():
        files.append(LEGACY_STATISTICS)
    return files


def plot_statistics() -> None:
    RUNS_FOLDER.mkdir(parents=True, exist_ok=True)
    files = _statistics_files()
    if not files:
        print(f"No solver statistics found under {RUNS_FOLDER}")
        return

    cases = [_read_case(path) for path in files]
    cases = [case for case in cases if case["entries"]]
    if not cases:
        print("Statistics files were found, but contain no time-step entries.")
        return

    case_palette = [
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#E69F00",
        "#332288",
        "#882255",
    ]
    colors = [case_palette[index % len(case_palette)] for index in range(len(cases))]
    case_handles = []
    status_handles = [
        Line2D([], [], color="0.25", marker="o", linestyle="None", label="Accepted"),
        Line2D([], [], color="0.25", marker="x", linestyle="None", label="Failed/retried"),
    ]

    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for color, case in zip(colors, cases):
        entries = case["entries"]
        trials = np.array([entry["trial"] for entry in entries])
        iterations = np.array([entry["iterations"] for entry in entries])
        axis.plot(trials, iterations, color=color, alpha=0.45, linewidth=1)
        for success, marker in ((True, "o"), (False, "x")):
            selected = np.array([entry["success"] == success for entry in entries])
            axis.scatter(
                trials[selected],
                iterations[selected],
                color=[color],
                marker=marker,
                s=48,
                zorder=3,
            )
        case_handles.append(
            Line2D([], [], color=color, marker="o", label=case["label"])
        )

    axis.set_title("Newton iterations by attempted time step")
    axis.set_xlabel("Time-step attempt (chronological order)")
    axis.set_ylabel("Newton iterations")
    axis.set_ylim(0, max(entry["iterations"] for case in cases for entry in case["entries"]) + 1)
    axis.grid(True, which="both", alpha=0.25)
    case_legend = axis.legend(
        handles=case_handles, title="Entry-pressure case", loc="upper left"
    )
    axis.add_artist(case_legend)
    axis.legend(
        handles=status_handles, title="Time-step result", loc="lower right"
    )
    iteration_path = RUNS_FOLDER / "newton_iterations_by_entry_pressure.png"
    fig.savefig(iteration_path, dpi=180)
    plt.close(fig)

    metric_titles = {
        "inc_abs": "Absolute increment",
        "inc_rel": "Relative increment",
        "res_abs": "Absolute residual",
        "res_rel": "Relative residual",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric_name in zip(axes.ravel(), metric_titles):
        has_values = False
        for color, case in zip(colors, cases):
            entries = case["entries"]
            trials = np.array([entry["trial"] for entry in entries])
            values = np.array([entry["metrics"][metric_name] for entry in entries])
            valid = np.isfinite(values) & (values > 0)
            successes = np.array([entry["success"] for entry in entries])
            has_values = has_values or bool(np.any(valid))
            axis.plot(
                trials[valid], values[valid], color=color, alpha=0.45, linewidth=1
            )
            for success, marker in ((True, "o"), (False, "x")):
                selected = valid & (successes == success)
                axis.scatter(
                    trials[selected],
                    values[selected],
                    color=[color],
                    marker=marker,
                    s=34,
                    zorder=3,
                )
        if has_values:
            axis.set_yscale("log")
        else:
            axis.text(
                0.5,
                0.5,
                "No finite data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(metric_titles[metric_name])
        axis.set_xlabel("Time-step attempt (chronological order)")
        axis.set_ylabel("Norm")
        axis.grid(True, which="both", alpha=0.25)

    fig.legend(handles=case_handles, title="Entry-pressure case", loc="outside upper center", ncols=2)
    fig.legend(handles=status_handles, title="Time-step result", loc="outside lower center", ncols=2)
    convergence_path = RUNS_FOLDER / "newton_convergence_by_entry_pressure.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)

    print(f"Compared {len(cases)} statistics file(s):")
    for case in cases:
        print(f"  {case['label']}: {len(case['entries'])} trial(s)")
    print(f"Wrote {iteration_path}")
    print(f"Wrote {convergence_path}")


if __name__ == "__main__":
    plot_statistics()