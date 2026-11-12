"""Nonlinear-solver cost for the entry-pressure study.

Reads ``solver_statistics.json`` from each run and produces, in ``figurer_results/``:

All figure names carry the suffix ``_vary_upper`` or ``_vary_lower``.

* ``fig_solver_cost_vs_pe``   -- total Newton iterations (split into useful and wasted),
                                 accepted/failed time-step attempts and mean iterations
                                 per accepted step, as functions of the varied entry
                                 pressure;
* ``fig_solver_iterations``   -- Newton iterations per attempted time step for every
                                 case (accepted: circles, failed: crosses);
* ``fig_solver_dt``           -- time-step size against simulated time for every case,
                                 with failed attempts marked;
* ``fig_solver_newton_histories`` -- per-iteration convergence histories of selected
                                 attempts (quadratic vs. divergent behaviour);
* ``tab_solver_cost.tex``     -- cost table for \\input in the thesis.

Usage::

    python plot_results_solver_cost.py --vary upper --fixed 50 [--cases 100 500 2000 60000]
    python plot_results_solver_cost.py --vary lower --fixed 60000
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import results_common as rc


def select_runs(vary: str, fixed: float, cases: list[float] | None) -> list[rc.RunInfo]:
    if vary == "upper":
        runs = rc.find_runs(pe_lower=fixed)
        key = lambda r: r.pe_upper  # noqa: E731
    else:
        runs = rc.find_runs(pe_upper=fixed)
        key = lambda r: r.pe_lower  # noqa: E731
    if cases:
        runs = [r for r in runs if any(np.isclose(key(r), c) for c in cases)]
    runs = [r for r in runs if rc.read_solver_statistics(r.folder) is not None]
    if not runs:
        raise SystemExit("No runs with solver statistics found.")
    return sorted(runs, key=key)


def varied_value(run: rc.RunInfo, vary: str) -> float:
    return run.pe_upper if vary == "upper" else run.pe_lower


def cost_summary(run: rc.RunInfo) -> dict[str, float]:
    stats = rc.read_solver_statistics(run.folder)
    attempts = rc.solver_attempts(stats)
    accepted = [a for a in attempts if a.success]
    failed = [a for a in attempts if not a.success]
    total = sum(a.iterations for a in attempts)
    wasted = sum(a.iterations for a in failed)
    return {
        "attempts": len(attempts),
        "accepted": len(accepted),
        "failed": len(failed),
        "total_iterations": total,
        "wasted_iterations": wasted,
        "mean_iterations": total / max(len(accepted), 1),
        "mean_iterations_accepted": np.mean([a.iterations for a in accepted]) if accepted else np.nan,
        "max_iterations_accepted": max((a.iterations for a in accepted), default=0),
        "final_time": run.final_time,
        "completed": run.completed,
        "min_dt": min((a.dt for a in attempts), default=np.nan),
    }


def cost_figure(runs: list[rc.RunInfo], summaries: list[dict[str, float]], vary: str) -> None:
    x = np.array([varied_value(r, vary) for r in runs])
    labels = [rc.entry_pressure_label(v) for v in x]
    pos = np.arange(len(x))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ax_it, ax_steps, ax_mean = axes

    useful = np.array([s["total_iterations"] - s["wasted_iterations"] for s in summaries])
    wasted = np.array([s["wasted_iterations"] for s in summaries])
    ax_it.bar(pos, useful, color=rc.PALETTE[0], label="in accepted steps")
    ax_it.bar(pos, wasted, bottom=useful, color=rc.PALETTE[1], label="wasted (failed steps)")
    ax_it.set_ylabel("Newton iterations, total")
    ax_it.set_ylim(0, 1.35 * max(useful + wasted))
    ax_it.legend(fontsize=7.5, loc="upper left")

    acc = np.array([s["accepted"] for s in summaries])
    fail = np.array([s["failed"] for s in summaries])
    ax_steps.bar(pos, acc, color=rc.PALETTE[0], label="accepted")
    ax_steps.bar(pos, fail, bottom=acc, color=rc.PALETTE[1], label="failed")
    ax_steps.set_ylabel("time-step attempts")
    ax_steps.set_ylim(0, 1.35 * max(acc + fail))
    ax_steps.legend(fontsize=7.5, loc="upper left")

    mean = np.array([s["mean_iterations_accepted"] for s in summaries])
    mx = np.array([s["max_iterations_accepted"] for s in summaries])
    ax_mean.plot(pos, mean, "o-", color=rc.PALETTE[0], label="mean per accepted step")
    ax_mean.plot(pos, mx, "s--", color=rc.PALETTE[1], label="max per accepted step")
    ax_mean.axhline(rc.PARAMS["nl_max_iterations"], color="k", ls=":", lw=0.9, label="iteration limit")
    ax_mean.set_ylabel("Newton iterations per step")
    ax_mean.set_ylim(0, 1.35 * rc.PARAMS["nl_max_iterations"])
    ax_mean.legend(fontsize=7.5, loc="upper left")

    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("$p_{e,\\mathrm{%s}}$" % ("up" if vary == "upper" else "low"))
        for i, r in enumerate(runs):
            if not r.completed:
                ax.get_xticklabels()[i].set_color("red")
    fig.suptitle(
        "Solver cost; red tick labels: run did not reach $t_{\\mathrm{final}}$", fontsize=9
    )
    fig.tight_layout()
    rc.save_figure(fig, f"fig_solver_cost_vs_pe_vary_{vary}")


def iterations_figure(runs: list[rc.RunInfo], vary: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for color, run in zip(rc.PALETTE, runs):
        attempts = rc.solver_attempts(rc.read_solver_statistics(run.folder))
        idx = np.array([a.index for a in attempts])
        it = np.array([a.iterations for a in attempts])
        ok = np.array([a.success for a in attempts])
        ax.plot(idx, it, color=color, lw=0.8, alpha=0.6)
        ax.plot(idx[ok], it[ok], "o", color=color, ms=4, label=rc.entry_pressure_label(varied_value(run, vary)))
        ax.plot(idx[~ok], it[~ok], "x", color=color, ms=7, mew=1.5)
    ax.axhline(rc.PARAMS["nl_max_iterations"], color="k", ls=":", lw=0.9)
    ax.set_xlabel("time-step attempt (chronological)")
    ax.set_ylabel("Newton iterations")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], marker="o", color="0.3", ls="None", label="accepted"),
        Line2D([], [], marker="x", color="0.3", ls="None", mew=1.5, label="failed"),
    ]
    ax.legend(handles=handles, fontsize=7.5, ncol=2, title="$p_{e,\\mathrm{%s}}$" % ("up" if vary == "upper" else "low"))
    fig.tight_layout()
    rc.save_figure(fig, f"fig_solver_iterations_vary_{vary}")


def dt_figure(runs: list[rc.RunInfo], vary: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for color, run in zip(rc.PALETTE, runs):
        attempts = rc.solver_attempts(rc.read_solver_statistics(run.folder))
        t = np.array([a.time for a in attempts])
        dt = np.array([a.dt for a in attempts])
        ok = np.array([a.success for a in attempts])
        ax.plot(t[ok], dt[ok], "o-", color=color, ms=3, lw=0.8, label=rc.entry_pressure_label(varied_value(run, vary)))
        ax.plot(t[~ok], dt[~ok], "x", color=color, ms=7, mew=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("simulated time at end of attempt [s]")
    ax.set_ylabel("$\\Delta t$ [s]")
    ax.legend(fontsize=7.5, ncol=2, title="$p_{e,\\mathrm{%s}}$ (x: failed)" % ("up" if vary == "upper" else "low"))
    fig.tight_layout()
    rc.save_figure(fig, f"fig_solver_dt_vary_{vary}")


def histories_figure(runs: list[rc.RunInfo], vary: str) -> None:
    """Per-iteration histories: one typical accepted step and the worst attempt per case."""
    fig, (ax_inc, ax_res) = plt.subplots(1, 2, figsize=(8.4, 3.8), layout="constrained")
    for color, run in zip(rc.PALETTE, runs):
        attempts = rc.solver_attempts(rc.read_solver_statistics(run.folder))
        accepted = [a for a in attempts if a.success and a.inc_abs]
        if not accepted:
            continue
        typical = min(accepted, key=lambda a: abs(a.iterations - np.median([b.iterations for b in accepted])))
        worst = max(attempts, key=lambda a: (a.iterations, not a.success, a.index))
        label = rc.entry_pressure_label(varied_value(run, vary))
        for attempt, ls in ((typical, "-"), (worst, "--")):
            k = np.arange(1, len(attempt.inc_abs) + 1)
            ax_inc.plot(k, attempt.inc_abs, ls, color=color, marker=".", label=f"{label}, attempt {attempt.index}" + ("" if attempt.success else " (failed)"))
            ax_res.plot(k, attempt.res_abs, ls, color=color, marker=".")
    ax_inc.set_yscale("log")
    ax_res.set_yscale("log")
    ax_inc.set_xlabel("Newton iteration $k$")
    ax_res.set_xlabel("Newton iteration $k$")
    ax_inc.set_ylabel("$\\|\\delta x^k\\|$ (absolute increment)")
    ax_res.set_ylabel("$\\|F(x^k)\\|$ (absolute residual)")
    ax_inc.axhline(rc.PARAMS["nl_convergence_inc_atol"], color="k", ls=":", lw=0.8)
    ax_res.axhline(rc.PARAMS["nl_convergence_res_atol"], color="k", ls=":", lw=0.8)
    handles, labels = ax_inc.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=min(len(labels), 5), loc="outside lower center")
    fig.suptitle("solid: typical accepted step, dashed: most expensive attempt; dotted: tolerance", fontsize=8.5)
    rc.save_figure(fig, f"fig_solver_newton_histories_vary_{vary}")


def cost_table(runs: list[rc.RunInfo], summaries: list[dict[str, float]], vary: str) -> None:
    rows = []
    for r, s in zip(runs, summaries):
        rows.append(
            [
                f"{varied_value(r, vary):g}",
                f"{s['final_time']:.6g}" + ("" if s["completed"] else "$^{*}$"),
                f"{int(s['accepted'])}",
                f"{int(s['failed'])}",
                f"{int(s['total_iterations'])}",
                f"{int(s['wasted_iterations'])}",
                f"{s['mean_iterations_accepted']:.1f}",
                f"{int(s['max_iterations_accepted'])}",
                f"{s['min_dt']:.3g}",
            ]
        )
    name = "up" if vary == "upper" else "low"
    rc.write_latex_table(
        f"tab_solver_cost_vary_{vary}",
        [
            f"$p_{{e,\\mathrm{{{name}}}}}$ [Pa]",
            "$t_{\\mathrm{end}}$ [s]",
            "accepted",
            "failed",
            "Newton it.",
            "wasted it.",
            "mean it.",
            "max it.",
            "min $\\Delta t$ [s]",
        ],
        rows,
        "Nonlinear solver cost. \\emph{accepted}/\\emph{failed}: number of time-step "
        "attempts; \\emph{Newton it.}: total number of Newton iterations; \\emph{wasted}: "
        "iterations spent in failed attempts; \\emph{mean}/\\emph{max}: per accepted "
        "step. $^{*}$Run did not reach the final time.",
        f"tab:solver_cost_vary_{vary}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vary", choices=("upper", "lower"), default="upper")
    parser.add_argument("--fixed", type=float, default=50.0, help="Value of the fixed entry pressure [Pa].")
    parser.add_argument("--cases", type=float, nargs="*", default=None)
    args = parser.parse_args()
    rc.set_style()
    runs = select_runs(args.vary, args.fixed, args.cases)
    summaries = [cost_summary(r) for r in runs]
    for r, s in zip(runs, summaries):
        print(
            f"{r.folder.name}: {int(s['accepted'])} accepted, {int(s['failed'])} failed, "
            f"{int(s['total_iterations'])} iterations ({int(s['wasted_iterations'])} wasted), "
            f"t_end={s['final_time']:g}"
        )
    cost_figure(runs, summaries, args.vary)
    iterations_figure(runs, args.vary)
    dt_figure(runs, args.vary)
    histories_figure(runs, args.vary)
    cost_table(runs, summaries, args.vary)


if __name__ == "__main__":
    main()
