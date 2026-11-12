"""Figures for the entry-pressure sensitivity study (fixed lower-layer entry pressure,
varying upper-layer entry pressure).

Produces, in ``figurer_results/``:

* ``fig_study1_sg_final``      -- final-time gas-saturation change for every case,
                                  common colour scale, interface and well overlaid;
* ``fig_study1_timeseries``    -- CO2 mass in the upper layer, CO2 mass added to the
                                  lower layer, maximum capillary pressure below the
                                  interface, and pressure change in the trap vs time;
* ``fig_study1_massbalance``   -- CO2 mass in the domain vs injected mass (validation);
* ``fig_study1_summary_vs_pe`` -- fraction of injected CO2 found in the upper layer and
                                  the capillary-pressure ratio at the interface at the
                                  final time, as functions of the upper entry pressure;
* ``tab_study1_summary.tex``   -- the same numbers as a table.

Usage::

    python plot_results_entry_pressure_study.py --lower 50 [--cases 100 500 2000 60000]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import results_common as rc


def select_runs(lower: float, cases: list[float] | None) -> list[rc.RunInfo]:
    runs = rc.find_runs(pe_lower=lower)
    if cases:
        runs = [r for r in runs if any(np.isclose(r.pe_upper, c) for c in cases)]
    if not runs:
        raise SystemExit(f"No runs with p_e,lower={lower} found in {rc.RUNS_FOLDER}")
    for run in runs:
        flag = "" if run.completed else "  <-- did NOT reach final time"
        print(f"{run.folder.name}: t_final={run.final_time:g} s{flag}")
    return runs


def final_fields_figure(runs: list[rc.RunInfo]) -> None:
    snaps = [rc.read_snapshot(rc.read_pvd(r.folder)[-1][1]) for r in runs]
    s_init = rc.PARAMS["s_gas_init_lower"]
    deltas = [rc.gas_saturation_from_z(s.fields["z_CO2"]) - s_init for s in snaps]
    vmax = max(float(np.max(d)) for d in deltas)
    vmax = max(vmax, 1e-6)
    n = len(runs)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 2.3 * nrows + 0.6), squeeze=False, layout="constrained"
    )
    for ax, run, snap, delta in zip(axes.ravel(), runs, snaps, deltas):
        coll = rc.draw_field(
            ax,
            snap,
            delta,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            title=f"$p_{{e,\\mathrm{{up}}}}$ = {rc.entry_pressure_label(run.pe_upper)}"
            + ("" if run.completed else f" (stopped at {run.final_time:.0f} s)"),
        )
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(
        coll, ax=axes.ravel().tolist(), shrink=0.8, label="$S_n - S_{n,\\mathrm{init}}$ [-]"
    )
    rc.save_figure(fig, "fig_study1_sg_final")


def timeseries_figure(runs: list[rc.RunInfo], series: list[dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))
    (ax_up, ax_low), (ax_pc, ax_p) = axes
    for color, run, ts in zip(rc.PALETTE, runs, series):
        label = f"$p_{{e,\\mathrm{{up}}}}$ = {rc.entry_pressure_label(run.pe_upper)}"
        t = ts["time"]
        ax_up.plot(t, 1e3 * (ts["mass_upper"] - ts["mass_upper"][0]), color=color, label=label)
        ax_low.plot(t, 1e3 * (ts["mass_lower"] - ts["mass_lower"][0]), color=color, label=label)
        ax_pc.plot(t, ts["pc_max_below_interface"], color=color, label=label)
        pc_upper_init = rc.capillary_pressure(
            np.array([1 - rc.PARAMS["s_gas_init_upper"]]), run.pe_upper
        )[0]
        ax_pc.axhline(pc_upper_init, color=color, ls=":", lw=0.9)
        # Pressure change relative to the first post-initial output (the t=0 state
        # is hydrostatic without capillary offset and therefore not comparable).
        ref = 1 if len(t) > 1 else 0
        ax_p.plot(t[ref:], ts["p_trap"][ref:] - ts["p_trap"][ref], color=color, label=label)
    ax_low.plot(
        series[0]["time"], 1e3 * series[0]["injected"], "k--", lw=0.9, label="injected"
    )
    ax_up.set_ylabel("CO$_2$ mass added to upper layer [g]")
    ax_low.set_ylabel("CO$_2$ mass added to lower layer [g]")
    ax_pc.set_ylabel("max $p_c$ just below interface [Pa]")
    ax_pc.set_yscale("log")
    ax_p.set_ylabel("$\\Delta p_n$ in trap [Pa]")
    for ax in axes.ravel():
        ax.set_xlabel("time [s]")
    ax_low.legend(fontsize=7.5)
    ax_pc.text(
        0.02,
        0.97,
        "dotted: $p_c$ of upper layer at $S_n=0.2$",
        transform=ax_pc.transAxes,
        va="top",
        fontsize=7.5,
    )
    fig.tight_layout()
    rc.save_figure(fig, "fig_study1_timeseries")


def massbalance_figure(runs: list[rc.RunInfo], series: list[dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for color, run, ts in zip(rc.PALETTE, runs, series):
        ax.plot(
            ts["time"],
            1e3 * (ts["mass_total"] - ts["mass_total"][0]),
            color=color,
            label=f"$p_{{e,\\mathrm{{up}}}}$ = {rc.entry_pressure_label(run.pe_upper)}",
        )
    ax.plot(series[0]["time"], 1e3 * series[0]["injected"], "k--", lw=1, label="injected $q\\,t$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("CO$_2$ mass in domain $-$ initial [g]")
    ax.set_title("CO$_2$ inventory and injected mass")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    rc.save_figure(fig, "fig_study1_massbalance")


def summary(runs: list[rc.RunInfo], series: list[dict[str, np.ndarray]]) -> None:
    pe = np.array([r.pe_upper for r in runs])
    frac_upper = np.array(
        [
            (ts["mass_upper"][-1] - ts["mass_upper"][0]) / max(ts["injected"][-1], 1e-30)
            for ts in series
        ]
    )
    retained = np.array(
        [
            (ts["mass_total"][-1] - ts["mass_total"][0]) / max(ts["injected"][-1], 1e-30)
            for ts in series
        ]
    )
    pc_ratio = np.array(
        [
            ts["pc_max_below_interface"][-1]
            / rc.capillary_pressure(np.array([1 - rc.PARAMS["s_gas_init_upper"]]), r.pe_upper)[0]
            for r, ts in zip(runs, series)
        ]
    )
    dp_trap = np.array(
        [ts["p_trap"][-1] - ts["p_trap"][min(1, len(ts["p_trap"]) - 1)] for ts in series]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax1.plot(pe, 100 * frac_upper, "o-", color=rc.PALETTE[0])
    ax1.set_xscale("log")
    ax1.set_xlabel("$p_{e,\\mathrm{up}}$ [Pa]")
    ax1.set_ylabel("added upper inventory / injected mass [%]")
    ax1.set_title("Upper-layer inventory at $t_{\\mathrm{final}}$")
    ax2.plot(pe, pc_ratio, "s-", color=rc.PALETTE[1])
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("$p_{e,\\mathrm{up}}$ [Pa]")
    ax2.set_ylabel("$\\max p_c^{\\mathrm{below}} / p_c^{\\mathrm{up}}(S_n{=}0.2)$ [-]")
    ax2.set_title("Capillary entry condition at the interface")
    for ax, run in ((ax1, None), (ax2, None)):
        for r in runs:
            if not r.completed:
                ax.axvline(r.pe_upper, color="0.6", ls="--", lw=0.8)
    fig.tight_layout()
    rc.save_figure(fig, "fig_study1_summary_vs_pe")

    rows = []
    for r, f, ret, pr, dp in zip(runs, frac_upper, retained, pc_ratio, dp_trap):
        rows.append(
            [
                f"{r.pe_upper:g}",
                f"{r.final_time:.0f}" + ("" if r.completed else "$^{*}$"),
                f"{100 * ret:.4f}",
                f"{100 * f:.5f}",
                f"{pr:.5f}",
                f"{dp:.4g}",
            ]
        )
    rc.write_latex_table(
        "tab_study1_summary",
        [
            "$p_{e,\\mathrm{up}}$ [Pa]",
            "$t_{\\mathrm{end}}$ [s]",
            "retained [\\%]",
            "in upper layer [\\%]",
            "$p_c$ ratio",
            "$\\Delta p_n$ trap [Pa]",
        ],
        rows,
        "Summary of the entry-pressure study at the final output time. "
        "\\emph{retained}: CO$_2$ mass added to the domain relative to the injected "
        "mass; \\emph{in upper layer}: net added inventory above the interface "
        "relative to injected mass (not cumulative transmission); "
        "$p_c$ ratio: maximum capillary pressure just below the interface divided by "
        "the capillary pressure of the upper layer at its initial saturation; "
        "$\\Delta p_n$: non-wetting pressure change below the crest of the interface. "
        "$^{*}$Run did not reach the final time.",
        "tab:study1_summary",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=float, default=50.0, help="Fixed lower entry pressure [Pa].")
    parser.add_argument("--cases", type=float, nargs="*", default=None, help="Upper entry pressures to include.")
    args = parser.parse_args()
    rc.set_style()
    rc.check_parameters_against_run_script()
    runs = select_runs(args.lower, args.cases)
    series = [rc.time_series(r) for r in runs]
    final_fields_figure(runs)
    timeseries_figure(runs, series)
    massbalance_figure(runs, series)
    summary(runs, series)


if __name__ == "__main__":
    main()
