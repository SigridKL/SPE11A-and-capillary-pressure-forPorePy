"""Figures for the long-term injection study at fixed entry pressures.

The study keeps both entry pressures fixed (default 50 Pa / 500 Pa) and injects for a
long time, so that the only varying quantity is time, i.e. the injected CO2 volume.
The figures answer: does CO2 accumulate under the trap, does the pressure build up,
and does the seal eventually fail (capillary breakthrough)?

Produces, in ``figurer_results/``:

* ``fig_study2_timeseries`` -- (a) CO2 mass below/above the interface and injected
                               mass; (b) maximum capillary pressure just below the
                               interface against the entry condition of the upper
                               layer; (c) pressure build-up at the monitor points;
                               (d) maximum gas saturation per layer;
* ``fig_study2_snapshots``  -- gas saturation at selected times.

The breakthrough time (first output where CO2 mass above the interface exceeds a
threshold fraction of the injected mass) is printed and written to
``study2_breakthrough.txt``.

Usage::

    python plot_results_long_injection.py [--lower 50 --upper 500] [--run FOLDER]
                                          [--snapshot-times 300 1000 ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import results_common as rc

BREAKTHROUGH_FRACTION = 0.01  # 1 % of injected mass above the interface


def select_run(args: argparse.Namespace) -> rc.RunInfo:
    if args.run:
        folder = Path(args.run)
        if not folder.is_absolute():
            folder = rc.RUNS_FOLDER / folder
        match = rc.PRESSURE_PATTERN.match(folder.name)
        if match is None:
            raise SystemExit(f"Folder name {folder.name} does not follow the run pattern.")
        runs = [r for r in rc.find_runs(latest_only=False) if r.folder == folder]
        if not runs:
            raise SystemExit(f"Run {folder} not found.")
        return runs[0]
    runs = rc.find_runs(pe_lower=args.lower, pe_upper=args.upper)
    if not runs:
        raise SystemExit(
            f"No run with p_e = {args.lower}/{args.upper} Pa in {rc.RUNS_FOLDER}."
        )
    run = runs[0]
    print(f"Using {run.folder.name} (t_final = {run.final_time:g} s)")
    return run


def breakthrough_time(ts: dict[str, np.ndarray]) -> float | None:
    added_upper = ts["mass_upper"] - ts["mass_upper"][0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(ts["injected"] > 0, added_upper / ts["injected"], 0.0)
    hits = np.nonzero(frac > BREAKTHROUGH_FRACTION)[0]
    return float(ts["time"][hits[0]]) if hits.size else None


def timeseries_figure(
    run: rc.RunInfo, ts: dict[str, np.ndarray], suffix: str = ""
) -> None:
    t = ts["time"]
    pc_entry_upper = rc.capillary_pressure(
        np.array([1 - rc.PARAMS["s_gas_init_upper"]]), run.pe_upper
    )[0]
    t_bt = breakthrough_time(ts)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8))
    (ax_m, ax_pc), (ax_p, ax_s) = axes

    ax_m.plot(t, 1e3 * (ts["mass_lower"] - ts["mass_lower"][0]), color=rc.PALETTE[0], label="below interface")
    ax_m.plot(t, 1e3 * (ts["mass_upper"] - ts["mass_upper"][0]), color=rc.PALETTE[1], label="above interface")
    ax_m.plot(t, 1e3 * ts["injected"], "k--", lw=0.9, label="injected")
    ax_m.set_ylabel("CO$_2$ mass added [g]")
    ax_m.legend(fontsize=7.5)
    ax_m.set_title("(a) CO$_2$ inventory")

    ax_pc.plot(t, ts["pc_max_below_interface"], color=rc.PALETTE[0], label="max $p_c$ just below interface")
    ax_pc.axhline(pc_entry_upper, color=rc.PALETTE[1], ls="--", lw=1, label="$p_c$ of upper layer at $S_n=0.2$")
    ax_pc.axhline(run.pe_upper, color=rc.PALETTE[1], ls=":", lw=1, label="$p_{e,\\mathrm{up}}$")
    ax_pc.set_ylabel("capillary pressure [Pa]")
    ax_pc.set_yscale("log")
    ax_pc.legend(fontsize=7.5)
    ax_pc.set_title("(b) Capillary entry condition")

    ref = 1 if len(t) > 1 else 0
    for color, key, label in (
        (rc.PALETTE[0], "p_trap", "below crest (trap)"),
        (rc.PALETTE[1], "p_above_trap", "above crest"),
        (rc.PALETTE[2], "p_well", "well"),
    ):
        ax_p.plot(t[ref:], ts[key][ref:] - ts[key][ref], color=color, label=label)
    ax_p.set_ylabel("$\\Delta p_n$ since first step [Pa]")
    ax_p.legend(fontsize=7.5)
    ax_p.set_title("(c) Pressure build-up")

    ax_s.plot(t, ts["sg_max_lower"], color=rc.PALETTE[0], label="lower layer")
    ax_s.plot(t, ts["sg_max_upper"], color=rc.PALETTE[1], label="upper layer")
    ax_s.axhline(1 - rc.PARAMS["sw_imm"], color="k", ls=":", lw=0.9, label="$1-S_{w,\\mathrm{imm}}$")
    ax_s.set_ylabel("max $S_n$ [-]")
    ax_s.legend(fontsize=7.5)
    ax_s.set_title("(d) Maximum gas saturation")

    for ax in axes.ravel():
        ax.set_xlabel("time [s]")
        if t_bt is not None:
            ax.axvline(t_bt, color="0.4", ls="-.", lw=0.9)
    if t_bt is not None:
        ax_m.text(t_bt, ax_m.get_ylim()[1] * 0.95, f" breakthrough\n $t\\approx{t_bt:.0f}$ s", fontsize=7.5, va="top")
    fig.suptitle(run.label, fontsize=10)
    fig.tight_layout()
    rc.save_figure(fig, f"fig_study2_timeseries{suffix}")

    note = rc.FIG_FOLDER / f"study2_breakthrough{suffix}.txt"
    added_upper = ts["mass_upper"] - ts["mass_upper"][0]
    fraction = np.divide(
        added_upper, ts["injected"], out=np.zeros_like(added_upper),
        where=ts["injected"] > 0,
    )
    text = (
        f"run: {run.folder.name}\nfinal time: {run.final_time:g} s\n"
        f"completed: {run.completed}\n"
        f"maximum output interval: {np.diff(t).max():.9g} s\n"
        f"breakthrough (added upper-layer inventory > {BREAKTHROUGH_FRACTION:.0%} "
        "of cumulative injected mass): "
        + (f"{t_bt:.6f} s" if t_bt is not None else "not reached")
        + f"\nfinal added upper-layer mass: {added_upper[-1]:.12g} kg\n"
        f"final upper-layer fraction of injected mass: {fraction[-1]:.12g}\n"
        f"maximum upper-layer fraction of injected mass: {fraction.max():.12g}\n"
        "This is an inventory criterion, not cumulative flux across the interface; "
        "an early threshold crossing need not indicate sustained seal failure.\n"
    )
    note.write_text(text, encoding="utf-8")
    print(text)


def snapshots_figure(
    run: rc.RunInfo, times: list[float] | None, suffix: str = ""
) -> None:
    outputs = rc.read_pvd(run.folder)
    available = np.array([t for t, _ in outputs])
    if times:
        indices = sorted({int(np.argmin(np.abs(available - t))) for t in times})
    else:
        indices = sorted({int(i) for i in np.linspace(1, len(outputs) - 1, 4)})
    snaps = [rc.read_snapshot(outputs[i][1], outputs[i][0]) for i in indices]
    s_gas = [rc.gas_saturation_from_z(s.fields["z_CO2"]) for s in snaps]
    vmin = rc.PARAMS["s_gas_init_lower"]
    vmax = max(float(np.max(s)) for s in s_gas)
    ncols = min(len(snaps), 2)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.3 * nrows + 0.5), squeeze=False, layout="constrained")
    for ax, snap, s in zip(axes.ravel(), snaps, s_gas):
        coll = rc.draw_field(ax, snap, s, cmap="viridis", vmin=vmin, vmax=vmax, title=f"$t$ = {snap.time:.0f} s")
        ax.grid(False)
    for ax in axes.ravel()[len(snaps):]:
        ax.axis("off")
    fig.colorbar(coll, ax=axes.ravel().tolist(), shrink=0.8, label="gas saturation $S_n$ [-]")
    fig.suptitle(run.label, fontsize=10)
    rc.save_figure(fig, f"fig_study2_snapshots{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=float, default=50.0)
    parser.add_argument("--upper", type=float, default=500.0)
    parser.add_argument("--run", type=str, default=None, help="Explicit run folder (overrides --lower/--upper).")
    parser.add_argument("--snapshot-times", type=float, nargs="*", default=None)
    parser.add_argument("--suffix", default="", help="Suffix for a supplementary case's output files.")
    args = parser.parse_args()
    rc.set_style()
    rc.check_parameters_against_run_script()
    run = select_run(args)
    ts = rc.time_series(run)
    timeseries_figure(run, ts, args.suffix)
    snapshots_figure(run, args.snapshot_times, args.suffix)


if __name__ == "__main__":
    main()
