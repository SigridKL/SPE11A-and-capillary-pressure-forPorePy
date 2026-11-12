"""Figures and table describing the numerical example (geometry and parameters).

Produces, in ``figurer_results/``:

* ``fig_setup_geometry``   -- sketch of the two-layer domain, interface, well and
                              boundary conditions, with layer properties annotated;
* ``fig_setup_constitutive`` -- Brooks-Corey capillary pressure for the entry
                              pressures used in the study, and the relative
                              permeabilities;
* ``tab_setup_parameters.tex`` -- parameter table for \\input in the thesis.

Usage::

    python plot_results_setup.py [--entry-pressures 50 100 500 2000]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import results_common as rc


def geometry_figure() -> None:
    p = rc.PARAMS
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.linspace(0, p["domain_width"], 400)
    y_if = rc.interface_height(x)
    ax.fill_between(x, 0, y_if, color="#f4d58d", label="lower layer")
    ax.fill_between(x, y_if, p["domain_height"], color="#c9d7e8", label="upper layer")
    ax.plot(x, y_if, "k-", lw=1.2)
    rc.draw_well(ax, edgecolor="red", lw=1.5)
    ax.text(
        p["well_pos_x"] + 0.13,
        p["well_pos_y"],
        "CO$_2$ injection\n$q=10^{-4}$ kg/s",
        va="center",
        fontsize=8.5,
        color="red",
    )
    ax.text(
        0.1,
        0.1,
        f"$\\phi={p['porosity_lower']}$, $\\kappa={p['permeability_lower']:.0e}$ m$^2$\n"
        "$p_{e,\\mathrm{low}}$ (fixed)",
        fontsize=8.5,
    )
    ax.text(
        0.1,
        1.0,
        f"$\\phi={p['porosity_upper']}$, $\\kappa={p['permeability_upper']:.0e}$ m$^2$\n"
        "$p_{e,\\mathrm{up}}$ (varied)",
        fontsize=8.5,
    )
    ax.annotate(
        "",
        xy=(p["domain_width"] / 2, p["y_split"] + p["buehøyde"]),
        xytext=(p["domain_width"] / 2, p["y_split"]),
        arrowprops={"arrowstyle": "<->", "lw": 0.9},
    )
    ax.text(
        p["domain_width"] / 2 + 0.03,
        p["y_split"] + p["buehøyde"] / 2,
        f"{p['buehøyde']} m",
        fontsize=8.5,
        va="center",
    )
    ax.axhline(p["domain_height"], color="#0072B2", lw=3)
    ax.text(
        0.05,
        p["domain_height"] + 0.03,
        "Dirichlet: $p_n=p_{\\mathrm{top}}$, $S_n=0.2$",
        ha="left",
        fontsize=8.5,
        color="#0072B2",
    )
    ax.text(
        p["domain_width"] / 2,
        -0.08,
        "no-flow on the remaining boundaries",
        ha="center",
        fontsize=8.5,
    )
    ax.set_xlim(0, p["domain_width"])
    ax.set_ylim(-0.15, p["domain_height"] + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    rc.save_figure(fig, "fig_setup_geometry")


def constitutive_figure(entry_pressures: list[float]) -> None:
    p = rc.PARAMS
    s_w = np.linspace(p["sw_imm"] + 1e-3, 1.0, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for color, pe in zip(rc.PALETTE, entry_pressures):
        ax1.plot(
            s_w,
            rc.capillary_pressure(s_w, pe),
            color=color,
            label=f"$p_e$ = {rc.entry_pressure_label(pe)}",
        )
    ax1.axhline(p["pcap_max"], color="k", ls=":", lw=1, label="$p_{c,\\max}$")
    s_w0 = 1.0 - p["s_gas_init_lower"]
    ax1.axvline(s_w0, color="0.5", ls="--", lw=0.8)
    ax1.text(s_w0 + 0.01, 2e4, "initial $S_w$", fontsize=8, color="0.4")
    ax1.set_yscale("log")
    ax1.set_xlabel("wetting saturation $S_w$ [-]")
    ax1.set_ylabel("capillary pressure $p_c$ [Pa]")
    ax1.set_title("Brooks–Corey capillary pressure")
    ax1.legend(fontsize=7.5)

    s = np.linspace(0, 1, 400)
    ax2.plot(s, rc.relative_permeability(s, "aqueous"), color=rc.PALETTE[0], label="$k_{r,w}(S_w)$")
    ax2.plot(s, rc.relative_permeability(s, "gaseous"), color=rc.PALETTE[1], label="$k_{r,n}(S_n)$")
    ax2.axvline(p["sw_imm"], color=rc.PALETTE[0], ls=":", lw=0.8)
    ax2.axvline(p["sn_imm"], color=rc.PALETTE[1], ls=":", lw=0.8)
    ax2.set_xlabel("phase saturation [-]")
    ax2.set_ylabel("relative permeability [-]")
    ax2.set_title("Relative permeabilities")
    ax2.legend()
    fig.tight_layout()
    rc.save_figure(fig, "fig_setup_constitutive")


def parameter_table(entry_pressures: list[float]) -> None:
    p = rc.PARAMS
    pe_values = ", ".join(f"{v:g}" for v in entry_pressures)
    rows = [
        ["Domain width $\\times$ height", f"{p['domain_width']} $\\times$ {p['domain_height']}", "m"],
        ["Interface: $y_{\\mathrm{if}}(x)=y_0+h\\sin(\\pi x/W)$, $y_0$, $h$", f"{p['y_split']}, {p['buehøyde']}", "m"],
        ["Cell size (Cartesian grid)", f"{p['cell_size']}", "m"],
        ["Porosity, lower / upper layer", f"{p['porosity_lower']} / {p['porosity_upper']}", "--"],
        ["Permeability $\\kappa$, lower / upper layer", f"{p['permeability_lower']:g} / {p['permeability_upper']:g}", "m$^2$"],
        ["Entry pressure, lower layer $p_{e,\\mathrm{low}}$", "50", "Pa"],
        ["Entry pressure, upper layer $p_{e,\\mathrm{up}}$", pe_values, "Pa"],
        ["Brooks--Corey index $\\lambda$", f"{p['c2']:g}", "--"],
        ["Capillary pressure cap $p_{c,\\max}$", f"{p['pcap_max']:g}", "Pa"],
        ["Residual saturations $S_{w,\\mathrm{imm}}$ / $S_{n,\\mathrm{imm}}$", f"{p['sw_imm']} / {p['sn_imm']}", "--"],
        ["Relative-permeability exponents $c_{w}$ / $c_{n}$", f"{p['c_w1']} / {p['c_n1']}", "--"],
        ["Water density / viscosity", f"{p['density_h2o']} / {p['viscosity_h2o']:g}", "kg/m$^3$ / Pa\\,s"],
        ["CO$_2$ density / viscosity", f"{p['density_co2']} / {p['viscosity_co2']:g}", "kg/m$^3$ / Pa\\,s"],
        ["Initial gas saturation (both layers)", f"{p['s_gas_init_lower']}", "--"],
        ["Top boundary pressure $p_{\\mathrm{top}}$", f"{p['initial_pressure']}", "Pa"],
        ["Top boundary gas saturation", f"{p['s_gas_bc']}", "--"],
        ["Well centre $(x,y)$, size", f"({p['well_pos_x']}, {p['well_pos_y']}), {p['well_size']}", "m"],
        ["Injection rate (pure CO$_2$)", f"{p['injection_rate_kg']:g}", "kg/s"],
        ["Final time, initial / minimum $\\Delta t$", f"{p['t_final']:g}, {p['dt_init']} / {p['dt_min']}", "s"],
        ["Newton: max. iterations", f"{p['nl_max_iterations']}", "--"],
        ["Newton: increment tol. (abs / rel)", f"{p['nl_convergence_inc_atol']:g} / {p['nl_convergence_inc_rtol']:g}", "--"],
        ["Newton: residual tol. (abs)", f"{p['nl_convergence_res_atol']:g}", "--"],
    ]
    rc.write_latex_table(
        "tab_setup_parameters",
        ["Parameter", "Value", "Unit"],
        rows,
        "Parameters of the two-layer capillary-barrier example. Values mirror "
        "\\texttt{purewaterstart.py}.",
        "tab:setup_parameters",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry-pressures",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 500.0, 1000.0, 2000.0],
        help="Entry pressures to show in the capillary-pressure plot and table.",
    )
    args = parser.parse_args()
    rc.set_style()
    rc.check_parameters_against_run_script()
    geometry_figure()
    constitutive_figure(args.entry_pressures)
    parameter_table([v for v in args.entry_pressures if v != 50.0] or args.entry_pressures)


if __name__ == "__main__":
    main()
