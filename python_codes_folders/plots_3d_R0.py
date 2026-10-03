"""
plots_3d_R0.py - three-dimensional views of the basic reproduction number
R_0 (eq:flu_R0, eta_A = 1) of the SVEAITHQR influenza model as three key
epidemiological parameters are varied simultaneously.

Two plot types are produced:
  * layered surfaces: R_0 (vertical axis) over a grid of two parameters, with
    one surface for each of three values of a third parameter, and the
    threshold plane R_0 = 1;
  * 3-D scatter "volume" plots: the three parameters on the axes and R_0 as
    the colour of each point.
All other parameters are held at their baseline (fitted or assumed) values.

Run after fit_national.py:  python src/plots_3d_R0.py
Outputs: outputs/figures/F19_ ... F24_*.png
Libraries: NumPy (Harris et al., 2020), Matplotlib mplot3d (Hunter, 2007).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import R0_closed_form  # noqa: E402
from parameters import FIXED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style
LABEL_FONTSIZE = 23      # axis labels
TICK_LABELSIZE = 20      # tick numbers
TITLE_FONTSIZE = 22
LEGEND_FONTSIZE = 18
LABEL_PAD_XY = 38        # gap between x/y tick numbers and axis labels
LABEL_PAD_Z = 44         # gap between z tick numbers and z label
TICK_PAD = 10            # gap between axis lines and tick numbers
TICK_PAD_Z = 18
TITLE_PAD = 30
FIG_SIZE = (17, 13)
LAYER_CMAPS = ["Blues", "Greens", "Reds"]
LAYER_COLOURS = ["#2471a3", "#1e8449", "#c0392b"]

SYMBOL = {"beta_0": r"$\beta_0$", "eta_I": r"$\eta_I$", "gamma_I": r"$\gamma_I$",
          "gamma_A": r"$\gamma_A$", "phi_I": r"$\phi_I$", "alpha": r"$\alpha$",
          "theta": r"$\theta$", "eps_V": r"$\varepsilon_V$", "omega": r"$\omega$",
          "xi": r"$\xi$"}


def baseline():
    """Assumed values plus the national fitted values (outputs/fit)."""
    p = dict(FIXED)
    f = ROOT / "outputs" / "fit" / "national_fit.json"
    fit = json.loads(f.read_text())
    est = dict(zip(fit["names"], fit["value"]))
    for k in ("beta_0", "p_c", "rho", "delta_I"):
        p[k] = est[k]
    return p


def R0_grid(p, **arrays):
    """R_0 with some parameters replaced by (broadcastable) arrays."""
    q = dict(p)
    q.update(arrays)
    return R0_closed_form(q)


def new_3d_figure():
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.05, top=0.93)
    return fig, ax


def style_3d_axes(ax, xlabel, ylabel, zlabel, title, elev=24, azim=-58):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD_XY)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD_XY)
    ax.set_zlabel(zlabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD_Z)
    ax.tick_params(axis="x", which="major", labelsize=TICK_LABELSIZE, pad=TICK_PAD)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABELSIZE, pad=TICK_PAD)
    ax.tick_params(axis="z", which="major", labelsize=TICK_LABELSIZE, pad=TICK_PAD_Z)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.view_init(elev=elev, azim=azim)


def layered_surface(fname, p, xname, xr, yname, yr, zname, zvals, title, n=40, elev=24, azim=-58):
    """R_0 over (x, y) with one surface per value of the third parameter z."""
    X, Y = np.meshgrid(np.linspace(*xr, n), np.linspace(*yr, n))
    fig, ax = new_3d_figure()
    handles = []
    for zv, cmap, col in zip(zvals, LAYER_CMAPS, LAYER_COLOURS):
        R = R0_grid(p, **{xname: X, yname: Y, zname: zv})
        ax.plot_surface(X, Y, R, cmap=cmap, alpha=0.75, linewidth=0, antialiased=True,
                        vmin=R.min() - 0.5 * np.ptp(R), vmax=R.max())
        handles.append(Patch(color=col, label=f"{SYMBOL[zname]} = {zv:.4g}"))
    # threshold plane R_0 = 1
    ax.plot_surface(X, Y, np.ones_like(X), color="grey", alpha=0.25, linewidth=0)
    handles.append(Patch(color="grey", alpha=0.4, label=r"$\mathcal{R}_0 = 1$"))
    style_3d_axes(ax, SYMBOL[xname], SYMBOL[yname], r"$\mathcal{R}_0$", title, elev, azim)
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc="upper left", frameon=False)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def volume_scatter(fname, p, names, ranges, title, n=14, elev=22, azim=-52):
    """Three parameters on the axes, R_0 as colour (threshold R_0 = 1 marked)."""
    g = [np.linspace(*r, n) for r in ranges]
    A, B, C = np.meshgrid(*g, indexing="ij")
    R = R0_grid(p, **{names[0]: A, names[1]: B, names[2]: C})
    fig, ax = new_3d_figure()
    # diverging colours centred on the threshold: blue R_0 < 1, red R_0 > 1
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(0.0, float(R.min())), vcenter=1.0,
                                          vmax=max(2.0, float(R.max())))
    sc = ax.scatter(A.ravel(), B.ravel(), C.ravel(), c=R.ravel(), cmap="coolwarm",
                    s=28, alpha=0.8, norm=norm)
    # highlight points close to the threshold surface R_0 = 1
    near = np.abs(R - 1) < 0.08 * max(1.0, np.ptp(R) / 10)
    ax.scatter(A[near], B[near], C[near], color="k", s=45, label=r"$\mathcal{R}_0 \approx 1$")
    style_3d_axes(ax, SYMBOL[names[0]], SYMBOL[names[1]], SYMBOL[names[2]], title, elev, azim)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, aspect=14, pad=0.12)
    cbar.set_label(r"$\mathcal{R}_0$", fontsize=LABEL_FONTSIZE, labelpad=18)
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", frameon=False)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def main():
    p = baseline()
    print(f"baseline R0 = {R0_closed_form(p):.4f}")

    # 1. transmission: beta_0 x eta_I, layers of symptomatic recovery gamma_I
    layered_surface("F19_R0_beta0_etaI_gammaI.png", p,
                    "beta_0", (0.1, 1.5), "eta_I", (0.2, 1.5),
                    "gamma_I", [0.15, 0.20, 0.30],
                    r"$\mathcal{R}_0$ vs $\beta_0$ and $\eta_I$ for three values of $\gamma_I$")

    # 2. control: isolation phi_I x antiviral uptake alpha, layers of beta_0
    layered_surface("F20_R0_phiI_alpha_beta0.png", p,
                    "phi_I", (0.0, 0.8), "alpha", (0.0, 0.8),
                    "beta_0", [0.4, 0.8, 1.125],
                    r"$\mathcal{R}_0$ vs $\phi_I$ and $\alpha$ for three values of $\beta_0$")

    # 3. asymptomatic infection: theta x gamma_A, layers of eta_I
    layered_surface("F21_R0_theta_gammaA_etaI.png", p,
                    "theta", (0.05, 0.9), "gamma_A", (0.1, 0.6),
                    "eta_I", [0.5, 1.0, 1.5],
                    r"$\mathcal{R}_0$ vs $\theta$ and $\gamma_A$ for three values of $\eta_I$")

    # 4. vaccination: efficacy eps_V x vaccination rate omega, layers of waning xi
    layered_surface("F22_R0_epsV_omega_xi.png", p,
                    "eps_V", (0.0, 0.95), "omega", (0.0, 0.02),
                    "xi", [1 / 365, 1 / 180, 1 / 90],
                    r"$\mathcal{R}_0$ vs $\varepsilon_V$ and $\omega$ for three values of $\xi$",
                    azim=-130)

    # 5. volume: beta_0, phi_I, alpha coloured by R_0
    volume_scatter("F23_R0_volume_beta0_phiI_alpha.png", p,
                   ["beta_0", "phi_I", "alpha"], [(0.1, 1.5), (0.0, 0.8), (0.0, 0.8)],
                   r"$\mathcal{R}_0$ over $(\beta_0, \phi_I, \alpha)$")

    # 6. volume: beta_0, eta_I, gamma_I coloured by R_0
    volume_scatter("F24_R0_volume_beta0_etaI_gammaI.png", p,
                   ["beta_0", "eta_I", "gamma_I"], [(0.1, 1.5), (0.2, 1.5), (0.1, 0.5)],
                   r"$\mathcal{R}_0$ over $(\beta_0, \eta_I, \gamma_I)$")
    print("figures F19-F24 ->", FIG)


if __name__ == "__main__":
    main()
