"""
plots_2d.py - two-dimensional parameter-variation plots for the SVEAITHQR
influenza model: the time course of a compartment (or of weekly reported
cases) when one key parameter is varied and all others stay at their
baseline (fitted or assumed) values.

Every run simulates the reference season 2023/24 with the fitted seasonal
forcing beta(t) = beta_0 f(t) and the fitted initial conditions, so the
curves show the model's actual seasonal (two-wave) dynamics.

Parameters that describe transmission or natural history are scaled by
0.2, 0.4, 0.6, 0.8 and 1.0 times baseline. Control and vaccination rates
are low at baseline, so they are scaled upwards (1x to 8x, or given values).

Run after fit_national.py:  python src/plots_2d.py
Outputs: outputs/figures/F25_ ... F36_*.png
Libraries: NumPy, SciPy solve_ivp (Virtanen et al., 2020), Matplotlib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_national as fn  # noqa: E402
from model import initial_state, rhs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style
LABEL_FONTSIZE = 23
TICK_LABELSIZE = 20
TITLE_FONTSIZE = 23
LEGEND_FONTSIZE = 15
LINE_WIDTH = 4
LABEL_PAD = 14            # gap between tick numbers and axis labels
FIG_SIZE = (12, 8)
COLORS = ["red", "blue", "green", "orange", "magenta"]

DOWN = [0.2, 0.4, 0.6, 0.8, 1.0]      # multipliers of the baseline value
UP = [1.0, 2.0, 4.0, 6.0, 8.0]

SYMBOL = {"beta_0": r"$\beta_0$", "theta": r"$\theta$", "eta_I": r"$\eta_I$",
          "gamma_I": r"$\gamma_I$", "rho": r"$\rho$", "alpha": r"$\alpha$",
          "phi_I": r"$\phi_I$", "kappa": r"$\kappa$", "omega": r"$\omega$",
          "sigma": r"$\sigma$", "delta_I": r"$\delta_I$", "eps_V": r"$\varepsilon_V$"}
IDX = {"S": 0, "V": 1, "E": 2, "A": 3, "I": 4, "H": 5, "T": 6, "Q": 7, "R": 8}
T_END = 364.0
T_EVAL = np.linspace(0.0, T_END, 729)            # every half day


def baseline():
    """Reference-season parameters (with fitted forcing) and initial state inputs."""
    res = json.loads((fn.OUT / "national_fit.json").read_text())
    v = np.array(res["value"])
    pred, _ = fn.predict(v)
    shared, season = fn.unpack(v)
    ref = fn.SEASONS[0]
    return pred[ref]["p"], season[ref]["s0"], shared["E0"]


def solve(p, s0, E0):
    y0 = initial_state(p, fn.N0, s0, E0)
    sol = solve_ivp(rhs, (0.0, T_END), y0, t_eval=T_EVAL, args=(p,), method="LSODA",
                    rtol=1e-7, atol=1e-6)
    return sol.y


def weekly_reported(Y, p):
    """Weekly reported epidemic cases p_c * increment of C_I."""
    CI = Y[9, ::14]                               # values every 7 days
    return p["p_c"] * np.diff(CI)


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE, pad=8)
    ax.yaxis.get_offset_text().set_fontsize(TICK_LABELSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=16)
    ax.grid(alpha=0.3)


def plot_variation(fname, base, s0, E0, pname, values, target, ylabel):
    """target: compartment letter, or 'cases' for weekly reported cases."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for val, col in zip(values, COLORS):
        p = {**base, pname: val}
        if pname == "mu":
            p["Lambda"] = p["mu"] * fn.N0
        Y = solve(p, s0, E0)
        lab = f"{SYMBOL[pname]} = {val:.4g}"
        if target == "cases":
            ax.plot(np.arange(1, 53), weekly_reported(Y, p), color=col, lw=LINE_WIDTH, label=lab)
        else:
            ax.plot(T_EVAL, Y[IDX[target]], color=col, lw=LINE_WIDTH, label=lab)
    xlabel = "Epidemiological week of season" if target == "cases" else "Time (days from 1 July)"
    style(ax, xlabel, ylabel, f"Effect of {SYMBOL[pname]} on {ylabel.split(' (')[0]}")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def main():
    base, s0, E0 = baseline()
    b = lambda k, mult: [base[k] * m for m in mult]  # noqa: E731
    specs = [
        ("F25_S_vs_beta0.png", "beta_0", b("beta_0", DOWN), "S", r"Susceptible ($S$)"),
        ("F26_E_vs_beta0.png", "beta_0", b("beta_0", DOWN), "E", r"Exposed ($E$)"),
        ("F27_A_vs_theta.png", "theta", [0.1, 0.2, 0.3, 0.45, 0.6], "A", r"Asymptomatic ($A$)"),
        ("F28_I_vs_etaI.png", "eta_I", b("eta_I", DOWN), "I", r"Symptomatic infectious ($I$)"),
        ("F29_I_vs_gammaI.png", "gamma_I", b("gamma_I", [1.0, 1.25, 1.5, 1.75, 2.0]), "I",
         r"Symptomatic infectious ($I$)"),
        ("F30_H_vs_rho.png", "rho", b("rho", DOWN), "H", r"Hospitalised ($H$)"),
        ("F31_T_vs_alpha.png", "alpha", b("alpha", UP), "T", r"On antiviral therapy ($T$)"),
        ("F32_Q_vs_phiI.png", "phi_I", b("phi_I", UP), "Q", r"Quarantined/isolated ($Q$)"),
        ("F33_R_vs_kappa.png", "kappa", [1 / (8 * 365), 1 / (4 * 365), 1 / (2 * 365), 1 / 365, 1 / 180],
         "R", r"Recovered ($R$)"),
        ("F34_V_vs_omega.png", "omega", [1e-5, 5e-4, 1e-3, 2e-3, 5e-3], "V", r"Vaccinated ($V$)"),
        ("F35_cases_vs_phiI.png", "phi_I", b("phi_I", UP), "cases", r"Weekly reported cases"),
        ("F36_cases_vs_beta0.png", "beta_0", b("beta_0", [0.6, 0.7, 0.8, 0.9, 1.0]), "cases",
         r"Weekly reported cases"),
    ]
    for i, (fname, pname, vals, target, ylabel) in enumerate(specs, 1):
        print(f"Plot {i}: {ylabel} vs {pname} ...", end=" ", flush=True)
        try:
            plot_variation(fname, base, s0, E0, pname, vals, target, ylabel)
            print("done")
        except Exception as e:  # keep going if one plot fails
            print(f"error: {e}")
    print("figures F25-F36 ->", FIG)


if __name__ == "__main__":
    main()
