"""
sensitivity.py - global uncertainty and sensitivity analysis by Latin
Hypercube Sampling (LHS; McKay et al., 1979) and Partial Rank Correlation
Coefficients (PRCC; Blower & Dowlatabadi, 1994; Marino et al., 2008).

Responses
  (1) R_0, the basic reproduction number eq:flu_R0 (eta_A = 1)
  (2) cumulative reported epidemic cases of the reference season 2023/24,
      simulated with the fitted seasonal forcing and initial conditions
  (3) peak weekly reported epidemic cases of the reference season

Each parameter is sampled uniformly within +/-25% of its baseline (fitted
value or fixed assumption). N = 1000 LHS draws. PRCC significance is tested
with t = r sqrt((N - 2 - p)/(1 - r^2)), df = N - 2 - p.

Run after fit_national.py:  python src/sensitivity.py   (about 1 minute)
Libraries: SciPy qmc.LatinHypercube and scipy.stats (Virtanen et al., 2020).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_national as fn  # noqa: E402
from model import R0_closed_form, initial_state, simulate_weekly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sensitivity"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 1000
SPREAD = 0.25
SEED = 2026

R0_PARAMS = ["beta_0", "sigma", "theta", "gamma_A", "gamma_I", "gamma_H", "gamma_T", "gamma_Q",
             "alpha_A", "alpha", "alpha_H", "phi_A", "phi_I", "phi_H", "rho", "delta_I", "delta_H",
             "eta_I", "eta_H", "eta_T", "eta_Q", "eps_V", "omega", "xi", "mu"]
SIM_EXTRA = ["kappa", "p_c"]

LATEX = {"beta_0": r"\beta_0", "sigma": r"\sigma", "theta": r"\theta", "gamma_A": r"\gamma_A",
         "gamma_I": r"\gamma_I", "gamma_H": r"\gamma_H", "gamma_T": r"\gamma_T", "gamma_Q": r"\gamma_Q",
         "alpha_A": r"\alpha_A", "alpha": r"\alpha", "alpha_H": r"\alpha_H", "phi_A": r"\phi_A",
         "phi_I": r"\phi_I", "phi_H": r"\phi_H", "rho": r"\rho", "delta_I": r"\delta_I",
         "delta_H": r"\delta_H", "eta_I": r"\eta_I", "eta_H": r"\eta_H", "eta_T": r"\eta_T",
         "eta_Q": r"\eta_Q", "eps_V": r"\varepsilon_V", "omega": r"\omega", "xi": r"\xi", "mu": r"\mu",
         "kappa": r"\kappa", "p_c": r"p_c"}


def prcc(X, y):
    """PRCC of each column of X with y, and two-sided p-values."""
    n, k = X.shape
    R = np.column_stack([stats.rankdata(X[:, j]) for j in range(k)])
    ry = stats.rankdata(y)
    out = []
    for j in range(k):
        Z = np.column_stack([np.ones(n), np.delete(R, j, axis=1)])
        ex = R[:, j] - Z @ np.linalg.lstsq(Z, R[:, j], rcond=None)[0]
        ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        r = np.corrcoef(ex, ey)[0, 1]
        df = n - 2 - (k - 1)
        t = r * np.sqrt(df / (1 - r ** 2))
        out.append((r, 2 * stats.t.sf(abs(t), df)))
    return np.array(out)


def main():
    res = json.loads((fn.OUT / "national_fit.json").read_text())
    v = np.array(res["value"])
    _, p_fit = fn.predict(v)
    ref = fn.SEASONS[0]
    shared, season = fn.unpack(v)
    base = {**p_fit}
    names = R0_PARAMS + SIM_EXTRA
    b = np.array([base[k] for k in names])
    lo, hi = b * (1 - SPREAD), b * (1 + SPREAD)
    hi[names.index("theta")] = min(hi[names.index("theta")], 0.99)
    hi[names.index("eps_V")] = min(hi[names.index("eps_V")], 0.99)
    hi[names.index("p_c")] = min(hi[names.index("p_c")], 1.0)
    U = qmc.LatinHypercube(d=len(names), seed=SEED).random(N_SAMPLES)
    X = qmc.scale(U, lo, hi)

    # fitted forcing and initial state of the reference season
    pred_ref, _ = fn.predict(v)
    forcing = pred_ref[ref]["p"]["forcing_fn"]
    R0s, cum, peak = [], [], []
    for row in X:
        p = {**base, **dict(zip(names, row))}
        p["Lambda"] = p["mu"] * fn.N0
        p["forcing_fn"] = forcing
        R0s.append(R0_closed_form(p))
        sim = simulate_weekly(p, initial_state(p, fn.N0, season[ref]["s0"], shared["E0"]))
        epi = p["p_c"] * sim["inc_I"]
        cum.append(epi[1:].sum())
        peak.append(epi.max())
    R0s, cum, peak = map(np.array, (R0s, cum, peak))

    tables = {}
    for resp, y, cols in [("R0", R0s, R0_PARAMS), ("cumulative_cases", cum, names), ("peak_cases", peak, names)]:
        idx = [names.index(c) for c in cols]
        pr = prcc(X[:, idx], y)
        df = pd.DataFrame({"parameter": cols, "PRCC": pr[:, 0], "p_value": pr[:, 1]})
        df = df.reindex(df["PRCC"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.to_csv(OUT / f"prcc_{resp}.csv", index=False)
        tables[resp] = df
        print(f"\n=== PRCC: {resp} ===\n", df.round(4).to_string(index=False))

    summ = pd.DataFrame({
        "response": ["R0", "cumulative reported epidemic cases 2023/24", "peak weekly reported cases 2023/24"],
        "baseline": [R0_closed_form(base), None, None],
        "median": [np.median(R0s), np.median(cum), np.median(peak)],
        "q2.5": [np.percentile(R0s, 2.5), np.percentile(cum, 2.5), np.percentile(peak, 2.5)],
        "q97.5": [np.percentile(R0s, 97.5), np.percentile(cum, 97.5), np.percentile(peak, 97.5)],
        "P(R0>1)": [np.mean(R0s > 1), None, None]})
    summ.to_csv(OUT / "uncertainty_summary.csv", index=False)
    print(summ.to_string(index=False))

    # ---- figures
    def bar(df, fname, title, top=None):
        d = df if top is None else df.head(top)
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(d) + 1.2))
        cols = ["#c0392b" if x > 0 else "#2471a3" for x in d["PRCC"]]
        ax.barh(range(len(d)), d["PRCC"], color=cols)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"${LATEX[k]}$" for k in d["parameter"]])
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        for i, (r, pv) in enumerate(zip(d["PRCC"], d["p_value"])):
            ax.text(r + (0.02 if r >= 0 else -0.02), i, f"{r:+.3f}" + ("*" if pv < 0.05 else ""),
                    va="center", ha="left" if r >= 0 else "right", fontsize=8)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("PRCC")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=200)
        plt.close(fig)

    bar(tables["R0"], "F14_prcc_R0_all.png", r"PRCC of all parameters with $\mathcal{R}_0$ (* p < 0.05)")
    bar(tables["R0"], "F15_prcc_R0_top10.png", r"Top 10 parameters by |PRCC| with $\mathcal{R}_0$", top=10)
    bar(tables["cumulative_cases"], "F16_prcc_cumulative_cases.png",
        "PRCC with cumulative reported cases, 2023/24 (* p < 0.05)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    top = tables["R0"]["parameter"].head(3).tolist()
    for ax, k in zip(axes, top):
        ax.scatter(X[:, names.index(k)], R0s, s=5, alpha=0.5)
        ax.set_xlabel(f"${LATEX[k]}$")
        ax.set_ylabel(r"$\mathcal{R}_0$")
    fig.suptitle(r"$\mathcal{R}_0$ against its three most influential parameters (1000 LHS samples)")
    fig.tight_layout()
    fig.savefig(FIG / "F17_R0_scatter_top3.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(R0s, bins=40, color="#7f8c8d")
    ax.axvline(R0_closed_form(base), color="r", label="fitted baseline")
    ax.axvline(1, color="k", ls="--", label=r"$\mathcal{R}_0=1$")
    ax.set_xlabel(r"$\mathcal{R}_0$")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    ax.set_title(r"Uncertainty distribution of $\mathcal{R}_0$ (LHS, $\pm$25%)")
    fig.tight_layout()
    fig.savefig(FIG / "F18_R0_uncertainty.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
