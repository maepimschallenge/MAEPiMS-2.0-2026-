"""
fit_state_severity.py - state-specific hospitalisation and death rates as
log-linear functions of the healthcare accessibility index (proposal P3).

Over a complete epidemic wave, I(0) ~ I(end) ~ 0, so integrating dI/dt gives
    int I dt = (1 - theta) sigma int E dt / K_I ,
and the model implies, for state i with reporting fraction p_c (shared),
    admissions / reported cases = rho_i / (p_c K_I,i)
    deaths / reported cases     = (delta_I,i + rho_i delta_H / K_H) / (p_c K_I,i)
with K_I,i = K_I0 + rho_i + delta_I,i. These two identities are solved exactly
for (rho_i, delta_I,i) for every state and season using background-corrected
(above-baseline) totals, then
    log rho_i     = a_rho + b_rho (h_i - hbar)
    log delta_I,i = a_d   + b_d   (h_i - hbar)
are fitted by ordinary least squares (pooled over seasons).
Run after fit_national.py:  python src/fit_state_severity.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_all  # noqa: E402
from parameters import FIXED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "fit"
FIG = ROOT / "outputs" / "figures"
BASE_WEEKS = (36, 52)


def solve_rates(h, d, p_c, p):
    """Solve h = rho/(p_c K_I), d = (dI + rho dH/K_H)/(p_c K_I) for rho, dI."""
    K0 = p["gamma_I"] + p["alpha"] + p["phi_I"] + p["mu"]
    K_H = p["gamma_H"] + p["alpha_H"] + p["delta_H"] + p["mu"]
    c = p["delta_H"] / K_H
    # rho = h p_c K_I ; dI = d p_c K_I - rho c = p_c K_I (d - h c)
    # K_I = K0 + rho + dI = K0 + p_c K_I (h + d - h c)  ->  K_I = K0 / (1 - p_c (h + d - h c))
    K_I = K0 / (1 - p_c * (h + d - h * c))
    return h * p_c * K_I, p_c * K_I * (d - h * c)


def main():
    fit = json.loads((OUT / "national_fit.json").read_text())
    est = dict(zip(fit["names"], fit["value"]))
    p = {**FIXED, **{k: est[k] for k in ("p_c", "rho", "delta_I")}}
    st = load_all()["state"]
    rows = []
    for (sea, state), g in st.groupby(["season", "state"]):
        g = g.sort_values("epi_week_of_season")
        g = g[g["epi_week_of_season"] >= 2]
        base = g[g["epi_week_of_season"].between(*BASE_WEEKS)][["cases", "hospitalizations", "deaths"]].median()
        ex = (g[["cases", "hospitalizations", "deaths"]] - base).clip(lower=0).sum()
        h, d = ex["hospitalizations"] / ex["cases"], ex["deaths"] / ex["cases"]
        rho, dI = solve_rates(h, d, p["p_c"], p)
        rows.append(dict(season=sea, state=state, hc_access=g["hc_access"].iloc[0] if "hc_access" in g else np.nan,
                         climate=g["climate"].iloc[0], hosp_per_case=h, deaths_per_case=d, rho=rho, delta_I=dI))
    df = pd.DataFrame(rows)
    meta = load_all()["meta"][["state", "hc_access"]]
    df = df.drop(columns="hc_access").merge(meta, on="state")
    hbar = df["hc_access"].mean()
    x = df["hc_access"] - hbar
    X = np.column_stack([np.ones(len(df)), x])
    coef_rows = []
    for col in ["rho", "delta_I"]:
        ok = df[col] > 0
        y = np.log(df.loc[ok, col])
        Xo = X[ok.to_numpy()]
        b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        resid = y - Xo @ b
        s2 = resid @ resid / (len(y) - 2)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xo.T @ Xo)))
        r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
        for j, nm in enumerate(["intercept (at mean h)", "slope b"]):
            coef_rows.append(dict(parameter=col, term=nm, estimate=b[j], se=se[j],
                                  lo95=b[j] - 1.96 * se[j], hi95=b[j] + 1.96 * se[j], R2=r2,
                                  n=int(ok.sum()), hbar=hbar))
        df[f"{col}_fitted"] = np.exp(b[0] + b[1] * x)
    coef = pd.DataFrame(coef_rows)
    coef.to_csv(OUT / "state_severity_regression.csv", index=False)
    state_tab = (df.groupby(["state", "climate", "hc_access"])[["rho", "delta_I", "rho_fitted", "delta_I_fitted"]]
                   .mean().reset_index().sort_values("hc_access"))
    state_tab.to_csv(OUT / "state_severity_rates.csv", index=False)
    print(coef.round(5).to_string(index=False))
    # consistency: national rho implied by the identity vs estimated rho
    print("national rho (fit) =", p["rho"], "; mean state rho (identity) =", df["rho"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, lab in zip(axes, ["rho", "delta_I"], [r"$\rho_i$ (day$^{-1}$)", r"$\delta_{I,i}$ (day$^{-1}$)"]):
        ax.scatter(df["hc_access"], df[col], s=12, alpha=0.7)
        hs = np.linspace(df["hc_access"].min(), df["hc_access"].max(), 50)
        c = coef[coef["parameter"] == col]["estimate"].to_numpy()
        ax.plot(hs, np.exp(c[0] + c[1] * (hs - hbar)), "r-")
        ax.set_yscale("log")
        ax.set_xlabel("Healthcare accessibility index")
        ax.set_ylabel(lab)
    fig.suptitle("State hospitalisation and community death rates vs healthcare access (3 seasons)")
    fig.tight_layout()
    fig.savefig(FIG / "F13_state_severity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
