"""
report_fit.py - builds the parameter table (CSV + LaTeX), derived threshold
quantities and fit figures from outputs/fit/national_fit.json.

Run after fit_national.py:  python src/report_fit.py
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
import fit_national as fn  # noqa: E402
from model import (R0_closed_form, R0_ngm, R0bar, beta_t, endemic_quadratic,  # noqa: E402
                   rates)
from parameters import FIXED, SEASONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "fit"
FIG = ROOT / "outputs" / "figures"
SEASON_COL = {"2023/2024": "#1f77b4", "2024/2025": "#d62728", "2025/2026": "#2ca02c"}

# symbol, description, unit
META = {
    "Lambda": (r"\Lambda", "Recruitment rate ($\\mu N_0$)", "persons day$^{-1}$"),
    "mu": (r"\mu", "Natural death rate (life expectancy 55 yr)", "day$^{-1}$"),
    "sigma": (r"\sigma", "Progression rate from E (latent period 2 d)", "day$^{-1}$"),
    "theta": (r"\theta", "Asymptomatic fraction", "--"),
    "gamma_A": (r"\gamma_A", "Recovery rate, asymptomatic", "day$^{-1}$"),
    "gamma_I": (r"\gamma_I", "Recovery rate, symptomatic (community)", "day$^{-1}$"),
    "gamma_H": (r"\gamma_H", "Recovery rate, hospitalised", "day$^{-1}$"),
    "gamma_T": (r"\gamma_T", "Recovery rate, on antiviral therapy", "day$^{-1}$"),
    "gamma_Q": (r"\gamma_Q", "Recovery rate, isolated", "day$^{-1}$"),
    "alpha_A": (r"\alpha_A", "Antiviral uptake rate, asymptomatic", "day$^{-1}$"),
    "alpha": (r"\alpha", "Antiviral uptake rate, symptomatic", "day$^{-1}$"),
    "alpha_H": (r"\alpha_H", "Antiviral uptake rate, hospitalised", "day$^{-1}$"),
    "phi_A": (r"\phi_A", "Isolation rate, asymptomatic", "day$^{-1}$"),
    "phi_I": (r"\phi_I", "Isolation rate, symptomatic", "day$^{-1}$"),
    "phi_H": (r"\phi_H", "Isolation rate, hospitalised", "day$^{-1}$"),
    "eta_I": (r"\eta_I", "Relative infectiousness of $I$ (reference $A$ = 1)", "--"),
    "eta_H": (r"\eta_H", "Relative infectiousness of $H$", "--"),
    "eta_T": (r"\eta_T", "Relative infectiousness of $T$", "--"),
    "eta_Q": (r"\eta_Q", "Relative infectiousness of $Q$", "--"),
    "eps_V": (r"\varepsilon_V", "Vaccine efficacy", "--"),
    "omega": (r"\omega", "Vaccination rate", "day$^{-1}$"),
    "xi": (r"\xi", "Waning rate of vaccine protection", "day$^{-1}$"),
    "kappa": (r"\kappa", "Waning rate of infection-acquired immunity", "day$^{-1}$"),
    "beta_0": (r"\beta_0", "Mean transmission rate", "day$^{-1}$"),
    "m": (r"m", "Season transmission multiplier (2023/24 = 1)", "--"),
    "h": (r"h", "Season Harmattan shift of spline knots 5--7 (2023/24 = 0)", "--"),
    **{f"g_{k}": (rf"g_{{{k}}}", f"Log-forcing spline knot {k} (day {365*k/12:.0f} after 1 July; $g_0=0$)", "--")
       for k in range(1, 12)},
    "delta_c": (r"\delta_c", "Amplitude of annual forcing", "--"),
    "phi": (r"\varphi", "Phase of annual forcing (days after 1 July)", "day"),
    "delta_2": (r"\delta_2", "Amplitude of semi-annual (Harmattan) forcing", "--"),
    "phi_2": (r"\varphi_2", "Phase of semi-annual forcing", "day"),
    "p_c": (r"p_c", "Reporting fraction of symptomatic infections", "--"),
    "rho": (r"\rho", "Hospitalisation rate of $I$", "day$^{-1}$"),
    "delta_I": (r"\delta_I", "Disease-induced death rate, community", "day$^{-1}$"),
    "delta_H": (r"\delta_H", "Disease-induced death rate, hospitalised", "day$^{-1}$"),
    "s0": (r"s_0", "Susceptible fraction at season start", "--"),
    "E0": (r"E_0", "Exposed seed at season start", "persons"),
    "B_c": (r"B_c", "Background weekly reported cases", "cases week$^{-1}$"),
    "B_h": (r"B_h", "Background weekly admissions", "adm. week$^{-1}$"),
    "B_d": (r"B_d", "Background weekly deaths", "deaths week$^{-1}$"),
}


def fmt(x):
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e4 or a < 1e-3:
        m, e = f"{x:.3e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.4g}"


def main():
    res = json.loads((OUT / "national_fit.json").read_text())
    v = np.array(res["value"])
    pred, p = fn.predict(v)

    # ------------------------------------------------------ parameter table
    rows = []
    for k in FIXED:
        sym, desc, unit = META[k]
        rows.append(dict(key=k, symbol=sym, description=desc, value=FIXED[k], lo95=np.nan,
                         hi95=np.nan, unit=unit, source="Assumed"))
    for i, name in enumerate(res["names"]):
        base, _, season = name.partition("|")
        sym, desc, unit = META[base]
        rows.append(dict(key=name, symbol=sym + (f"^{{({season[2:4]}/{season[7:9]})}}" if season else ""),
                         description=desc + (f", {season}" if season else ""), value=v[i],
                         lo95=res["lo95"][i], hi95=res["hi95"][i], unit=unit,
                         source="Estimated" + (" (at bound)" if res["at_bound"][i] else "")))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "parameter_table.csv", index=False)

    # ------------------------------------------------------ derived quantities
    r0 = R0_closed_form(p)            # reference season (2023/24, m = 1); beta_0 is the mean of beta(t)
    t = np.linspace(0, 365, 3651)
    pref = pred[SEASONS[0]]["p"]
    fac = beta_t(t, pref) / pref["beta_0"]
    q = endemic_quadratic(p)
    r = rates(p)
    derived = {
        "R0 (closed form, eta_A = 1)": r0,
        "R0 (spectral radius of F V^-1)": R0_ngm(p),
        "R0bar = R0 m0/m1 (GAS threshold)": R0bar(p),

        "endemic quadratic a2": q["a2"], "endemic quadratic a1": q["a1"],
        "endemic quadratic a0": q["a0"], "discriminant a1^2-4a2a0": q["disc"],
        "backward bifurcation possible (a1<0 and disc>0 for R0<1)": bool(q["a1"] < 0),
        "hospitalised fraction of symptomatic, rho/K_I": p["rho"] / r["K_I"],
        "infection fatality (sympt.), (delta_I + rho delta_H/K_H)/K_I":
            (p["delta_I"] + p["rho"] * p["delta_H"] / r["K_H"]) / r["K_I"],
    }
    for s in SEASONS:
        ps = pred[s]["p"]
        derived[f"season R0 = m_s R0 {s}"] = R0_closed_form(ps)
        fs = beta_t(t, ps) / ps["beta_0"]
        derived[f"max_t R0(t) {s}"] = R0_closed_form(ps) * fs.max()
        derived[f"min_t R0(t) {s}"] = R0_closed_form(ps) * fs.min()
        derived[f"day of max beta(t) after season start {s}"] = float(t[np.argmax(fs)])
        Y = pred[s]["Y"]
        infections = Y[9, -1] / (1 - p["theta"])       # all new infections
        derived[f"true infection attack rate {s} (%)"] = 100 * infections / fn.N0
        derived[f"reported attack rate {s} (%) model"] = 100 * pred[s]["cases"][1:].sum() / fn.N0
        derived[f"R_eff at season start {s}"] = R0_closed_form(ps) * fs[0] * (Y[0, 0] + (1 - p["eps_V"]) * Y[1, 0]) / Y[:9, 0].sum() / (r["m1"] / r["m0"])
    der = pd.DataFrame({"quantity": list(derived), "value": list(derived.values())})
    der.to_csv(OUT / "derived_quantities.csv", index=False)
    print(der.to_string(index=False))

    # ------------------------------------------------------ goodness of fit
    gof = []
    for s in SEASONS:
        for c in fn.SERIES:
            y, yh = fn.OBS[s][c][1:], pred[s][c][1:]
            gof.append(dict(season=s, series=c, MAE=np.mean(np.abs(y - yh)),
                            RMSE=np.sqrt(np.mean((y - yh) ** 2)),
                            R2=1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
                            obs_peak_week=int(np.argmax(fn.OBS[s][c]) + 1),
                            fit_peak_week=int(np.argmax(pred[s][c]) + 1),
                            obs_total=y.sum(), fit_total=yh.sum()))
    gof = pd.DataFrame(gof)
    gof.to_csv(OUT / "goodness_of_fit.csv", index=False)
    print(gof.round(3).to_string(index=False))

    # identifiability: strongly correlated estimated pairs
    C = np.array(res["corr"])
    pairs = [(res["names"][i], res["names"][j], C[i, j]) for i in range(len(C)) for j in range(i + 1, len(C))
             if abs(C[i, j]) > 0.9]
    pd.DataFrame(pairs, columns=["param_1", "param_2", "corr"]).to_csv(OUT / "high_correlations.csv", index=False)
    print("highly correlated pairs (|r|>0.9):", pairs)

    # ------------------------------------------------------ LaTeX table
    lines = [r"\begin{longtable}{llll l}",
             r"\caption{Parameters of the SVEAITHQR model. Estimated values are maximum-likelihood",
             r"estimates from the national data (2023/24--2025/26) with asymptotic 95\% intervals;",
             r"assumed values are modelling assumptions varied in the sensitivity analysis.}",
             r"\label{tab:parameters}\\",
             r"\hline Symbol & Description & Value (95\% CI) & Unit & Source\\ \hline\endfirsthead",
             r"\hline Symbol & Description & Value (95\% CI) & Unit & Source\\ \hline\endhead"]
    for _, rw in tab.iterrows():
        val = fmt(rw["value"])
        if np.isfinite(rw["lo95"]):
            val += f" ({fmt(rw['lo95'])}, {fmt(rw['hi95'])})"
        lines.append(f"${rw['symbol']}$ & {rw['description']} & {val} & {rw['unit']} & {rw['source']}\\\\")
    lines += [r"\hline", r"\end{longtable}"]
    (OUT / "parameter_table.tex").write_text("\n".join(lines))

    # ------------------------------------------------------ figures
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for j, c in enumerate(fn.SERIES):
        for i, s in enumerate(SEASONS):
            ax = axes[j, i]
            wk = np.arange(1, 53)
            ax.plot(wk[1:], fn.OBS[s][c][1:], "o", ms=3, color="k", label="observed")
            ax.plot(wk[0], fn.OBS[s][c][0], "x", ms=5, color="grey", label="week 1 (excluded)")
            ax.plot(wk, pred[s][c], "-", color=SEASON_COL[s], lw=1.8, label="model")
            ax.set_yscale("log")
            if j == 0:
                ax.set_title(s)
            if i == 0:
                ax.set_ylabel(["Weekly cases", "Weekly hospitalisations", "Weekly deaths"][j])
            if j == 2:
                ax.set_xlabel("Epi week of season")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("SVEAITHQR model fitted jointly to national cases, hospitalisations and deaths")
    fig.tight_layout()
    fig.savefig(FIG / "F11_national_fit.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for s in SEASONS:
        ps = pred[s]["p"]
        axes[0].plot(t / 7 + 1, R0_closed_form(ps) * beta_t(t, ps) / ps["beta_0"], color=SEASON_COL[s], label=s)
    axes[0].axhline(1, ls="--", color="grey")
    axes[0].set_xlabel("Epi week of season")
    axes[0].set_ylabel(r"$\mathcal{R}_0(t)$")
    axes[0].set_title("Seasonal basic reproduction number")
    for s in SEASONS:
        Y = pred[s]["Y"]
        Nt = Y[:9].sum(0)
        ps = pred[s]["p"]
        reff = (R0_closed_form(ps) * beta_t(7 * np.arange(53), ps) / ps["beta_0"] * (Y[0] + (1 - p["eps_V"]) * Y[1]) / Nt
                / (r["m1"] / r["m0"]))
        axes[1].plot(np.arange(1, 54), reff, color=SEASON_COL[s], label=s)
    axes[1].axhline(1, ls="--", color="grey")
    axes[1].set_xlabel("Epi week of season")
    axes[1].set_ylabel(r"$\mathcal{R}_{\rm eff}(t)$")
    axes[1].set_title("Effective reproduction number (fitted)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "F12_reproduction_numbers.png", dpi=200)
    plt.close(fig)
    print("tables ->", OUT, "; figures F11, F12 ->", FIG)


if __name__ == "__main__":
    main()
