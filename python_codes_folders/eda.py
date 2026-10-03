"""
eda.py - exploratory data analysis for the MAEPiMS Challenge 2026.

Run from the repository root:
    python src/eda.py
Outputs: outputs/figures/*.png and outputs/tables/*.csv

Libraries: NumPy (Harris et al., 2020), pandas (McKinney, 2010),
Matplotlib (Hunter, 2007).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_all, regional_series, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SEASON_COL = {"2023/2024": "#1f77b4", "2024/2025": "#d62728", "2025/2026": "#2ca02c"}
CLIM_COL = {"sahel": "#b8860b", "savanna": "#6b8e23", "tropical": "#1e6091"}
REG_COL = {"North": "#c0392b", "South": "#2471a3"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# Wave windows (epi_week_of_season). Chosen from the national curves; the
# trough between waves is located inside [TROUGH_LO, TROUGH_HI] per series.
W1_SEARCH = (2, 22)      # main (wet-season / early-dry) wave peak search
W2_SEARCH = (22, 35)     # Harmattan wave peak search
BASELINE = (36, 52)      # inter-epidemic plateau used as background level


def wave_decomposition(y: np.ndarray) -> dict:
    """y indexed by epi_week_of_season 1..52 (y[0] = week 1).
    Returns peak weeks, trough, baseline and above-baseline burden per wave."""
    wk = np.arange(1, 53)
    base = float(np.median(y[(wk >= BASELINE[0]) & (wk <= BASELINE[1])]))
    m1 = (wk >= W1_SEARCH[0]) & (wk <= W1_SEARCH[1])
    m2 = (wk >= W2_SEARCH[0]) & (wk <= W2_SEARCH[1])
    pk1 = int(wk[m1][np.argmax(y[m1])])
    pk2 = int(wk[m2][np.argmax(y[m2])])
    between = (wk > pk1) & (wk < pk2)
    tr = int(wk[between][np.argmin(y[between])])
    ex = np.clip(y - base, 0, None)
    ex[0] = 0.0  # week-1 artefact excluded
    b1, b2 = ex[(wk >= 2) & (wk <= tr)].sum(), ex[(wk > tr) & (wk <= 35)].sum()
    # epidemic growth rate (per week): steepest 3-week log-slope of excess on
    # the rising limb of each wave
    # Only weeks whose excess is >= 5% of that wave's peak excess are used, so
    # that noise around the baseline cannot produce spurious slopes.
    def max_growth(lo, hi):
        seg = ex[lo - 1:hi]
        thr = 0.05 * seg.max()
        slopes = [(np.log(seg[i + 2]) - np.log(seg[i])) / 2 for i in range(len(seg) - 2)
                  if seg[i] >= thr and seg[i + 2] >= thr]
        return float(max(slopes)) if slopes else np.nan
    return dict(peak1_week=pk1, peak1=float(y[pk1 - 1]), peak2_week=pk2, peak2=float(y[pk2 - 1]),
                trough_week=tr, baseline=base, excess_wave1=float(b1), excess_wave2=float(b2),
                wave2_share=float(b2 / (b1 + b2)), growth_wave1=max_growth(2, pk1),
                growth_wave2=max_growth(tr, pk2))


def month_ticks(ax):
    # epi_week_of_season 1 starts ~1 July; label approximate month starts
    ticks = [1, 5.4, 9.7, 14, 18.4, 22.7, 27.1, 31.5, 35.4, 39.7, 44, 48.3]
    labs = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labs, fontsize=8)


def main():
    d = load_all()
    nat, st, meta = d["national"], d["state"], d["meta"]

    # ------------------------------------------------------------------ checks
    chk = validate(d)
    chk.to_csv(TAB / "00_data_checks.csv", index=False)
    print(chk.to_string(index=False))

    # ------------------------------------------------ national season totals
    pop = int(nat["population"].iloc[0])
    tot = nat.groupby("season")[["cases", "hospitalizations", "deaths"]].sum()
    tot["attack_rate_pct"] = 100 * tot["cases"] / pop
    tot["hosp_per_case_pct"] = 100 * tot["hospitalizations"] / tot["cases"]
    tot["cfr_pct"] = 100 * tot["deaths"] / tot["cases"]
    tot["deaths_per_hosp_pct"] = 100 * tot["deaths"] / tot["hospitalizations"]
    waves = pd.DataFrame({s: wave_decomposition(g.sort_values("epi_week_of_season")["cases"].to_numpy(float))
                          for s, g in nat.groupby("season")}).T
    nat_tab = tot.join(waves)
    nat_tab.to_csv(TAB / "01_national_season_summary.csv")
    print(nat_tab.round(3).T.to_string())

    # --------------------------------------------- per-state wave statistics
    rows = []
    for (sea, state), g in st.groupby(["season", "state"]):
        g = g.sort_values("epi_week_of_season")
        w = wave_decomposition(g["cases"].to_numpy(float))
        w.update(season=sea, state=state,
                 attack_rate_pct=100 * g["cases"].sum() / g["population"].iloc[0],
                 hosp_per_case_pct=100 * g["hospitalizations"].sum() / g["cases"].sum(),
                 cfr_pct=100 * g["deaths"].sum() / g["cases"].sum(),
                 deaths_per_100k=1e5 * g["deaths"].sum() / g["population"].iloc[0],
                 week1_ratio=g["cases"].iloc[0] / g["cases"].iloc[1])
        rows.append(w)
    sw = pd.DataFrame(rows).merge(meta, on="state")
    sw["baseline_per_100k"] = 1e5 * sw["baseline"] / sw["population"]
    sw.to_csv(TAB / "02_state_season_stats.csv", index=False)
    clim = sw.groupby(["season", "climate"])[["peak1_week", "peak2_week", "wave2_share", "attack_rate_pct",
                                               "hosp_per_case_pct", "cfr_pct", "growth_wave1",
                                               "baseline_per_100k"]].mean()
    clim.to_csv(TAB / "03_climate_means.csv")
    print(clim.round(3).to_string())

    # correlations with healthcare access (pooled over seasons, and partial on climate)
    corr_rows = []
    for y in ["hosp_per_case_pct", "cfr_pct", "attack_rate_pct", "wave2_share", "peak1_week"]:
        r_all = np.corrcoef(sw["hc_access"], sw[y])[0, 1]
        # within-climate correlation: demean both by season x climate
        dm = sw.groupby(["season", "climate"])[["hc_access", y]].transform(lambda v: v - v.mean())
        r_within = np.corrcoef(dm["hc_access"], dm[y])[0, 1]
        corr_rows.append(dict(variable=y, corr_with_hc_access=r_all, within_climate_corr=r_within))
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(TAB / "04_hc_access_correlations.csv", index=False)
    print(corr.round(3).to_string(index=False))

    # -------------------------------------------- reporting-noise diagnostics
    base = st[st["epi_week_of_season"].between(*BASELINE)]
    mv = base.groupby(["season", "state"])["cases"].agg(["mean", "var"]).reset_index()
    slope, icpt = np.polyfit(np.log(mv["mean"]), np.log(mv["var"]), 1)
    k_hat = (mv["mean"] ** 2 / (mv["var"] - mv["mean"]).clip(lower=1e-9))
    noise = pd.DataFrame({"loglog_slope_var_on_mean": [slope], "loglog_intercept": [icpt],
                          "median_NB_size_k": [k_hat.median()],
                          "median_var_over_mean": [(mv["var"] / mv["mean"]).median()],
                          "median_week1_ratio": [sw["week1_ratio"].median()]})
    noise.to_csv(TAB / "05_noise_diagnostics.csv", index=False)
    print(noise.round(3).to_string(index=False))

    # ================================================================ FIGURES
    # F1 national cases / hospitalisations / deaths, three seasons overlaid
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, col, lab in zip(axes, ["cases", "hospitalizations", "deaths"],
                            ["Weekly cases", "Weekly hospitalisations", "Weekly deaths"]):
        for s, g in nat.groupby("season"):
            ax.plot(g["epi_week_of_season"], g[col], "-o", ms=2.5, lw=1.4, color=SEASON_COL[s], label=s)
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.axvspan(0.5, 1.5, color="grey", alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    axes[0].text(1.7, axes[0].get_ylim()[1] * 0.5, "week-1 artefact", fontsize=8, color="grey")
    month_ticks(axes[0])
    axes[-1].set_xlabel("Epidemiological week of season (week 1 = first week of July)")
    fig.suptitle("Nigeria: national weekly influenza surveillance, 2023/24-2025/26 (log scale)")
    fig.tight_layout()
    fig.savefig(FIG / "F1_national_three_series.png")
    plt.close(fig)

    # F2 continuous national timeline (Sunday-start epi weeks)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(nat["epiweek_start"], nat["cases"] / 1e6, color="k", lw=1.3)
    for s, g in nat.groupby("season"):
        ax.axvspan(g["epiweek_start"].min(), g["epiweek_end"].max(), color=SEASON_COL[s], alpha=0.07)
        ax.text(g["epiweek_start"].min(), nat["cases"].max() / 1e6 * 0.95, s, color=SEASON_COL[s], fontsize=9)
    ax.set_ylabel("Weekly cases (millions)")
    ax.set_xlabel("Epi week start (Sunday)")
    ax.set_title("National weekly cases: main wave (Sep-Nov) and Harmattan wave (Dec-Jan) each season")
    fig.tight_layout()
    fig.savefig(FIG / "F2_national_timeline.png")
    plt.close(fig)

    # F3 North vs South incidence per 100k
    reg = regional_series(st, "region")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (s, g) in zip(axes, reg.groupby("season")):
        for r, gg in g.groupby("region"):
            ax.plot(gg["epi_week_of_season"], gg["cases_per_100k"], color=REG_COL[r], lw=1.6, label=r)
        ax.set_title(s)
        ax.set_xlabel("Epi week of season")
        month_ticks(ax)
    axes[0].set_ylabel("Weekly cases per 100,000")
    axes[0].legend(frameon=False)
    fig.suptitle("North vs South: the South leads the main wave; the North has the larger Harmattan wave", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "F3_north_south.png", bbox_inches="tight")
    plt.close(fig)

    # F4 climate-zone curves normalised to each curve's maximum (timing view)
    cz = regional_series(st, "climate")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (s, g) in zip(axes, cz.groupby("season")):
        for c, gg in g.groupby("climate"):
            y = gg["cases_per_100k"].to_numpy()
            ax.plot(gg["epi_week_of_season"], y / y[1:].max(), color=CLIM_COL[c], lw=1.6, label=c)
        ax.set_xlim(6, 36)
        ax.set_title(s)
        ax.set_xlabel("Epi week of season")
    axes[0].set_ylabel("Incidence / season maximum")
    axes[0].legend(frameon=False)
    fig.suptitle("Timing by climate zone: tropical -> savanna -> sahel lag in the main wave", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "F4_climate_timing.png", bbox_inches="tight")
    plt.close(fig)

    # F5 state heatmaps (log10 incidence per 100k), states ordered South->North
    order = (meta.assign(cl=meta["climate"].map({"tropical": 0, "savanna": 1, "sahel": 2}))
                 .sort_values(["cl", "hc_access"], ascending=[True, False])["state"].tolist())
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    for ax, (s, g) in zip(axes, st.groupby("season")):
        M = g.pivot(index="state", columns="epi_week_of_season", values="cases_per_100k").loc[order]
        im = ax.imshow(np.log10(M.to_numpy() + 1), aspect="auto", cmap="magma_r",
                       extent=[0.5, 52.5, len(order) - 0.5, -0.5], vmin=0, vmax=3.3)
        ax.set_title(s)
        ax.set_xlabel("Epi week of season")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{x} ({meta.set_index('state').loc[x, 'climate'][:3]})" for x in order], fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.6)
    cb.set_label("log10(weekly cases per 100k + 1)")
    fig.suptitle("State-level weekly incidence (tropical at top, sahel at bottom)")
    fig.savefig(FIG / "F5_state_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    # F6 state attack rates and mortality per 100k by season
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    piv_ar = sw.pivot(index="state", columns="season", values="attack_rate_pct").loc[order]
    piv_dm = sw.pivot(index="state", columns="season", values="deaths_per_100k").loc[order]
    yy = np.arange(len(order))
    for i, s in enumerate(piv_ar.columns):
        axes[0].barh(yy + (i - 1) * 0.27, piv_ar[s], height=0.27, color=SEASON_COL[s], label=s)
        axes[1].barh(yy + (i - 1) * 0.27, piv_dm[s], height=0.27, color=SEASON_COL[s])
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(order, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Reported attack rate (%)")
    axes[1].set_xlabel("Deaths per 100,000")
    axes[0].legend(frameon=False)
    fig.suptitle("State attack rates and mortality (states ordered tropical -> savanna -> sahel)")
    fig.tight_layout()
    fig.savefig(FIG / "F6_state_attack_mortality.png")
    plt.close(fig)

    # F7 healthcare access effects
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, y, lab in zip(axes, ["hosp_per_case_pct", "cfr_pct", "wave2_share"],
                          ["Hospitalisations per 100 cases", "Case fatality ratio (%)", "Harmattan-wave share of burden"]):
        for c, g in sw.groupby("climate"):
            ax.scatter(g["hc_access"], g[y], s=14, color=CLIM_COL[c], alpha=0.8, label=c)
        ax.set_xlabel("Healthcare accessibility index")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False)
    fig.suptitle("Metadata effects (all states x 3 seasons)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "F7_hc_access_effects.png", bbox_inches="tight")
    plt.close(fig)

    # F8 peak-week summary by state
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, col, ttl in zip(axes, ["peak1_week", "peak2_week"], ["Main-wave peak week", "Harmattan-wave peak week"]):
        for j, (s, g) in enumerate(sw.groupby("season")):
            for c, gg in g.groupby("climate"):
                xpos = {"tropical": 0, "savanna": 1, "sahel": 2}[c] + (j - 1) * 0.25
                jitter = np.random.default_rng(0).uniform(-0.06, 0.06, len(gg))
                ax.scatter(xpos + jitter, gg[col], s=12, color=SEASON_COL[s], alpha=0.8,
                           label=s if c == "tropical" else None)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["tropical", "savanna", "sahel"])
        ax.set_title(ttl)
    axes[0].set_ylabel("Epi week of season")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "F8_peak_weeks.png")
    plt.close(fig)

    # F9 reporting noise: mean-variance relation on the inter-epidemic plateau
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(mv["mean"], mv["var"], s=12, alpha=0.7)
    xs = np.logspace(np.log10(mv["mean"].min()), np.log10(mv["mean"].max()), 50)
    ax.plot(xs, xs, "k--", lw=1, label="Poisson (var = mean)")
    ax.plot(xs, np.exp(icpt) * xs ** slope, "r-", lw=1.2, label=f"fit: var ~ mean^{slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean weekly cases (plateau weeks 36-52)")
    ax.set_ylabel("Variance")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reporting noise is strongly over-dispersed")
    fig.tight_layout()
    fig.savefig(FIG / "F9_noise_mean_variance.png")
    plt.close(fig)

    # F10 severity ratios through time (national)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(nat["epiweek_start"], 100 * nat["hospitalizations"] / nat["cases"], lw=1, label="hosp / cases (%)")
    ax.plot(nat["epiweek_start"], 1000 * nat["deaths"] / nat["cases"], lw=1, label="deaths / cases (per 1,000)")
    ax.set_ylim(0, 3)
    ax.legend(frameon=False)
    ax.set_title("National severity ratios are stable over time (supports fixed observation fractions)")
    fig.tight_layout()
    fig.savefig(FIG / "F10_severity_ratios.png")
    plt.close(fig)

    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
