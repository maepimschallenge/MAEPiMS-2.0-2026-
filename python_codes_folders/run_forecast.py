"""
run_forecast.py - probabilistic forecasts for the 2026/27 season (52
epidemiological weeks, Sunday 28 June 2026 - Saturday 26 June 2027) for
Nigeria, Northern and Southern Nigeria and all 36 states + FCT.

Final ensemble (chosen by time-ordered validation, src/validate.py):
  mechanistic weight w_mech and interval recalibration factor c read from
  outputs/validation/ensemble_weight.json; mechanistic members fitted to all
  three seasons with 1-year (outputs/fit) and 4-year (outputs/fit_kappa4y)
  immunity; benchmark = analogue seasons 2023/24-2025/26.

Outputs (outputs/forecast/):
  forecast_2026_27_weekly_quantiles.csv   weekly quantile forecasts (23 levels)
  forecast_2026_27_seasonal_targets.csv   peak week, peak incidence, cumulative
                                          burden and attack rate (quantiles)
  forecast_2026_27_probabilities.csv      P(peak in 4-week windows), weekly
                                          P(increase) / P(decrease)
Figures F39-F45 in outputs/figures/.
Run: python src/run_forecast.py   (about 2 minutes)
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
import forecast as fc  # noqa: E402
from data_io import load_all, mmwr_week  # noqa: E402
from scoring import QUANTILES  # noqa: E402
from validate import shrink  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "forecast"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
N_MECH = 1000            # paths per mechanistic member
N_TOTAL = 2000
SEED = 2027
START = pd.Timestamp("2026-06-28")          # Sunday starting epi week 1 of 2026/27
SEASON_TAG = "2026/2027"
WINDOWS = [(a, a + 3) for a in range(1, 53, 4)]

# approximate state centroids (latitude, longitude) for schematic maps
CENTROID = {
    "Abia": (5.4, 7.5), "Adamawa": (9.3, 12.4), "Akwa Ibom": (5.0, 7.9), "Anambra": (6.2, 6.9),
    "Bauchi": (10.6, 9.8), "Bayelsa": (4.8, 6.1), "Benue": (7.3, 8.7), "Borno": (11.8, 13.2),
    "Cross River": (5.9, 8.6), "Delta": (5.7, 6.0), "Ebonyi": (6.3, 8.0), "Edo": (6.6, 5.9),
    "Ekiti": (7.7, 5.3), "Enugu": (6.5, 7.4), "FCT": (8.9, 7.2), "Gombe": (10.3, 11.2),
    "Imo": (5.5, 7.0), "Jigawa": (12.2, 9.6), "Kaduna": (10.4, 7.7), "Kano": (11.9, 8.5),
    "Katsina": (12.5, 7.6), "Kebbi": (11.5, 4.2), "Kogi": (7.8, 6.7), "Kwara": (8.9, 4.6),
    "Lagos": (6.5, 3.4), "Nasarawa": (8.5, 8.2), "Niger": (9.9, 5.6), "Ogun": (7.0, 3.4),
    "Ondo": (7.1, 5.0), "Osun": (7.6, 4.5), "Oyo": (8.0, 3.6), "Plateau": (9.2, 9.5),
    "Rivers": (4.8, 6.9), "Sokoto": (13.0, 5.3), "Taraba": (8.0, 10.5), "Yobe": (12.3, 11.4),
    "Zamfara": (12.2, 6.2)}


def build_ensemble(rng):
    cfg = json.loads((ROOT / "outputs" / "validation" / "ensemble_weight.json").read_text())
    states, meta, Y = fc.state_arrays(SEASONS)
    prof = fc.spatial_profiles(Y)
    mech = []
    for d, k in [("fit", 1.0), ("fit_kappa4y", 4.0)]:
        nat, back, sd = fc.mechanistic_national(ROOT / "outputs" / d, k, N_MECH, rng)
        mech.append(fc.to_states(nat, back, sd, prof, rng))
    M = fc.mixture(mech, [1, 1], rng, N_TOTAL)
    B = fc.analogue_states(Y, N_TOTAL, rng)
    w = cfg["w_mech"]
    mix = fc.mixture([M, B], [w, 1 - w], rng, N_TOTAL) if 0 < w < 1 else (M if w == 1 else B)
    agg = shrink(fc.aggregate(mix, meta), cfg["shrink_c"])
    return agg, meta, cfg, Y


def main():
    rng = np.random.default_rng(SEED)
    agg, meta, cfg, Y = build_ensemble(rng)
    pop = {"Nigeria": meta["population"].sum(),
           "North": meta.loc[meta["zone"].isin(["NC", "NE", "NW"]), "population"].sum(),
           "South": meta.loc[~meta["zone"].isin(["NC", "NE", "NW"]), "population"].sum(),
           **meta["population"].to_dict()}
    region_of = {s: ("North" if z in ("NC", "NE", "NW") else "South") for s, z in meta["zone"].items()}
    weeks = np.arange(1, 53)
    ew_start = [START + pd.Timedelta(days=7 * (w - 1)) for w in weeks]
    ew_end = [d + pd.Timedelta(days=6) for d in ew_start]
    mm = [mmwr_week(d.date()) for d in ew_start]

    # ------------------------------------------------- weekly quantile CSV
    rows = []
    for loc, ser in agg.items():
        ltype = "national" if loc == "Nigeria" else ("region" if loc in ("North", "South") else "state")
        reg = loc if ltype != "state" else region_of[loc]
        for c in fc.SERIES:
            q = fc.quantiles(ser[c])
            for t in range(52):
                for k, tau in enumerate(QUANTILES):
                    rows.append((SEASON_TAG, loc, ltype, reg, c, ew_start[t].date(), ew_end[t].date(),
                                 mm[t][0], mm[t][1], t + 1, tau, round(float(q[t, k]), 1)))
    wq = pd.DataFrame(rows, columns=["season", "location", "location_type", "region", "target",
                                     "epiweek_start_sunday", "epiweek_end_saturday", "mmwr_year",
                                     "mmwr_week", "season_week", "quantile", "value"])
    wq.to_csv(OUT / "forecast_2026_27_weekly_quantiles.csv", index=False)

    # ------------------------------------------------- seasonal targets
    srows, prows, summ = [], [], []
    for loc, ser in agg.items():
        ltype = "national" if loc == "Nigeria" else ("region" if loc in ("North", "South") else "state")
        C, H, D = (ser[c].astype(float) for c in fc.SERIES)
        pk = C[:, 1:].argmax(1) + 2                                   # season week (week 1 excluded)
        targ = {"peak_week": pk.astype(float),
                "peak_incidence_per_100k": C[:, 1:].max(1) / pop[loc] * 1e5,
                "cumulative_cases": C.sum(1), "cumulative_hospitalizations": H.sum(1),
                "cumulative_deaths": D.sum(1), "attack_rate_pct": 100 * C.sum(1) / pop[loc],
                "deaths_per_100k": 1e5 * D.sum(1) / pop[loc]}
        for name, v in targ.items():
            for tau in QUANTILES:
                srows.append((SEASON_TAG, loc, ltype, name, tau, float(np.quantile(v, tau))))
        for a, b in WINDOWS:
            prows.append((SEASON_TAG, loc, ltype, f"P(peak in season weeks {a}-{b})",
                          str(ew_start[a - 1].date()), float(np.mean((pk >= a) & (pk <= b)))))
        for t in range(51):
            inc = float(np.mean(C[:, t + 1] > C[:, t]))
            prows.append((SEASON_TAG, loc, ltype, f"P(increase) week {t + 1}->{t + 2}",
                          str(ew_start[t + 1].date()), inc))
            prows.append((SEASON_TAG, loc, ltype, f"P(decrease) week {t + 1}->{t + 2}",
                          str(ew_start[t + 1].date()), float(np.mean(C[:, t + 1] < C[:, t]))))
        summ.append(dict(location=loc, level=ltype,
                         peak_week_median=float(np.median(pk)),
                         peak_week_date=str(ew_start[int(np.median(pk)) - 1].date()),
                         peak_week_50=f"{np.quantile(pk, .25):.0f}-{np.quantile(pk, .75):.0f}",
                         peak_inc_100k=np.median(targ["peak_incidence_per_100k"]),
                         cum_cases_med=np.median(C.sum(1)), cum_cases_lo=np.quantile(C.sum(1), .05),
                         cum_cases_hi=np.quantile(C.sum(1), .95),
                         cum_hosp_med=np.median(H.sum(1)), cum_hosp_lo=np.quantile(H.sum(1), .05),
                         cum_hosp_hi=np.quantile(H.sum(1), .95),
                         cum_deaths_med=np.median(D.sum(1)), cum_deaths_lo=np.quantile(D.sum(1), .05),
                         cum_deaths_hi=np.quantile(D.sum(1), .95),
                         attack_rate_med=np.median(targ["attack_rate_pct"]),
                         attack_rate_lo=np.quantile(targ["attack_rate_pct"], .05),
                         attack_rate_hi=np.quantile(targ["attack_rate_pct"], .95),
                         deaths_100k_med=np.median(targ["deaths_per_100k"]),
                         P_main_peak_wk13_18=float(np.mean((pk >= 13) & (pk <= 18))),
                         P_peak_after_wk22=float(np.mean(pk >= 22))))
    pd.DataFrame(srows, columns=["season", "location", "location_type", "target", "quantile", "value"]) \
        .to_csv(OUT / "forecast_2026_27_seasonal_targets.csv", index=False)
    pd.DataFrame(prows, columns=["season", "location", "location_type", "quantity", "reference_date", "probability"]) \
        .to_csv(OUT / "forecast_2026_27_probabilities.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "forecast_2026_27_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm[sm["level"] != "state"].round(2).T.to_string())
    print(sm[sm["level"] == "state"][["location", "peak_week_median", "attack_rate_med", "deaths_100k_med"]]
          .sort_values("attack_rate_med").round(2).to_string(index=False))

    # ===================================================== FIGURES
    d = load_all()
    nat = d["national"]
    # F39 national fan charts with history
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, c, lab in zip(axes, fc.SERIES, ["Weekly cases", "Weekly hospitalisations", "Weekly deaths"]):
        ax.plot(nat["epiweek_start"], nat[c], "k.-", ms=3, lw=0.8, label="observed 2023/24-2025/26")
        q = fc.quantiles(agg["Nigeria"][c].astype(float))
        ax.fill_between(ew_start, q[:, 2], q[:, 20], color="#fdae6b", alpha=0.6, label="90% PI")
        ax.fill_between(ew_start, q[:, 6], q[:, 16], color="#e6550d", alpha=0.5, label="50% PI")
        ax.plot(ew_start, q[:, 11], color="#a63603", lw=2, label="median forecast 2026/27")
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.axvline(START, color="grey", ls="--", lw=1)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[-1].set_xlabel("Epidemiological week start (Sunday)")
    fig.suptitle("Nigeria: probabilistic forecast for the 2026/27 season with observed history")
    fig.tight_layout()
    fig.savefig(FIG / "F39_forecast_national_2026_27.png", dpi=200)
    plt.close(fig)

    # F40 North vs South per 100k
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, c, lab in zip(axes, fc.SERIES, ["Cases", "Hospitalisations", "Deaths"]):
        for reg, col in [("North", "#c0392b"), ("South", "#2471a3")]:
            q = fc.quantiles(agg[reg][c].astype(float)) / pop[reg] * 1e5
            ax.fill_between(weeks, q[:, 2], q[:, 20], color=col, alpha=0.15)
            ax.fill_between(weeks, q[:, 6], q[:, 16], color=col, alpha=0.3)
            ax.plot(weeks, q[:, 11], color=col, lw=2, label=reg)
        ax.set_title(f"{lab} per 100,000")
        ax.set_xlabel("Season week (week 1 = 28 Jun 2026)")
    axes[0].legend(frameon=False)
    fig.suptitle("Northern vs Southern Nigeria, 2026/27 forecast (median, 50% and 90% PI)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "F40_forecast_north_south_2026_27.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F41 state heatmap of median weekly incidence per 100k
    order = (meta.assign(cl=meta["climate"].map({"tropical": 0, "savanna": 1, "sahel": 2}))
                 .sort_values(["cl", "hc_access"], ascending=[True, False]).index.tolist())
    M = np.array([np.median(agg[s]["cases"], 0) / pop[s] * 1e5 for s in order])
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(np.log10(M + 1), aspect="auto", cmap="magma_r", extent=[0.5, 52.5, len(order) - 0.5, -0.5])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{s} ({meta.loc[s, 'climate'][:3]})" for s in order], fontsize=7)
    ax.set_xlabel("Season week (week 1 = 28 Jun 2026)")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("log10(median weekly cases per 100k + 1)")
    ax.set_title("2026/27 forecast: state weekly incidence (tropical top, sahel bottom)")
    fig.tight_layout()
    fig.savefig(FIG / "F41_forecast_state_heatmap_2026_27.png", dpi=200)
    plt.close(fig)

    # F42 / F43 schematic maps (approximate state centroids)
    smi = sm.set_index("location")
    for fname, col, lab, cmap in [("F42_forecast_state_attack_rate_map.png", "attack_rate_med",
                                   "Median reported attack rate (%)", "YlOrRd"),
                                  ("F43_forecast_state_mortality_map.png", "deaths_100k_med",
                                   "Median deaths per 100,000", "PuRd")]:
        fig, ax = plt.subplots(figsize=(9, 8))
        lat = np.array([CENTROID[s][0] for s in meta.index])
        lon = np.array([CENTROID[s][1] for s in meta.index])
        val = smi.loc[meta.index, col].to_numpy()
        sc = ax.scatter(lon, lat, c=val, s=meta["population"].to_numpy() / 6e4 + 150, cmap=cmap,
                        edgecolor="k", linewidth=0.6)
        for s, x, y, v in zip(meta.index, lon, lat, val):
            ax.annotate(f"{s}\n{v:.2f}" if col == "deaths_100k_med" else f"{s}\n{v:.1f}", (x, y),
                        fontsize=6.5, ha="center", va="center")
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(lab)
        ax.set_xlabel("Longitude (approx.)")
        ax.set_ylabel("Latitude (approx.)")
        ax.set_title(f"2026/27 forecast: {lab.lower()} by state\n(schematic map, approximate state centroids; "
                     "marker size ~ population)", fontsize=10)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=200)
        plt.close(fig)

    # F44 peak-week distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
    for reg, col in [("Nigeria", "k"), ("North", "#c0392b"), ("South", "#2471a3")]:
        pk = agg[reg]["cases"][:, 1:].argmax(1) + 2
        h = np.bincount(pk, minlength=53)[1:] / len(pk)
        axes[0].plot(weeks, h, "-o", ms=3, color=col, label=reg)
    axes[0].set_xlabel("Season week of peak weekly cases")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Forecast distribution of the peak week, 2026/27")
    axes[0].legend(frameon=False)
    for clim, col in [("tropical", "#1e6091"), ("savanna", "#6b8e23"), ("sahel", "#b8860b")]:
        ss = meta.index[meta["climate"] == clim]
        axes[1].scatter(meta.loc[ss, "hc_access"], smi.loc[ss, "peak_week_median"], color=col, label=clim, s=30)
    axes[1].set_xlabel("Healthcare accessibility index")
    axes[1].set_ylabel("Median forecast peak week")
    axes[1].set_title("State median peak week by climate zone")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "F44_forecast_peak_week_2026_27.png", dpi=200)
    plt.close(fig)

    # F45 probability of increasing transmission
    fig, ax = plt.subplots(figsize=(13, 4))
    for reg, col in [("Nigeria", "k"), ("North", "#c0392b"), ("South", "#2471a3")]:
        C = agg[reg]["cases"].astype(float)
        p_inc = (C[:, 1:] > C[:, :-1]).mean(0)
        ax.plot(weeks[1:], p_inc, "-", color=col, lw=1.8, label=reg)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Season week (change from previous week)")
    ax.set_ylabel("P(weekly cases increase)")
    ax.set_title("Forecast probability of increasing transmission, 2026/27")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "F45_forecast_prob_increase_2026_27.png", dpi=200)
    plt.close(fig)
    print("forecast CSVs ->", OUT, "; figures F39-F45 ->", FIG)


if __name__ == "__main__":
    main()
