"""
validate.py - time-ordered (out-of-sample) validation.

Design (V1): every model is fitted to 2023/24 and 2024/25 only and then
forecasts the entire 2025/26 season before it starts (52 weeks ahead), which
is exactly the situation of the real 2026/27 forecast. The members and
candidate ensembles are scored on 2025/26 with the challenge metrics at the
national, North/South and state levels, and the mechanistic weight of the
final ensemble is chosen by the national weighted interval score (WIS) of
cases. Requires the two training-subset fits:
    TRAIN_SEASONS=2023/2024,2024/2025 python src/fit_national.py
    TRAIN_SEASONS=2023/2024,2024/2025 KAPPA_YEARS=4 python src/fit_national.py
Run:  python src/validate.py
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
from scoring import QUANTILES, crps_samples, score_quantile_forecast, seasonal_errors, wis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = ["2023/2024", "2024/2025"]
TEST = "2025/2026"
N_PATHS = 500
SEED = 11
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]          # candidate mechanistic weights


SHRINK = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # candidate interval-recalibration factors


def shrink(agg, c):
    """Recalibrate sample paths: on log(1+y), pull every path towards the
    weekly median by the factor c (c = 1: unchanged, c < 1: narrower)."""
    if c == 1.0:
        return agg
    out = {}
    for loc, ser in agg.items():
        out[loc] = {}
        for k, S in ser.items():
            L = np.log1p(S.astype(float))
            med = np.median(L, 0, keepdims=True)
            out[loc][k] = np.expm1(med + c * (L - med))
    return out


def member_draws(rng):
    states, meta, Ytr = fc.state_arrays(TRAIN)
    prof = fc.spatial_profiles(Ytr)
    mech = []
    for d, k in [("fit_train20232024_20242025", 1.0), ("fit_kappa4y_train20232024_20242025", 4.0)]:
        nat, back, sd = fc.mechanistic_national(ROOT / "outputs" / d, k, N_PATHS, rng)
        mech.append(fc.to_states(nat, back, sd, prof, rng))
    M = fc.mixture(mech, [1, 1], rng, N_PATHS)
    B = fc.analogue_states(Ytr, N_PATHS, rng)
    return states, meta, M, B


def score_all(agg, obs_agg, label):
    rows = []
    for loc, ser in agg.items():
        level = "national" if loc == "Nigeria" else ("region" if loc in ("North", "South") else "state")
        for c in fc.SERIES:
            S = ser[c].astype(float)
            y = obs_agg[loc][c].astype(float)
            q = fc.quantiles(S)
            sc = score_quantile_forecast(y[1:], q[1:])
            se = seasonal_errors(y[1:], np.median(S, 0)[1:])
            pk_s = S[:, 1:].argmax(1)
            rows.append(dict(model=label, location=loc, level=level, series=c, **sc,
                             CRPS=float(np.mean(crps_samples(y[1:], S[:, 1:].T))),
                             peak_week_obs=int(np.argmax(y[1:]) + 2),
                             peak_week_median=int(np.median(pk_s) + 2),
                             P_peak_within_1wk=float(np.mean(np.abs(pk_s - np.argmax(y[1:])) <= 1)),
                             peak_rel_err=se["peak_incidence_rel_error"],
                             cum_rel_err=se["cumulative_rel_error"]))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    states, meta, M, B = member_draws(rng)
    _, _, Yte = fc.state_arrays([TEST])
    obs = {c: Yte[c][0][None] for c in fc.SERIES}
    obs_agg = {k: {c: v[c][0] for c in fc.SERIES} for k, v in fc.aggregate(obs, meta).items()}

    rows, store = [], {}
    for w in WEIGHTS:
        mix = fc.mixture([M, B], [w, 1 - w], rng, N_PATHS) if 0 < w < 1 else (M if w == 1 else B)
        lab = {0.0: "Benchmark (analogue)", 1.0: "Mechanistic"}.get(w, f"Ensemble w_mech={w}")
        agg = fc.aggregate(mix, meta)
        store[w] = agg
        rows += score_all(agg, obs_agg, lab)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "validation_scores_by_location.csv", index=False)
    summ = (df.groupby(["model", "level", "series"])[["MAE", "RMSE", "WIS", "CRPS", "Coverage50", "Coverage90",
                                                      "P_peak_within_1wk", "cum_rel_err"]]
              .mean().reset_index())
    summ.to_csv(OUT / "validation_summary.csv", index=False)
    nat = summ[(summ["level"] == "national") & (summ["series"] == "cases")].set_index("model")
    # selection rule: mean WIS relative to the benchmark over all 40 locations x 3 series
    piv = df.pivot_table(index=["location", "series"], columns="model", values="WIS")
    bench_wis = piv["Benchmark (analogue)"]
    rel = (piv.div(bench_wis, axis=0)).mean()
    best = rel.idxmin()
    w_best = {"Benchmark (analogue)": 0.0, "Mechanistic": 1.0}.get(best)
    if w_best is None:
        w_best = float(best.split("=")[1])
    # interval recalibration factor for the selected ensemble
    crow = []
    for c in SHRINK:
        r = pd.DataFrame(score_all(shrink(store[w_best], c), obs_agg, f"c={c}"))
        rr = r.set_index(["location", "series"])["WIS"].div(bench_wis).mean()
        crow.append(dict(c=c, mean_relative_WIS=rr, coverage50=r["Coverage50"].mean(),
                         coverage90=r["Coverage90"].mean()))
    cdf = pd.DataFrame(crow)
    cdf.to_csv(OUT / "recalibration_factor_search.csv", index=False)
    c_best = float(cdf.loc[cdf["mean_relative_WIS"].idxmin(), "c"])
    store["final"] = shrink(store[w_best], c_best)
    fin = pd.DataFrame(score_all(store["final"], obs_agg, "Final (recalibrated ensemble)"))
    df = pd.concat([df, fin])
    df.to_csv(OUT / "validation_scores_by_location.csv", index=False)
    summ = (df.groupby(["model", "level", "series"])[["MAE", "RMSE", "WIS", "CRPS", "Coverage50", "Coverage90",
                                                      "P_peak_within_1wk", "cum_rel_err"]]
              .mean().reset_index())
    summ.to_csv(OUT / "validation_summary.csv", index=False)
    print(cdf.round(3).to_string(index=False))
    json.dump({"w_mech": w_best, "shrink_c": c_best,
               "selected_by": "mean WIS relative to benchmark over 40 locations x 3 series",
               "national_cases_WIS": nat["WIS"].to_dict(),
               "mean_relative_WIS_vs_benchmark": rel.to_dict(),
               "final_mean_relative_WIS": float(cdf["mean_relative_WIS"].min())},
              open(OUT / "ensemble_weight.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print("selected w_mech =", w_best)
    print("mean relative WIS vs benchmark:\n", rel.round(3))

    # ---------------- figure: national validation fan charts (selected ensemble vs members)
    wk = np.arange(1, 53)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for col, (w, lab) in enumerate([(1.0, "Mechanistic"), (0.0, "Benchmark (analogue)"),
                                    ("final", f"Final ensemble (w_mech = {w_best}, c = {c_best})")]):
        agg = store[w]["Nigeria"]
        for row, c in enumerate(fc.SERIES):
            ax = axes[row, col]
            q = fc.quantiles(agg[c].astype(float))
            ax.fill_between(wk, q[:, 2], q[:, 20], color="#9ecae1", alpha=0.6, label="90% PI")
            ax.fill_between(wk, q[:, 6], q[:, 16], color="#3182bd", alpha=0.6, label="50% PI")
            ax.plot(wk, q[:, 11], color="#08519c", lw=1.8, label="median")
            ax.plot(wk, obs_agg["Nigeria"][c], "ko", ms=3, label="observed 2025/26")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(lab, fontsize=11)
            if col == 0:
                ax.set_ylabel(["Weekly cases", "Weekly hospitalisations", "Weekly deaths"][row])
            if row == 2:
                ax.set_xlabel("Epi week of season")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Out-of-sample validation: pre-season forecasts of the 2025/26 season (trained on 2023/24-2024/25)")
    fig.tight_layout()
    fig.savefig(FIG / "F37_validation_national_2025_26.png", dpi=200)
    plt.close(fig)

    # ---------------- figure: WIS by level and model, and coverage
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    order = ["Benchmark (analogue)", "Ensemble w_mech=0.25", "Ensemble w_mech=0.5", "Ensemble w_mech=0.75",
             "Mechanistic", "Final (recalibrated ensemble)"]
    sub = summ[summ["series"] == "cases"]
    xs = np.arange(3)
    for i, m in enumerate(order):
        vals = [sub[(sub["model"] == m) & (sub["level"] == lv)]["WIS"].values[0] for lv in ["national", "region", "state"]]
        axes[0].bar(xs + (i - 2.5) * 0.14, vals, width=0.14, label=m)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["National", "North/South (mean)", "States (mean)"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean WIS, weekly cases")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Weighted interval score, 2025/26 (lower is better)")
    nominal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
    for m, w in [("Benchmark (analogue)", 0.0), ("Mechanistic", 1.0), ("Final ensemble", "final")]:
        cov = []
        for a in nominal:
            hits, tot = 0, 0
            for loc, ser in store[w].items():
                for c in fc.SERIES:
                    S = ser[c][:, 1:].astype(float)
                    lo, hi = np.quantile(S, [(1 - a) / 2, 1 - (1 - a) / 2], axis=0)
                    y = obs_agg[loc][c][1:]
                    hits += np.sum((y >= lo) & (y <= hi))
                    tot += len(y)
            cov.append(hits / tot)
        axes[1].plot(nominal, cov, "o-", label=m)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("Nominal central interval coverage")
    axes[1].set_ylabel("Empirical coverage (all locations, series)")
    axes[1].set_title("Calibration of prediction intervals, 2025/26")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "F38_validation_scores_calibration.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
