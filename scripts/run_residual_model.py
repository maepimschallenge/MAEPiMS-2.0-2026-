"""
Residual quantile ML layer end-to-end run, leave-one-season-out.

For each of the 3 held-out seasons:
  1. Build the calibration baseline for the HELD-OUT season the normal way (zone
     offsets fit on both training seasons, phase_weight=0.25 default).
  2. Build the calibration baseline for EACH TRAINING season using zone offsets fit
     from the *other* training season only -- so residual-model training rows aren't
     built from a baseline that already "knows" their own season's pattern (the same
     leakage src/calibration.py's postmortem warns about, one level down).
  3. Fit LightGBM quantile models (src.residual_model) on the training seasons'
     (baseline, observed) log-ratio residuals.
  4. Predict quantiles for the held-out season from its own baseline, and score:
     median MAE/RMSE against baseline-only, WIS/CRPS/quantile loss, peak-week
     accuracy, and interval coverage (PIT).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import load_national, load_by_state, load_state_metadata
from src.renewal import fit_series, generation_kernel, N_SEED
from src.calibration import (
    fit_all_states, allocate_with_phase_shift,
    build_shape_delta_table, fit_zone_shape_offsets, zone_offsets_dict,
)
from src.residual_model import (
    build_feature_target_table, fit_quantile_models, predict_log_ratio_quantiles,
    reconstruct_quantile_forecast,
)
from src.metrics import (
    mae, rmse, weighted_interval_score, crps_from_quantiles, mean_quantile_loss,
    peak_week_accuracy, pit_values,
)

SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
national = load_national()
by_state = load_by_state()
meta = load_state_metadata()
kernel = generation_kernel()

print("Fitting national + per-state renewal cores for all 3 seasons...")
national_params, national_sims, national_weeks, national_seeds, national_pops = {}, {}, {}, {}, {}
state_fits = {}
for season in SEASONS:
    d = national[national["season"] == season].sort_values("epi_week_of_season")
    weeks = d["epi_week_of_season"].to_numpy(dtype=float)
    observed = d["cases"].to_numpy(dtype=float)
    population = float(d["population"].iloc[0])
    params, sim = fit_series(weeks, observed, population, kernel)
    national_params[season] = params
    national_sims[season] = sim
    national_weeks[season] = weeks
    national_seeds[season] = observed[:N_SEED]
    national_pops[season] = population
    state_fits[season] = fit_all_states(by_state[by_state["season"] == season], params)
    print(f"  {season}: national MAE={mae(observed, sim):,.0f}, {len(state_fits[season])} states fit")


def baseline_for(season: str, offset_source_seasons: list[str], phase_weight: float = 0.25) -> pd.DataFrame:
    """Calibration-layer baseline for `season`, using zone offsets fit only from
    `offset_source_seasons` (never `season` itself)."""
    delta_table = build_shape_delta_table(
        {s: state_fits[s] for s in offset_source_seasons},
        {s: national_params[s] for s in offset_source_seasons},
        meta,
    )
    zone_offsets = fit_zone_shape_offsets(delta_table)
    offsets_dict = zone_offsets_dict(zone_offsets)
    return allocate_with_phase_shift(
        national_params[season], offsets_dict, meta,
        national_weeks[season], national_seeds[season], national_pops[season], kernel,
        phase_weight=phase_weight,
    )


summary_rows = []
for test_season in SEASONS:
    train_seasons = [s for s in SEASONS if s != test_season]
    print(f"\n{'='*70}\nHeld-out season: {test_season}  (trained on {train_seasons})\n{'='*70}")

    # ---- leak-free baselines + feature/target tables for the 2 training seasons --
    train_tables = []
    for s in train_seasons:
        other = [x for x in train_seasons if x != s]  # the *other* training season only
        base = baseline_for(s, other)
        tbl = build_feature_target_table(
            by_state[by_state["season"] == s], base, meta,
            national_params[s].t1, national_params[s].t2, s,
        )
        train_tables.append(tbl)
    train_df = pd.concat(train_tables, ignore_index=True)

    # ---- fit quantile models -----------------------------------------------------
    models = fit_quantile_models(train_df)

    # ---- held-out season: standard baseline (offsets from both training seasons) -
    test_baseline = baseline_for(test_season, train_seasons)
    test_by_state = by_state[by_state["season"] == test_season]
    test_df = build_feature_target_table(
        test_by_state, test_baseline, meta,
        national_params[test_season].t1, national_params[test_season].t2, test_season,
    )

    log_ratio_q = predict_log_ratio_quantiles(models, test_df)
    count_q = reconstruct_quantile_forecast(test_df["baseline"].to_numpy(), log_ratio_q)

    observed = test_df["observed"].to_numpy()
    baseline_only = test_df["baseline"].to_numpy()
    median_corrected = count_q[0.5]

    print(f"Median MAE   baseline-only={mae(observed, baseline_only):>9,.1f}   "
          f"residual-corrected={mae(observed, median_corrected):>9,.1f}")
    print(f"Median RMSE  baseline-only={rmse(observed, baseline_only):>9,.1f}   "
          f"residual-corrected={rmse(observed, median_corrected):>9,.1f}")
    wis = weighted_interval_score(observed, median_corrected, count_q)
    crps = crps_from_quantiles(observed, count_q)
    mql = mean_quantile_loss(observed, count_q)
    print(f"WIS={wis:,.1f}   CRPS_approx={crps:,.1f}   mean_quantile_loss={mql:,.1f}")

    pit = pit_values(observed, count_q)
    band90 = np.mean((pit >= 0.025) & (pit <= 0.975))
    print(f"PIT mean={pit.mean():.3f} (well-calibrated ~0.5)   "
          f"frac inside nominal-90% band={band90:.1%} (target ~90%)")

    # peak-week accuracy per state (median-corrected vs baseline-only)
    median_series = pd.Series(median_corrected, index=test_df.index)
    rows = []
    for state, d in test_df.groupby("state"):
        d = d.sort_values("epi_week_of_season")
        weeks = d["epi_week_of_season"].to_numpy()
        true_pk = weeks[d["observed"].to_numpy().argmax()]
        base_pk = weeks[d["baseline"].to_numpy().argmax()]
        med = median_series.loc[d.index].to_numpy()
        corr_pk = weeks[med.argmax()]
        rows.append({"state": state, "true_pk": true_pk, "base_pk": base_pk, "corr_pk": corr_pk})
    pk_df = pd.DataFrame(rows)
    base_acc = peak_week_accuracy(pk_df["true_pk"], pk_df["base_pk"])
    corr_acc = peak_week_accuracy(pk_df["true_pk"], pk_df["corr_pk"])
    print(f"Peak-week acc (+-1wk)  baseline-only={base_acc:.2%}   residual-corrected={corr_acc:.2%}")

    summary_rows.append({
        "test_season": test_season,
        "mae_baseline": mae(observed, baseline_only), "mae_corrected": mae(observed, median_corrected),
        "wis": wis, "crps": crps, "pit_mean": pit.mean(), "band90_coverage": band90,
        "peak_acc_baseline": base_acc, "peak_acc_corrected": corr_acc,
    })

summary = pd.DataFrame(summary_rows)
print(f"\n{'='*70}\nLEAVE-ONE-SEASON-OUT SUMMARY - residual quantile layer\n{'='*70}")
print(summary.to_string(index=False))
out = Path(__file__).resolve().parents[1] / "outputs" / "residual_model_loso_summary.csv"
summary.to_csv(out, index=False)
print(f"\nSaved {out}")
