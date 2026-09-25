"""
Component 4 end-to-end run: ties the validated Components 1-3 together into the
actual submission artifacts, leave-one-season-out across all 3 seasons (same
discipline as scripts/run_calibration_loso.py and run_residual_model.py - every
number here is out-of-sample for the season it's reported against).

For each held-out season, produces:
  - outputs/forecast_<season>.csv       state-week quantile forecasts, cases +
                                         hospitalisations + deaths (long format)
  - outputs/seasonal_summary_<season>.csv   per-state peak week/incidence + cumulative
                                             case quantiles
  - outputs/regional_national_<season>.csv  North/South/national aggregate quantiles
                                             (Monte Carlo summed, see src/ensemble.py)
Plus one demonstration of the optional probabilistic targets (P(rising transmission),
P(peak within k weeks), full peak-week distribution) via the parametric bootstrap,
printed for the most recent held-out season only (they're cheap to add for every
season/state in the real submission - shown once here to keep the run readable).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import load_national, load_by_state, load_state_metadata
from src.renewal import fit_series, fit_series_with_covariance, fit_rate_ratios, generation_kernel, N_SEED
from src.calibration import (
    fit_all_states, allocate_with_phase_shift,
    build_shape_delta_table, fit_zone_shape_offsets, zone_offsets_dict,
)
from src.residual_model import build_feature_target_table, fit_quantile_models, predict_log_ratio_quantiles, reconstruct_quantile_forecast, QUANTILE_LEVELS
from src.ensemble import (
    bootstrap_params, bootstrap_trajectories, probability_of_rising, probability_peak_within,
    peak_week_distribution, build_state_quantile_frame, apply_rate_ratio, aggregate_via_monte_carlo,
    build_seasonal_summary,
)
from src.metrics import mae

SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
NORTH_ZONES = {"NW", "NC", "NE"}
national = load_national()
by_state = load_by_state()
meta = load_state_metadata()
kernel = generation_kernel()
rng = np.random.default_rng(42)

print("Fitting national + per-state renewal cores for all 3 seasons...")
national_params, national_weeks, national_seeds, national_pops = {}, {}, {}, {}
state_fits = {}
for season in SEASONS:
    d = national[national["season"] == season].sort_values("epi_week_of_season")
    weeks = d["epi_week_of_season"].to_numpy(dtype=float)
    observed = d["cases"].to_numpy(dtype=float)
    population = float(d["population"].iloc[0])
    params, sim = fit_series(weeks, observed, population, kernel)
    national_params[season] = params
    national_weeks[season] = weeks
    national_seeds[season] = observed[:N_SEED]
    national_pops[season] = population
    state_fits[season] = fit_all_states(by_state[by_state["season"] == season], params)


def baseline_for(season, offset_source_seasons, phase_weight=0.25):
    delta_table = build_shape_delta_table(
        {s: state_fits[s] for s in offset_source_seasons},
        {s: national_params[s] for s in offset_source_seasons}, meta,
    )
    offsets_dict = zone_offsets_dict(fit_zone_shape_offsets(delta_table))
    return allocate_with_phase_shift(
        national_params[season], offsets_dict, meta,
        national_weeks[season], national_seeds[season], national_pops[season], kernel,
        phase_weight=phase_weight,
    )


for test_season in SEASONS:
    train_seasons = [s for s in SEASONS if s != test_season]
    print(f"\n{'='*70}\nForecasting {test_season}  (trained on {train_seasons})\n{'='*70}")

    # ---- residual quantile model, trained leak-free on the 2 training seasons -----
    train_tables = []
    for s in train_seasons:
        other = [x for x in train_seasons if x != s]
        base = baseline_for(s, other)
        train_tables.append(build_feature_target_table(
            by_state[by_state["season"] == s], base, meta, national_params[s].t1, national_params[s].t2, s,
        ))
    train_df = pd.concat(train_tables, ignore_index=True)
    models = fit_quantile_models(train_df)

    # ---- per-state case quantile forecasts for the held-out season ----------------
    test_baseline = baseline_for(test_season, train_seasons)
    test_by_state = by_state[by_state["season"] == test_season].sort_values(["state", "epi_week_of_season"])
    weeks = national_weeks[test_season]
    week_starts = test_by_state[test_by_state["state"] == meta["state"].iloc[0]]["week_start"].to_numpy()

    test_df = build_feature_target_table(test_by_state, test_baseline, meta, national_params[test_season].t1, national_params[test_season].t2, test_season)
    log_ratio_q = predict_log_ratio_quantiles(models, test_df)
    count_q_flat = reconstruct_quantile_forecast(test_df["baseline"].to_numpy(), log_ratio_q)

    # reshape flat (state,week)-ordered arrays back into {state: {q: array}}
    state_case_quantiles = {}
    for state, idx in test_df.groupby("state").groups.items():
        pos = test_df.index.get_indexer(idx)
        state_case_quantiles[state] = {q: count_q_flat[q][pos] for q in QUANTILE_LEVELS}

    # ---- per-state hospitalisation / death rate ratios, fit on training seasons ---
    train_state_data = by_state[by_state["season"].isin(train_seasons)]
    hosp_quantiles, death_quantiles = {}, {}
    for state in meta["state"]:
        d = train_state_data[train_state_data["state"] == state]
        ratios = fit_rate_ratios(d["cases"].to_numpy(), d["hospitalizations"].to_numpy(), d["deaths"].to_numpy())
        hosp_quantiles[state] = apply_rate_ratio({state: state_case_quantiles[state]}, ratios["hosp_ratio"])[state]
        death_quantiles[state] = apply_rate_ratio({state: state_case_quantiles[state]}, ratios["death_ratio"])[state]

    # ---- assemble + save the long-format forecast table ---------------------------
    cases_tbl = build_state_quantile_frame(state_case_quantiles, weeks, week_starts, test_season, "cases")
    hosp_tbl = build_state_quantile_frame(hosp_quantiles, weeks, week_starts, test_season, "hospitalizations")
    death_tbl = build_state_quantile_frame(death_quantiles, weeks, week_starts, test_season, "deaths")
    forecast_tbl = pd.concat([cases_tbl, hosp_tbl, death_tbl], ignore_index=True)
    out = Path(__file__).resolve().parents[1] / "outputs" / f"forecast_{test_season.replace('/', '_')}.csv"
    forecast_tbl.to_csv(out, index=False)
    print(f"Forecast table: {len(forecast_tbl):,} rows -> {out}")

    # ---- seasonal summary (peak week/incidence, cumulative cases) -----------------
    seasonal = build_seasonal_summary(state_case_quantiles, weeks, QUANTILE_LEVELS, rng=rng)
    seasonal = seasonal.merge(meta[["state", "zone"]], on="state")
    out_seasonal = Path(__file__).resolve().parents[1] / "outputs" / f"seasonal_summary_{test_season.replace('/', '_')}.csv"
    seasonal.to_csv(out_seasonal, index=False)
    print(f"Seasonal summary -> {out_seasonal}")

    # ---- regional (North/South) + national aggregation via Monte Carlo ------------
    zone_map = meta.set_index("state")["zone"].to_dict()
    north_group = {s: q for s, q in state_case_quantiles.items() if zone_map[s] in NORTH_ZONES}
    south_group = {s: q for s, q in state_case_quantiles.items() if zone_map[s] not in NORTH_ZONES}
    north_q = aggregate_via_monte_carlo(north_group, QUANTILE_LEVELS, rng=rng)
    south_q = aggregate_via_monte_carlo(south_group, QUANTILE_LEVELS, rng=rng)
    national_q = aggregate_via_monte_carlo(state_case_quantiles, QUANTILE_LEVELS, rng=rng)

    true_national = national[national["season"] == test_season].sort_values("epi_week_of_season")["cases"].to_numpy()
    print(f"National aggregate median MAE vs true national cases: {mae(true_national, national_q[0.5]):,.0f}")

    regional_rows = []
    for label, qdict in [("North", north_q), ("South", south_q), ("National", national_q)]:
        for wk, wk_start in zip(weeks, week_starts):
            i = int(np.where(weeks == wk)[0][0])
            row = {"season": test_season, "region": label, "epi_week_of_season": wk, "week_start": wk_start}
            row.update({f"q{q}": qdict[q][i] for q in QUANTILE_LEVELS})
            regional_rows.append(row)
    regional_tbl = pd.DataFrame(regional_rows)
    out_regional = Path(__file__).resolve().parents[1] / "outputs" / f"regional_national_{test_season.replace('/', '_')}.csv"
    regional_tbl.to_csv(out_regional, index=False)
    print(f"Regional/national aggregate -> {out_regional}")

# ---- probabilistic-targets demonstration (parametric bootstrap), most recent split -
print(f"\n{'='*70}\nProbabilistic targets demo (parametric bootstrap): {test_season}\n{'='*70}")
d = national[national["season"] == test_season].sort_values("epi_week_of_season")
weeks = d["epi_week_of_season"].to_numpy(dtype=float)
observed = d["cases"].to_numpy(dtype=float)
population = float(d["population"].iloc[0])
params, sim, cov = fit_series_with_covariance(weeks, observed, population, kernel)
draws = bootstrap_params(params, cov, n_draws=500, rng=rng)
trajectories = bootstrap_trajectories(draws, weeks, observed, population, kernel)

wk_idx = int(np.where(weeks == 10)[0][0])  # example: "as of week 10, is transmission rising?"
p_rising = probability_of_rising(trajectories, wk_idx)
print(f"P(rising transmission), week 10 -> 11: {p_rising:.1%}")
p_peak_window = probability_peak_within(trajectories, weeks, 14, 18)
print(f"P(epidemic peak occurs within weeks 14-18): {p_peak_window:.1%}")
dist = peak_week_distribution(trajectories, weeks)
top5 = dist.sort_values(ascending=False).head(5)
print("Most likely peak weeks (bootstrap distribution):")
for wk, p in top5.items():
    print(f"  week {wk:.0f}: {p:.1%}")

print("\nDone.")
