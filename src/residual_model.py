"""
Residual quantile ML correction - Component 3 of the CP-DWRE. See docs/DESIGN.md.

Fits a LightGBM quantile regressor (one model per level in QUANTILE_LEVELS) on the
*log-ratio* residual between observed weekly state cases and the calibration layer's
baseline prediction (Component 2's population-proportional + zone-phase-blended
allocation): `target = log((observed+1)/(baseline+1))`. Log-ratio rather than raw
residual for two reasons: (1) it keeps reconstructed forecasts positive by
construction (`prediction = baseline * exp(quantile)`), and (2) it's the natural
scale for the per-state *magnitude* correction the calibration layer deliberately
left out (see src/calibration.py's postmortems) - a state whose true attack rate
runs consistently above or below the national average shows up as a roughly
constant log-ratio offset, which is exactly the kind of nonlinear, state-specific
pattern a tree ensemble can pick up from static metadata (population, hc_access,
zone, climate) without needing to touch the renewal equation's own dynamics (and
without that layer's magnitude/width coupling problems).

Given this challenge's lesson from Component 2 - a result validated on one split can
fail badly on another - this layer is meant to be trained and evaluated the same
leave-one-season-out way; `scripts/run_residual_model.py` does that, including
building each training row's baseline from zone offsets fit on the *other* training
season only (not the season the row itself belongs to), so the residual target isn't
contaminated by information about its own season's true pattern.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
CATEGORICAL_COLUMNS = ["zone", "climate"]
FEATURE_COLUMNS = [
    "log_population", "hc_access", "zone", "climate",
    "epi_week_of_season", "weeks_since_t1", "weeks_since_t2", "log_baseline",
]

_DEFAULT_LGBM_PARAMS = dict(
    n_estimators=150,
    num_leaves=7,           # small: ~3,800 training rows for 2 seasons x 37 states x 52 weeks
    min_child_samples=40,   # conservative given the dataset size - avoid overfitting to noise
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
)


def build_feature_target_table(
    by_state_season: pd.DataFrame,
    baseline_alloc: pd.DataFrame,
    meta: pd.DataFrame,
    national_t1: float,
    national_t2: float,
    season: str,
) -> pd.DataFrame:
    """One row per (state, week) for a single season: features + log-ratio target.

    `baseline_alloc`: the calibration layer's per-state weekly trajectories (columns =
    state names, as returned by src.calibration.allocate_national_to_states /
    allocate_with_phase_shift), aligned to `by_state_season`'s week ordering.
    `national_t1`/`national_t2`: that season's national renewal fit's wave-center
    weeks, used to build the "weeks since wave center" position features.
    """
    meta_idx = meta.set_index("state")
    rows = []
    for state, d in by_state_season.groupby("state"):
        d = d.sort_values("epi_week_of_season")
        weeks = d["epi_week_of_season"].to_numpy(dtype=float)
        observed = d["cases"].to_numpy(dtype=float)
        baseline = baseline_alloc[state].to_numpy()[: len(weeks)]
        m = meta_idx.loc[state]
        for wk, obs, base in zip(weeks, observed, baseline):
            rows.append({
                "season": season, "state": state, "epi_week_of_season": wk,
                "log_population": np.log(m["population"]), "hc_access": m["hc_access"],
                "zone": m["zone"], "climate": m["climate"],
                "weeks_since_t1": wk - national_t1, "weeks_since_t2": wk - national_t2,
                "log_baseline": np.log(base + 1), "baseline": base, "observed": obs,
                "log_ratio": np.log((obs + 1) / (base + 1)),
            })
    df = pd.DataFrame(rows)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def fit_quantile_models(
    train_df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
    quantile_levels: list[float] = QUANTILE_LEVELS,
    lgbm_params: dict | None = None,
) -> dict[float, lgb.LGBMRegressor]:
    """One LightGBM quantile regressor per level, predicting `log_ratio`."""
    params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
    X = train_df[feature_columns]
    y = train_df["log_ratio"]
    models = {}
    for q in quantile_levels:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        model.fit(X, y, categorical_feature=categorical_columns)
        models[q] = model
    return models


def predict_log_ratio_quantiles(
    models: dict[float, lgb.LGBMRegressor], feature_df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS
) -> dict[float, np.ndarray]:
    X = feature_df[feature_columns]
    return {q: model.predict(X) for q, model in models.items()}


def enforce_monotonic_quantiles(quantile_preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Independent per-quantile models can cross; sort pointwise so q=0.05's prediction
    never exceeds q=0.5's, etc. (standard quantile-crossing fix)."""
    levels = sorted(quantile_preds)
    stacked = np.sort(np.stack([quantile_preds[q] for q in levels], axis=0), axis=0)
    return {q: stacked[i] for i, q in enumerate(levels)}


DEFAULT_INTERVAL_INFLATION = 1.25  # see inflate_quantiles docstring - set from LOSO evidence


def inflate_quantiles(count_quantiles: dict[float, np.ndarray], factor: float) -> dict[float, np.ndarray]:
    """Scale each quantile's distance from the median by `factor`, median unchanged.

    Same lesson as src/calibration.py's `phase_weight`: a model trained on 2 of 3
    seasons is naturally calibrated to *those* seasons' residual spread, and
    leave-one-season-out testing (`scripts/run_residual_model.py`) shows that's
    overconfident whenever the held-out season is the atypical one - on 2024/25 at
    inflation=1.0, the nominal 90% interval covered only ~45% of true values (vs
    ~87-89% on the other 2 splits).

    A sweep of this factor from 1.0 to 4.0 across all 3 LOSO splits found 1.25 is a
    genuine Pareto improvement over no adjustment at all: mean WIS across the 3 splits
    is *lower* at 1.25 than at 1.0 (1,239 vs 1,243) while mean 90%-band coverage rises
    from 73% to 80% - 2024/25's WIS specifically improves (2,047 -> 2,012) because the
    interval-score penalty for missing a too-narrow band outweighs the small width
    penalty from widening it slightly. Factors above ~1.5 trade this away: coverage on
    2024/25 keeps climbing but only slowly (plateaus in the high-60s% even at 4.0,
    since the underlying bias is in the *median*, not just the width) while WIS on the
    2 typical splits gets steadily worse from the unnecessary extra width. 1.25 is the
    best empirically-grounded compromise given only 3 seasons to validate against.
    """
    if factor == 1.0:
        return count_quantiles
    median = count_quantiles[0.5]
    return {q: np.maximum(median + factor * (v - median), 0) for q, v in count_quantiles.items()}


def reconstruct_quantile_forecast(
    baseline: np.ndarray,
    log_ratio_quantiles: dict[float, np.ndarray],
    interval_inflation: float = DEFAULT_INTERVAL_INFLATION,
) -> dict[float, np.ndarray]:
    """`baseline * exp(quantile)` per level, with monotonicity enforced first so the
    reconstructed case-count quantiles are guaranteed non-crossing, then optionally
    widened via `inflate_quantiles` (see its docstring for why the default isn't 1.0
    once `scripts/run_residual_model.py`'s sweep has set `DEFAULT_INTERVAL_INFLATION`)."""
    log_ratio_quantiles = enforce_monotonic_quantiles(log_ratio_quantiles)
    count_quantiles = {q: np.maximum(baseline * np.exp(pred), 0) for q, pred in log_ratio_quantiles.items()}
    return inflate_quantiles(count_quantiles, interval_inflation)
