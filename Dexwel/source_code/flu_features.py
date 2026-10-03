"""Shared feature-engineering logic for the Nigeria influenza LightGBM forecaster.

This module is imported by BOTH the training script (ml_reference.py / the R
port's Python-equivalent logic) and the deployed app (app.py), so that the
features an app builds at inference time can never silently drift out of sync
with the features the model was trained on. An earlier draft of this project
computed the seasonal week_sin/week_cos feature differently at training time
(from the forecast *origin* week) than at inference time (from the forecast
*target* week); the mismatch caused multi-week-ahead forecasts to explode to
implausible values even though the underlying model was fine. Centralising
the feature logic here is the fix, and the reason it must not be duplicated.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

OUTCOMES = ["cases_per_100k", "hosp_per_100k", "deaths_per_100k"]
OUTCOME_LABEL = {"cases_per_100k": "cases", "hosp_per_100k": "hospitalizations", "deaths_per_100k": "deaths"}
QUANTILE_LEVELS = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
LAGS = [1, 2, 3, 4]
HORIZON = 8


def static_origin_features(state: str, zone: str, climate: str, population: float,
                            epi_week_of_season: int, season_len: int) -> Dict:
    """Features known at the forecast origin that do not depend on horizon."""
    origin_angle = 2 * math.pi * epi_week_of_season / season_len
    return {
        "state": state, "zone": zone, "climate": climate,
        "population": population, "log_population": float(np.log(population)),
        "epi_week_of_season": epi_week_of_season,
        "origin_week_sin": math.sin(origin_angle),
        "origin_week_cos": math.cos(origin_angle),
    }


def lag_rolling_features(outcome_history: Dict[str, np.ndarray], i: int) -> Dict:
    """outcome_history: {outcome_name: full array of that outcome for one state,
    sorted by time}. i: index of the origin week within that array."""
    feat = {}
    for oc, series in outcome_history.items():
        log_series = np.log1p(series)
        for lag in LAGS:
            feat[f"{oc}_lag{lag}"] = log_series[i - lag + 1]
        roll = log_series[i - 3:i + 1]
        feat[f"{oc}_roll_mean4"] = float(np.mean(roll))
        feat[f"{oc}_roll_std4"] = float(np.std(roll))
        denom = max(series[i - 2], 1e-3)
        feat[f"{oc}_growth_ratio"] = float(np.clip(series[i] / denom, 0.1, 10.0))
    return feat


def horizon_row(base_feat: Dict, origin_epi_week: int, origin_global_week: int,
                 h: int, season_len: int) -> Dict:
    """Expand the origin's static+lag features into one row for horizon h,
    with the TARGET week's (not the origin week's) seasonal sin/cos — this is
    the feature that must be computed identically in training and inference."""
    row = dict(base_feat)
    target_epi_week = origin_epi_week + h
    wrapped_week = ((target_epi_week - 1) % season_len) + 1
    target_angle = 2 * math.pi * wrapped_week / season_len
    row["week_sin"] = math.sin(target_angle)
    row["week_cos"] = math.cos(target_angle)
    row["horizon"] = h
    row["target_week"] = origin_global_week + h
    row["target_epi_week"] = target_epi_week
    return row


def get_feature_cols(columns: List[str]) -> List[str]:
    exclude = {"state", "season", "origin_global_week", "epi_week_of_season", "target_week",
               "target_epi_week", "target_season", "population"} \
              | {f"target_{oc}" for oc in OUTCOMES} | {f"target_log_{oc}" for oc in OUTCOMES}
    return [c for c in columns if c not in exclude]


def rows_for_state_origin(state: str, g: pd.DataFrame, i: int, season_len: int,
                           horizons: List[int] | None = None) -> List[Dict]:
    """g: one state's weekly rows, sorted by global_week, index reset 0..n-1.
    i: integer position of the forecast origin within g. Returns one dict per
    horizon in `horizons` (default 1..HORIZON), ready to become a model input row
    (after wrapping zone/climate as pandas categoricals matching the training
    categories -- see align_categoricals below)."""
    horizons = horizons or list(range(1, HORIZON + 1))
    base = static_origin_features(
        state=state, zone=g["zone"].iloc[i], climate=g["climate"].iloc[i],
        population=g["population"].iloc[i], epi_week_of_season=int(g["epi_week_of_season"].iloc[i]),
        season_len=season_len,
    )
    outcome_history = {oc: g[oc].to_numpy() for oc in OUTCOMES}
    base.update(lag_rolling_features(outcome_history, i))
    origin_gw = int(g["global_week"].iloc[i])
    origin_epi_week = int(g["epi_week_of_season"].iloc[i])
    return [horizon_row(base, origin_epi_week, origin_gw, h, season_len) for h in horizons]


def align_categoricals(df: pd.DataFrame, zone_categories: List[str], climate_categories: List[str]) -> pd.DataFrame:
    """Cast zone/climate to pandas Categorical with the EXACT category set (and
    order) used at training time. Skipping this step is the single most
    dangerous mistake when serving a LightGBM model that has categorical
    features: a categorical re-derived only from the (small) inference batch
    can silently assign different integer codes to the same category labels,
    corrupting every prediction without raising an error."""
    df = df.copy()
    df["zone"] = pd.Categorical(df["zone"], categories=zone_categories)
    df["climate"] = pd.Categorical(df["climate"], categories=climate_categories)
    return df


def one_hot_categoricals(df: pd.DataFrame, zone_categories: List[str], climate_categories: List[str]) -> pd.DataFrame:
    """One-hot encode zone/climate for models (Random Forest, etc.) that do not
    take native categorical features, using a FIXED category list so the set
    of dummy columns produced is identical whether df has 43,000 rows or 1 --
    the same discipline as align_categoricals, and for the same reason."""
    zone_dum = pd.get_dummies(pd.Categorical(df["zone"], categories=zone_categories), prefix="zone")
    climate_dum = pd.get_dummies(pd.Categorical(df["climate"], categories=climate_categories), prefix="climate")
    out = pd.concat([df.reset_index(drop=True), zone_dum.reset_index(drop=True), climate_dum.reset_index(drop=True)], axis=1)
    return out


def rf_feature_cols(feature_cols: List[str], zone_categories: List[str], climate_categories: List[str]) -> List[str]:
    numeric = [c for c in feature_cols if c not in ("zone", "climate")]
    return numeric + [f"zone_{z}" for z in zone_categories] + [f"climate_{c}" for c in climate_categories]
