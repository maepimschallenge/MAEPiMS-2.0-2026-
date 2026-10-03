"""Reference implementation of the gradient-boosted-trees (LightGBM) forecasting
pipeline for the Nigeria influenza dataset.

This is the training script of the MAEPiMS 2026 submission. It reads only the
five supplied CSV files in MAEPiMS_Challenge_Data/, trains the LightGBM +
Random Forest ensemble, runs the rolling-origin evaluation, produces the
8-week forecast and writes every table, figure and the trained model bundle
to outputs/.

Design
------
This is a "global" (single, pooled) direct multi-horizon forecasting model:
one LightGBM model per (outcome, quantile) pair is trained on a stacked table
of (state, origin week, horizon) rows drawn from every state at once, with
the forecast horizon h itself included as a feature. This is standard
practice in modern applied forecasting (e.g. the M5 competition winners;
Makridakis et al., 2022) and is far simpler to reproduce than one model per
state or one model per horizon.

Targets: cases_per_100k, hosp_per_100k, deaths_per_100k, h = 1..8 weeks ahead.
Point forecast: LightGBM regression (L2 loss).
Probabilistic forecast: nine separate LightGBM quantile-regression models
(pinball loss) per outcome, at the levels 0.025, 0.05, 0.10, 0.25, 0.50,
0.75, 0.90, 0.95, 0.975.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "MAEPiMS_Challenge_Data"
OUT_DIR = HERE / "outputs"
FIG_DIR = OUT_DIR / "figures"
EDA_DIR = OUT_DIR / "eda"
for d in (OUT_DIR, FIG_DIR, EDA_DIR):
    d.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flu_features import (
    OUTCOMES, OUTCOME_LABEL, QUANTILE_LEVELS, LAGS, HORIZON,
    rows_for_state_origin, get_feature_cols, align_categoricals,
    one_hot_categoricals, rf_feature_cols,
)

LGB_PARAMS = dict(
    num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=20260905,
    verbosity=-1,
)

# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data():
    weekly = pd.read_csv(DATA_DIR / "nigeria_flu_weekly_by_state.csv")
    metadata = pd.read_csv(DATA_DIR / "nigeria_flu_state_metadata.csv")
    weekly["season_order"] = weekly["season"].str[:4].astype(int)
    weekly = weekly.sort_values(["state", "season_order", "epi_week_of_season"]).reset_index(drop=True)
    season_len = int(weekly.groupby("season")["epi_week_of_season"].max().median())
    min_season = weekly["season_order"].min()
    weekly["global_week"] = (weekly["season_order"] - min_season) * season_len + weekly["epi_week_of_season"]
    return weekly, metadata, season_len


def build_feature_table(weekly: pd.DataFrame, season_len: int) -> pd.DataFrame:
    """One row per (state, origin global_week, horizon h): built with the exact
    same shared feature function (flu_features.rows_for_state_origin) that
    the deployed app uses at inference time, so the two can never drift apart."""
    rows = []
    for state, g in weekly.groupby("state"):
        g = g.sort_values("global_week").reset_index(drop=True)
        gw = g["global_week"].to_numpy()
        n = len(g)
        for i in range(max(LAGS), n - 1):
            origin_gw = gw[i]
            valid_h = [h for h in range(1, HORIZON + 1) if i + h < n and gw[i + h] == origin_gw + h]
            if not valid_h:
                continue
            base_rows = rows_for_state_origin(state, g, i, season_len, horizons=valid_h)
            for h, row in zip(valid_h, base_rows):
                j = i + h
                row["season"] = g["season"].iloc[i]
                row["target_season"] = g["season"].iloc[j]
                for oc in OUTCOMES:
                    row[f"target_{oc}"] = g[oc].iloc[j]
                    row[f"target_log_{oc}"] = float(np.log1p(g[oc].iloc[j]))
                rows.append(row)
    df = pd.DataFrame(rows)
    zone_categories = sorted(df["zone"].unique().tolist())
    climate_categories = sorted(df["climate"].unique().tolist())
    df = align_categoricals(df, zone_categories, climate_categories)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(train_df: pd.DataFrame, feature_cols):
    models = {}
    cat_cols = ["zone", "climate"]
    X = train_df[feature_cols]
    for oc in OUTCOMES:
        y = train_df[f"target_log_{oc}"]
        mean_model = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
        mean_model.fit(X, y, categorical_feature=cat_cols)
        models[(oc, "mean")] = mean_model
        for q in QUANTILE_LEVELS:
            qm = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS)
            qm.fit(X, y, categorical_feature=cat_cols)
            models[(oc, q)] = qm
    return models


def predict_all(models, df: pd.DataFrame, feature_cols):
    X = df[feature_cols]
    keep = [c for c in ["state", "target_week", "target_epi_week", "target_season", "horizon", "population"] if c in df.columns]
    out = df[keep].copy()
    for oc in OUTCOMES:
        out[f"{oc}_mean"] = np.clip(np.expm1(models[(oc, "mean")].predict(X)), 0, None)
        for q in QUANTILE_LEVELS:
            out[f"{oc}_q{q}"] = np.clip(np.expm1(models[(oc, q)].predict(X)), 0, None)
        # enforce monotonic quantiles (standard post-processing for independently
        # trained quantile models, which can otherwise cross)
        qcols = [f"{oc}_q{q}" for q in QUANTILE_LEVELS]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


# ---------------------------------------------------------------------------
# Random Forest component and the LightGBM + Random Forest ENSEMBLE
# ---------------------------------------------------------------------------
# The deployed model is a simple, equal-weight average of two decorrelated
# tree-ensemble families: LightGBM (boosting) and Random Forest (bagging).
# The head-to-head experiment in model_comparison.py showed this average beat
# either component alone on MAE, WIS* and CRPS for cases, hospitalisations
# and deaths. Averaging is done in RAW (original-unit) space, after each
# model's log1p predictions are back-transformed, then quantiles are sorted.

RF_PARAMS = dict(n_estimators=100, max_depth=None, min_samples_leaf=5,
                 max_features="sqrt", n_jobs=-1, random_state=20260905)


def train_rf_models(train_df: pd.DataFrame, feature_cols, zone_categories, climate_categories):
    rf_cols = rf_feature_cols(feature_cols, zone_categories, climate_categories)
    X = one_hot_categoricals(train_df, zone_categories, climate_categories)[rf_cols]
    rf_models = {}
    for oc in OUTCOMES:
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X, train_df[f"target_log_{oc}"].to_numpy())
        rf_models[oc] = rf
    return rf_models, rf_cols


def predict_rf(rf_models, df: pd.DataFrame, feature_cols, zone_categories, climate_categories):
    """Random Forest mean = forest average; quantiles approximated from the
    spread of individual trees' predictions (a simplified quantile regression
    forest; Meinshausen, 2006). Returns raw-unit predictions."""
    rf_cols = rf_feature_cols(feature_cols, zone_categories, climate_categories)
    X = one_hot_categoricals(df, zone_categories, climate_categories)[rf_cols].to_numpy()
    out = {}
    for oc in OUTCOMES:
        rf = rf_models[oc]
        out[(oc, "mean")] = np.clip(np.expm1(rf.predict(X)), 0, None)
        tree_preds = np.stack([t.predict(X) for t in rf.estimators_], axis=1)
        for q in QUANTILE_LEVELS:
            out[(oc, q)] = np.clip(np.expm1(np.quantile(tree_preds, q, axis=1)), 0, None)
    return out


def predict_ensemble(lgb_models, rf_models, df: pd.DataFrame, feature_cols, zone_categories, climate_categories):
    lgb_out = predict_all(lgb_models, df, feature_cols)
    rf_out = predict_rf(rf_models, df, feature_cols, zone_categories, climate_categories)
    out = lgb_out.copy()
    for oc in OUTCOMES:
        out[f"{oc}_mean"] = (lgb_out[f"{oc}_mean"].to_numpy() + rf_out[(oc, "mean")]) / 2
        for q in QUANTILE_LEVELS:
            out[f"{oc}_q{q}"] = (lgb_out[f"{oc}_q{q}"].to_numpy() + rf_out[(oc, q)]) / 2
        qcols = [f"{oc}_q{q}" for q in QUANTILE_LEVELS]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


# ---------------------------------------------------------------------------
# Scoring: MAE, RMSE, WIS*, CRPS approx, 90% coverage
# ---------------------------------------------------------------------------

def interval_score(y, l, u, alpha):
    return (u - l) + (2 / alpha) * np.maximum(l - y, 0) + (2 / alpha) * np.maximum(y - u, 0)


def score(pred: pd.DataFrame, truth: pd.DataFrame, oc: str) -> dict:
    m = pred.merge(truth[["state", "target_week", oc]], on=["state", "target_week"])
    y = m[oc].to_numpy()
    mean_pred = m[f"{oc}_mean"].to_numpy()
    q = {lv: m[f"{oc}_q{lv}"].to_numpy() for lv in QUANTILE_LEVELS}
    mae = mean_absolute_error(y, mean_pred)
    rmse = math.sqrt(mean_squared_error(y, mean_pred))
    is90 = interval_score(y, q[0.05], q[0.95], 0.10)
    is80 = interval_score(y, q[0.10], q[0.90], 0.20)
    is50 = interval_score(y, q[0.25], q[0.75], 0.50)
    wis = (0.5 * np.abs(y - q[0.5]) + 0.5 * is90 + 0.5 * is80 + 0.5 * is50) / 3.5
    pinball = np.mean([np.mean(np.maximum(lv * (y - q[lv]), (lv - 1) * (y - q[lv]))) for lv in QUANTILE_LEVELS])
    coverage90 = float(np.mean((y >= q[0.05]) & (y <= q[0.95])))
    return {"mae": mae, "rmse": rmse, "wis_star": float(np.mean(wis)),
            "crps_approx": float(2 * pinball), "coverage_90": coverage90, "n": len(y)}


def main():
    weekly, metadata, season_len = load_data()
    seasons = sorted(weekly["season"].unique(), key=lambda x: int(x[:4]))
    print("Seasons:", seasons, "season_len:", season_len)

    full_table = build_feature_table(weekly, season_len)
    feature_cols = get_feature_cols(full_table.columns.tolist())
    full_table.to_parquet(OUT_DIR / "feature_table.parquet") if False else None
    zone_cats = sorted(full_table["zone"].cat.categories.tolist())
    climate_cats = sorted(full_table["climate"].cat.categories.tolist())
    print("Feature table shape:", full_table.shape)
    print("Feature columns:", feature_cols)

    # ---- rolling-origin evaluation across the two most recent seasons ----
    eval_rows = []
    for test_season in seasons[1:]:
        train_df = full_table[full_table["target_season"] != test_season]
        train_df = train_df[train_df["season"].map(lambda s: seasons.index(s)) < seasons.index(test_season)]
        test_df = full_table[full_table["target_season"] == test_season]
        test_df = test_df[test_df["season"] == test_season]  # origin also within/around test season start
        if train_df.empty or test_df.empty:
            continue
        models = train_models(train_df, feature_cols)
        rf_models, _ = train_rf_models(train_df, feature_cols, zone_cats, climate_cats)
        truth = weekly.rename(columns={"global_week": "target_week"})
        candidates = {
            "Ensemble (LightGBM + RF)": predict_ensemble(models, rf_models, test_df, feature_cols, zone_cats, climate_cats),
            "LightGBM": predict_all(models, test_df, feature_cols),
        }
        for model_name, preds in candidates.items():
            for oc in OUTCOMES:
                s = score(preds, truth, oc)
                s.update({"test_season": test_season, "outcome": OUTCOME_LABEL[oc], "model": model_name})
                eval_rows.append(s)
    evaluation_all = pd.DataFrame(eval_rows)
    evaluation_all.to_csv(OUT_DIR / "evaluation_all_models.csv", index=False)
    evaluation = evaluation_all[evaluation_all["model"] == "Ensemble (LightGBM + RF)"].drop(columns="model")
    evaluation.to_csv(OUT_DIR / "evaluation.csv", index=False)
    print(evaluation_all.groupby(["model", "outcome"])[["mae", "rmse", "wis_star", "crps_approx", "coverage_90"]].mean())

    # ---- final model: train on everything, forecast 8 weeks beyond last observed week ----
    final_models = train_models(full_table, feature_cols)
    final_rf_models, rf_cols = train_rf_models(full_table, feature_cols, zone_cats, climate_cats)

    latest_season = seasons[-1]
    latest = weekly[weekly["season"] == latest_season]
    origin_gw = int(latest["global_week"].max())
    origin_epi_week = int(latest[latest["global_week"] == origin_gw]["epi_week_of_season"].iloc[0])

    # ---- serialize a self-contained deployment bundle (models + everything app.py needs) ----
    import pickle
    history_cols = ["state", "season", "season_order", "epi_week_of_season", "global_week",
                    "zone", "climate", "population"] + OUTCOMES
    recent_history = weekly[weekly["global_week"] > origin_gw - 12][history_cols].copy()
    bundle = {
        "models": final_models,
        "rf_models": final_rf_models,
        "rf_feature_cols": rf_cols,
        "rf_params": RF_PARAMS,
        "ensemble": "equal-weight average of LightGBM and Random Forest, in raw units",
        "feature_cols": feature_cols,
        "outcomes": OUTCOMES,
        "outcome_label": OUTCOME_LABEL,
        "quantile_levels": QUANTILE_LEVELS,
        "lags": LAGS,
        "horizon": HORIZON,
        "season_len": season_len,
        "zone_categories": sorted(full_table["zone"].cat.categories.tolist()),
        "climate_categories": sorted(full_table["climate"].cat.categories.tolist()),
        "metadata": metadata[["state", "zone", "climate", "population", "hc_access"]].copy(),
        "recent_history": recent_history,
        "origin_global_week": origin_gw,
        "origin_epi_week": origin_epi_week,
        "trained_on_seasons": seasons,
        "lgb_params": LGB_PARAMS,
    }
    import gzip
    # gzip-compressed pickle: the two tree-ensemble families together are large
    # (~250 MB raw); compressed it is ~75 MB. Load with pickle.load(gzip.open(path, "rb")).
    with gzip.open(OUT_DIR / "model_bundle.pkl.gz", "wb", compresslevel=6) as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model bundle: {(OUT_DIR / 'model_bundle.pkl.gz').stat().st_size / 1e6:.1f} MB")

    fc_rows = []
    for state, g in weekly.groupby("state"):
        g = g.sort_values("global_week").reset_index(drop=True)
        i = g.index[g["global_week"] == origin_gw][0]
        fc_rows.extend(rows_for_state_origin(state, g, i, season_len))
    fc_df = pd.DataFrame(fc_rows)
    fc_df = align_categoricals(fc_df, bundle["zone_categories"], bundle["climate_categories"])
    forecast = predict_ensemble(final_models, final_rf_models, fc_df, feature_cols, zone_cats, climate_cats)

    # convert per-100k forecasts to raw counts and aggregate
    for oc in OUTCOMES:
        cols = [f"{oc}_mean"] + [f"{oc}_q{q}" for q in QUANTILE_LEVELS]
        for c in cols:
            forecast[c.replace(oc, OUTCOME_LABEL[oc])] = forecast[c] * forecast["population"] / 100000.0
    forecast = forecast.merge(metadata[["state", "zone"]], on="state", how="left")
    forecast.to_csv(OUT_DIR / "forecasts_state.csv", index=False)

    numeric_cols = [c for c in forecast.columns if any(c.startswith(lbl) for lbl in OUTCOME_LABEL.values())]
    agg_pieces = [forecast.assign(geography="STATE")]
    for geo, mask in {
        "NATIONAL": np.ones(len(forecast), dtype=bool),
        "NORTH": forecast["zone"].isin(["NW", "NE", "NC"]),
        "SOUTH": forecast["zone"].isin(["SW", "SE", "SS"]),
    }.items():
        sub = forecast[mask].groupby(["target_week", "target_epi_week", "horizon"], as_index=False)[numeric_cols].sum()
        sub["state"] = geo
        sub["geography"] = geo
        agg_pieces.append(sub)
    forecast_agg = pd.concat(agg_pieces, ignore_index=True)
    forecast_agg.to_csv(OUT_DIR / "forecasts_aggregated.csv", index=False)

    # ---- Monte Carlo peak-week / rising probability from the quantile forecasts ----
    national = forecast_agg[forecast_agg["geography"] == "NATIONAL"].sort_values("target_epi_week")
    rng = np.random.default_rng(7)
    n_sims = 4000
    qlevels = np.array(QUANTILE_LEVELS)
    sims = np.zeros((n_sims, len(national)))
    for j, (_, row) in enumerate(national.iterrows()):
        qvals = np.array([row[f"cases_q{q}"] for q in QUANTILE_LEVELS])
        u = rng.uniform(0, 1, n_sims)
        sims[:, j] = np.interp(u, qlevels, qvals)
    peak_idx = sims.argmax(axis=1)
    peak_prob = pd.Series(peak_idx).value_counts(normalize=True).sort_index()
    peak_week_prob = pd.DataFrame({
        "epi_week_of_season": national["target_epi_week"].to_numpy(),
        "peak_probability": [float(peak_prob.get(i, 0.0)) for i in range(len(national))],
    })
    increasing = (sims[:, 1:] > sims[:, :-1]).mean(axis=0)
    increasing_prob = pd.DataFrame({
        "epi_week_of_season": national["target_epi_week"].to_numpy()[1:],
        "prob_increasing": increasing, "prob_decreasing": 1 - increasing,
    })
    peak_week_prob.to_csv(OUT_DIR / "peak_week_probability.csv", index=False)
    increasing_prob.to_csv(OUT_DIR / "increasing_probability.csv", index=False)

    # ---- feature importances (gain) for the cases-mean model, for interpretability ----
    importances = pd.Series(final_models[("cases_per_100k", "mean")].feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(OUT_DIR / "feature_importance_cases.csv", header=["gain_importance"])

    # ---- EDA ----
    season_profile = weekly.groupby("season").agg(
        states=("state", "nunique"), weeks=("epi_week_of_season", "nunique"),
        total_cases=("cases", "sum"), total_hospitalizations=("hospitalizations", "sum"),
        total_deaths=("deaths", "sum"), mean_cases_per_100k=("cases_per_100k", "mean")).reset_index()
    season_profile.to_csv(EDA_DIR / "season_profile.csv", index=False)

    regional = weekly.merge(metadata[["state", "hc_access"]], on="state", how="left")
    regional_summary = regional.groupby(["season", "zone"], as_index=False).agg(
        total_cases=("cases", "sum"), mean_incidence=("cases_per_100k", "mean"),
        mean_healthcare_access=("hc_access", "mean"))
    regional_summary.to_csv(EDA_DIR / "regional_summary.csv", index=False)

    corr = regional[["cases_per_100k", "hosp_per_100k", "deaths_per_100k", "hc_access"]].corr(numeric_only=True)
    corr.to_csv(EDA_DIR / "exploratory_correlations.csv")

    # ---- figures ----
    national_obs = weekly.groupby(["season", "epi_week_of_season"], as_index=False)[["cases", "hospitalizations", "deaths"]].sum()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, outcome in zip(axes, ["cases", "hospitalizations", "deaths"]):
        for season, g in national_obs.groupby("season"):
            ax.plot(g["epi_week_of_season"], g[outcome], label=season, linewidth=2)
        ax.set_ylabel(outcome.title())
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Epidemiological week of season")
    fig.suptitle("Observed national influenza burden by season")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "national_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importances.head(15).iloc[::-1]
    ax.barh(top.index, top.values, color="#2e6f95")
    ax.set(title="Top 15 features by gain (cases, LightGBM component of the ensemble)", xlabel="LightGBM gain importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)

    latest_fc = national.copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(latest_fc["target_epi_week"], latest_fc["cases_mean"], color="#153b50", label="Point forecast")
    ax.fill_between(latest_fc["target_epi_week"], latest_fc["cases_q0.05"], latest_fc["cases_q0.95"],
                     color="#f28e5b", alpha=0.25, label="90% interval")
    ax.fill_between(latest_fc["target_epi_week"], latest_fc["cases_q0.25"], latest_fc["cases_q0.75"],
                     color="#f28e5b", alpha=0.45, label="50% interval")
    ax.set(title="National weekly case forecast with probabilistic intervals (LightGBM + RF ensemble)",
           xlabel="Epidemiological week", ylabel="Cases")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "national_forecast.png", dpi=180)
    plt.close(fig)

    latest_hw = weekly[weekly["season"] == latest_season]
    heat = latest_hw.pivot_table(index="state", columns="epi_week_of_season", values="cases_per_100k", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(np.log1p(heat.to_numpy()), aspect="auto", cmap="YlOrRd")
    ax.set(title="State incidence heatmap: latest season (log1p scale)", xlabel="Epidemiological week", ylabel="State")
    ax.set_yticks(np.arange(len(heat.index)), heat.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="log(1+cases per 100k)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "state_incidence_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(peak_week_prob["epi_week_of_season"], peak_week_prob["peak_probability"], color="#2e6f95")
    ax.set(title="Monte Carlo probability that a given week holds the season peak",
           xlabel="Epidemiological week", ylabel="P(peak week)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "peak_week_probability.png", dpi=180)
    plt.close(fig)

    manifest = {
        "seasons": seasons, "season_length": season_len, "latest_season": latest_season,
        "forecast_origin_epi_week": origin_epi_week, "forecast_horizon": HORIZON,
        "n_feature_rows": int(len(full_table)), "n_features": len(feature_cols),
        "lgb_params": LGB_PARAMS, "rf_params": RF_PARAMS, "deployed_model": "Ensemble (LightGBM + RF)",
        "evaluation_summary": evaluation.groupby("outcome")[["mae", "rmse", "wis_star", "crps_approx", "coverage_90"]]
            .mean().round(3).to_dict(orient="index"),
        "top_features_cases": importances.head(10).round(1).to_dict(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
