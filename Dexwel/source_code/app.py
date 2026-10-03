"""Nigeria influenza forecaster -- Streamlit app (free on Streamlit Community Cloud).

LightGBM + Random Forest ensemble, up to 8-week probabilistic forecasts of
cases, hospitalizations and deaths per state, per zone, or nationally.
Forecasting code is shared with ml_reference.py via flu_features.py.

Run locally (from this folder):  pip install -r requirements.txt && streamlit run app.py
The model is outputs/model_bundle.pkl.gz (shipped; re-created by `python ml_reference.py`).
"""
from __future__ import annotations

import gzip
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from flu_features import align_categoricals, one_hot_categoricals, rows_for_state_origin

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BUNDLE_PATH = Path(__file__).resolve().parent / "outputs" / "model_bundle.pkl.gz"


@st.cache_resource(show_spinner="Loading model (first visit only)...")
def load_bundle(path: str = str(BUNDLE_PATH)) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def _state_history(bundle: dict, state: str, extra_rows: Optional[List[dict]] = None) -> pd.DataFrame:
    hist = bundle["recent_history"]
    g = hist[hist["state"] == state].sort_values("global_week").reset_index(drop=True)
    if g.empty:
        raise ValueError(f"Unknown state: {state!r}. See GET /states for valid names.")
    if extra_rows:
        last_gw = int(g["global_week"].iloc[-1])
        static = g.iloc[-1][["state", "zone", "climate", "population"]].to_dict()
        new = []
        for k, row in enumerate(extra_rows, start=1):
            new.append({**static, "global_week": last_gw + k,
                        "season": row.get("season", g["season"].iloc[-1]),
                        "epi_week_of_season": row["epi_week_of_season"],
                        "cases_per_100k": row["cases_per_100k"],
                        "hosp_per_100k": row["hosp_per_100k"],
                        "deaths_per_100k": row["deaths_per_100k"]})
        g = pd.concat([g, pd.DataFrame(new)], ignore_index=True)
    return g


def forecast_state(state: str, bundle: dict, horizon: int = 8,
                    extra_rows: Optional[List[dict]] = None) -> pd.DataFrame:
    """Return a DataFrame with one row per forecast week, mean + quantile
    forecasts for cases, hospitalisations and deaths, in raw counts."""
    g = _state_history(bundle, state, extra_rows)
    season_len = bundle["season_len"]
    i = len(g) - 1  # forecast from the most recent available week
    horizons = list(range(1, horizon + 1))
    rows = rows_for_state_origin(state, g, i, season_len, horizons=horizons)
    fc_df = pd.DataFrame(rows)
    fc_df = align_categoricals(fc_df, bundle["zone_categories"], bundle["climate_categories"])

    X = fc_df[bundle["feature_cols"]]
    X_rf = one_hot_categoricals(fc_df, bundle["zone_categories"], bundle["climate_categories"])[
        bundle["rf_feature_cols"]].to_numpy()
    out = fc_df[["target_week", "target_epi_week", "horizon"]].copy()
    population = float(g["population"].iloc[-1])
    for oc in bundle["outcomes"]:
        label = bundle["outcome_label"][oc]
        rf = bundle["rf_models"][oc]
        tree_preds = np.stack([t.predict(X_rf) for t in rf.estimators_], axis=1)

        # ENSEMBLE = equal-weight average, in raw units, of LightGBM and Random Forest
        lgb_mean = np.clip(np.expm1(bundle["models"][(oc, "mean")].predict(X)), 0, None)
        rf_mean = np.clip(np.expm1(rf.predict(X_rf)), 0, None)
        out[f"{label}_mean"] = (lgb_mean + rf_mean) / 2 * population / 100000.0
        qcols = []
        for q in bundle["quantile_levels"]:
            lgb_q = np.clip(np.expm1(bundle["models"][(oc, q)].predict(X)), 0, None)
            rf_q = np.clip(np.expm1(np.quantile(tree_preds, q, axis=1)), 0, None)
            col = f"{label}_q{q}"
            out[col] = (lgb_q + rf_q) / 2 * population / 100000.0
            qcols.append(col)
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)  # enforce monotone quantiles
    out.insert(0, "state", state)
    return out


def forecast_group(states: List[str], bundle: dict, horizon: int = 8) -> pd.DataFrame:
    """Sum per-state forecasts (mean and each quantile independently) to get a
    zone- or national-level forecast. See the paper's Discussion for why
    summing quantiles is a conservative approximation, not a joint simulation."""
    parts = [forecast_state(s, bundle, horizon) for s in states]
    combined = pd.concat(parts, ignore_index=True)
    value_cols = [c for c in combined.columns if c not in ("state", "target_week", "target_epi_week", "horizon")]
    return combined.groupby(["target_week", "target_epi_week", "horizon"], as_index=False)[value_cols].sum()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
OUTCOMES_UI = {"Cases": "cases", "Hospitalizations": "hospitalizations", "Deaths": "deaths"}
NEW_COLS = ["epi_week_of_season", "season", "cases_per_100k", "hosp_per_100k", "deaths_per_100k"]


def fan_chart(df: pd.DataFrame, label: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = df["horizon"].to_numpy()
    for lo, hi, alpha, name in [(0.025, 0.975, 0.15, "95% interval"), (0.05, 0.95, 0.20, "90% interval"),
                                (0.25, 0.75, 0.35, "50% interval")]:
        lo_c, hi_c = f"{label}_q{lo}", f"{label}_q{hi}"
        if lo_c in df and hi_c in df:
            ax.fill_between(x, df[lo_c], df[hi_c], alpha=alpha, color="tab:blue", label=name, linewidth=0)
    if f"{label}_q0.5" in df:
        ax.plot(x, df[f"{label}_q0.5"], color="tab:blue", lw=1.5, ls="--", label="median")
    ax.plot(x, df[f"{label}_mean"], color="navy", lw=2.2, marker="o", label="ensemble mean")
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{h}w\n(wk {w})" for h, w in zip(df["horizon"], df["target_epi_week"])], fontsize=8)
    ax.set_ylabel(f"Weekly {label} (count)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Computing forecast...")
def cached_forecast(scope: str, choice: str, horizon: int, extra_json: Optional[str]) -> pd.DataFrame:
    b = load_bundle()
    if scope == "State":
        extra = json.loads(extra_json) if extra_json else None
        return forecast_state(choice, b, horizon, extra_rows=extra)
    if scope == "Zone":
        zs = b["metadata"].loc[b["metadata"]["zone"] == choice, "state"].tolist()
        return forecast_group(zs, b, horizon)
    return forecast_group(b["metadata"]["state"].tolist(), b, horizon)


def parse_extra(edited: pd.DataFrame) -> Optional[str]:
    rows = edited.dropna(subset=["epi_week_of_season", "cases_per_100k", "hosp_per_100k", "deaths_per_100k"])
    if rows.empty:
        return None
    out = []
    for _, r in rows.iterrows():
        d = {"epi_week_of_season": int(r["epi_week_of_season"]),
             "cases_per_100k": float(r["cases_per_100k"]),
             "hosp_per_100k": float(r["hosp_per_100k"]),
             "deaths_per_100k": float(r["deaths_per_100k"])}
        if isinstance(r.get("season"), str) and r["season"].strip():
            d["season"] = r["season"].strip()
        out.append(d)
    return json.dumps(out)


def main():
    st.set_page_config(page_title="Nigeria Influenza Forecaster", page_icon="🦠", layout="wide")
    if not BUNDLE_PATH.exists():
        st.error("Model file not found: outputs/model_bundle.pkl.gz. Run `python ml_reference.py` first to create it.")
        st.stop()
    b = load_bundle()
    state_list = sorted(b["metadata"]["state"].unique().tolist())
    zone_list = sorted(b["metadata"]["zone"].unique().tolist())

    st.title("Nigeria influenza forecaster")
    st.caption(
        f"LightGBM + Random Forest ensemble. Trained on seasons {', '.join(b['trained_on_seasons'])}; "
        f"forecasts start after epi week {b['origin_epi_week']}. Zone and national forecasts sum per-state "
        "quantiles (a conservative approximation, not a joint simulation).")

    with st.sidebar:
        st.header("Forecast settings")
        scope = st.radio("Level", ["State", "Zone", "National"])
        if scope == "State":
            choice = st.selectbox("State", state_list, index=state_list.index("Lagos") if "Lagos" in state_list else 0)
        elif scope == "Zone":
            choice = st.selectbox("Zone", zone_list)
        else:
            choice = "Nigeria"
        horizon = st.slider("Horizon (weeks)", 1, 8, 8)

    extra_json = None
    if scope == "State":
        with st.expander("Optional: add newer weekly observations (rates per 100k)"):
            st.caption("Rows in chronological order. Does not retrain the model; it only moves the forecast's starting point.")
            edited = st.data_editor(
                pd.DataFrame({"epi_week_of_season": pd.Series(dtype="float"), "season": pd.Series(dtype="str"),
                              "cases_per_100k": pd.Series(dtype="float"), "hosp_per_100k": pd.Series(dtype="float"),
                              "deaths_per_100k": pd.Series(dtype="float")}),
                num_rows="dynamic", key="extra")
            extra_json = parse_extra(edited)

    try:
        df = cached_forecast(scope, choice, int(horizon), extra_json)
    except Exception as e:
        st.error(f"Could not compute forecast: {e}")
        return

    st.subheader(f"{choice}: {horizon}-week forecast" if scope != "National" else f"Nigeria: {horizon}-week forecast")
    tabs = st.tabs(list(OUTCOMES_UI))
    for tab, (name, label) in zip(tabs, OUTCOMES_UI.items()):
        with tab:
            peak = df.loc[df[f"{label}_mean"].idxmax()]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Peak weekly {label} (mean)", f"{peak[f'{label}_mean']:,.0f}")
            c2.metric("Peak week", f"epi wk {int(peak['target_epi_week'])} (+{int(peak['horizon'])}w)")
            c3.metric(f"Total {label}, {horizon} wk", f"{df[f'{label}_mean'].sum():,.0f}")
            fig = fan_chart(df, label, f"{choice}: {name.lower()} forecast")
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Forecast table (counts)")
    table = df.drop(columns=["state"], errors="ignore").round(2)
    st.dataframe(table, hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False).encode(),
                       file_name=f"forecast_{choice.replace(' ', '_')}.csv", mime="text/csv")


main()
