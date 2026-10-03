# Nigeria Influenza Forecaster: LightGBM + Random Forest Ensemble

**MAEPiMS Challenge 2026, Nigeria Influenza Forecasting Challenge**
**Author:** Dexwel

An original machine-learning model that forecasts weekly influenza **cases, hospitalisations and deaths**
for all 36 states and the FCT (and North / South / national totals), **1 to 8 weeks ahead**, with
quantile (probabilistic) output. The model is an equal-weight ensemble of LightGBM and Random Forest.
Only the five supplied CSV files are used.

## 1. Submission checklist (what the brief asked for, and where it is)

| Required by the brief | Location |
|---|---|
| Technical Report (single PDF) | `technical_report/MAEPiMS_2026_Technical_Report.pdf` (Word copy alongside) |
| Forecast dataset (CSV) | `forecast_dataset/forecasts_aggregated.csv` (national, North, South, zones and states; 8-week horizon). Also `forecasts_state.csv`, `peak_week_probability.csv`, `increasing_probability.csv` |
| Source code | `source_code/` |
| Documentation | this README, `documentation/` (evaluation tables, figures, manifest) and the report |

## 2. Folder layout

```
README.md
technical_report/    the report (PDF)
forecast_dataset/    the forecast CSVs
documentation/       evaluation results, model comparison, figures, EDA tables, manifest
source_code/
    ml_reference.py        trains the model, evaluates it, forecasts, writes every output
    model_comparison.py    the 5-model comparison experiment (report Table 4)
    flu_features.py        feature engineering shared by training and the app
    app.py                 Streamlit web app (optional demo)
    requirements.txt
    MAEPiMS_Challenge_Data/   the 5 supplied CSV files (the only data used)
    outputs/model_bundle.pkl.gz   the trained model (re-created by ml_reference.py)
```

## 3. Reproduce the results

Needs Python 3.10+ (developed on 3.12).

```bash
cd source_code
pip install -r requirements.txt
```

**Step A: train, evaluate and forecast (about 5 minutes on one CPU core)**

```bash
python ml_reference.py
```

Reads only `MAEPiMS_Challenge_Data/`, and writes to `source_code/outputs/`: the trained model
(`model_bundle.pkl.gz`), `evaluation.csv`, `evaluation_all_models.csv`, the forecast CSVs, `manifest.json`
and the report figures. The copies in `forecast_dataset/` and `documentation/` are the submitted snapshot of these.

How to check you reproduced the report: `outputs/evaluation_all_models.csv`, averaged over the two test
seasons, should give for **cases** (report Table 3):

| Model | MAE | WIS* | 90% coverage |
|---|---|---|---|
| Ensemble (LightGBM + RF), the submitted model | 149.0 | 491.3 | 0.504 |
| LightGBM alone | 148.6 | 542.7 | 0.437 |

The forecast CSVs regenerate identically to the ones in `forecast_dataset/`.

**Step B: 5-model comparison (slower; scikit-learn Gradient Boosting takes about 2 minutes per outcome)**

```bash
python model_comparison.py
```

Writes `outputs/model_comparison.csv` (report Table 4).

All random seeds are fixed (`20260905`).

## 4. Run the web app (optional)

```bash
cd source_code
streamlit run app.py
```

Pick a state, zone or all of Nigeria, choose 1 to 8 weeks, and see fan charts (50/90/95% intervals) for
cases, hospitalisations and deaths, with a downloadable table. The app loads `outputs/model_bundle.pkl.gz`.

**Free hosting (Streamlit Community Cloud):** push this repository to GitHub, go to
https://share.streamlit.io, click *Create app*, choose this repository, set the main file path to
`source_code/app.py`, and under *Advanced settings* choose Python 3.12. Streamlit installs
`source_code/requirements.txt` automatically.

## 5. Where each result in the report comes from

| In the report | File |
|---|---|
| Tables 1 and 2 (descriptive / regional summaries) | `documentation/eda/*.csv` |
| Figures 1 and 2 (national curves, feature importance) | `documentation/figures/` |
| Table 3 (rolling-origin evaluation) | `documentation/evaluation.csv`, `documentation/evaluation_all_models.csv` |
| Table 4 (5-model comparison) | `documentation/model_comparison.csv` |
| Table 5 and Figure 3 (8-week forecast) | `forecast_dataset/forecasts_aggregated.csv`, `documentation/figures/national_forecast.png` |
| Figures 4 and 5 (state heatmap, peak-week probability) | `documentation/figures/` |

## 6. Libraries (cited in the report)

pandas, NumPy, scikit-learn, LightGBM, Matplotlib; Streamlit for the optional demo app.
