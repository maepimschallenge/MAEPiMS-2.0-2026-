# MAEPiMS Challenge 2026: Nigeria Influenza Forecasting

Solution for the [MAEPiMS Challenge 2026](_MAEPiMS_Challenge_2026.pdf) (The Adejimi Adeniji
Foundation): forecast weekly influenza cases, hospitalisations and deaths across Nigeria's 36
states + FCT, using only the supplied synthetic surveillance dataset.

Submission deadline: **5 October 2026, midnight WAT**.
Submitted by: **Pinnk**
Repository: https://github.com/maepimschallenge/MAEPiMS-2.0-2026-

## Model: Climate-Phased Dual-Wave Renewal Ensemble (CP-DWRE)

An original hybrid: a covariate-forced renewal-equation core (captures the two-wave wet-season /
Harmattan dynamics) + hierarchical state calibration (zone/climate/healthcare-access effects) +
a residual quantile-ML correction layer, combined into a bootstrapped probabilistic ensemble.
Full rationale and architecture: **[docs/DESIGN.md](docs/DESIGN.md)**.

## Repo layout

```
MAEPiMS_Challenge_Data/   supplied dataset (do not modify)
eda/                      exploratory analysis (explore.py + figures/)
scripts/                  one-off diagnostic / smoke-test scripts
src/                      the model pipeline (importable package)
  data.py                   load & join the challenge CSVs
  metrics.py                MAE, RMSE, WIS, CRPS, quantile loss, peak accuracy, calibration
  renewal.py                Component 1: mechanistic renewal core [implemented]
  calibration.py            Component 2: hierarchical state calibration [implemented v1]
  residual_model.py         Component 3: residual quantile ML correction [implemented v1]
  ensemble.py                Component 4: probabilistic ensemble + submission builder [implemented v1]
tests/                    pytest unit tests for src/
docs/DESIGN.md            architecture + rationale (read this first)
report/                   technical report draft (submission deliverable #1)
outputs/                  generated forecast CSVs (submission deliverable #2, gitignored)
models/                   saved fitted model artifacts (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat for cmd.exe
pip install -r requirements.txt
```

## Running things

```bash
# exploratory analysis -> eda/figures/
python eda/explore.py

# renewal-core fit sanity check -> eda/figures/05_renewal_fit_smoke_test.png
python scripts/smoke_test_renewal.py

# calibration layer: train on 2023/24+2024/25, evaluate on held-out 2025/26
# -> eda/figures/06_calibration_holdout.png, outputs/calibration_eval_2025_26.csv
python scripts/run_calibration.py

# calibration robustness check: all 3 leave-one-season-out train/test splits
# -> outputs/calibration_loso_summary.csv
python scripts/run_calibration_loso.py

# residual quantile ML layer: LightGBM log-ratio correction + interval inflation,
# evaluated leave-one-season-out -> outputs/residual_model_loso_summary.csv
python scripts/run_residual_model.py

# full pipeline -> outputs/forecast_<season>.csv, seasonal_summary_<season>.csv,
# regional_national_<season>.csv (all 3 seasons, leave-one-season-out) + a
# probabilistic-targets (P(rising), P(peak within k weeks)) demo
python scripts/run_ensemble.py

# unit tests
python -m pytest tests/ -q
```

## Status

All 4 components are implemented and tested (24 unit tests) - see the checklist at the
bottom of [docs/DESIGN.md](docs/DESIGN.md) for what's built vs still open (a proper
rolling-origin backtest harness and the technical report itself). Every headline number is
*leave-one-season-out validated*, not cherry-picked from a single split, after an early
single-split calibration result (81% -> 97% peak-week accuracy) turned out not to
generalize - see [docs/DESIGN.md](docs/DESIGN.md) for that correction and the residual
layer's parallel finding (raw intervals were overconfident on the atypical 2024/25 season,
fixed with an evidence-based, genuinely Pareto-improving interval-widening factor). The
ensemble layer produces the actual deliverable tables (`outputs/forecast_<season>.csv` etc.)
plus the optional probabilistic targets via a parametric bootstrap, and documents its own
simplification (Monte-Carlo aggregation across states/weeks assumes independence, which
likely understates true aggregate uncertainty) rather than presenting it as exact.

## Rules note

Per the challenge rules, only the supplied simulated dataset (`MAEPiMS_Challenge_Data/`) is used
for model development and evaluation - no real-world influenza data.
