#!/usr/bin/env bash
# Reproduces every table, figure and forecast file (run from the repository root).
set -e
python src/scoring.py
python src/check_theory.py
python src/eda.py
python src/fit_national.py
KAPPA_YEARS=4 python src/fit_national.py
TRAIN_SEASONS=2023/2024,2024/2025 python src/fit_national.py
TRAIN_SEASONS=2023/2024,2024/2025 KAPPA_YEARS=4 python src/fit_national.py
python src/report_fit.py
python src/fit_state_severity.py
python src/sensitivity.py
python src/plots_3d_R0.py
python src/plots_2d.py
python src/validate.py
python src/run_forecast.py
