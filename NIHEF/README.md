# NIHEF: Nigeria Influenza Hierarchical Ensemble Forecasting

### MAEPiMS Challenge 2026 Technical Submission

NIHEF is a probabilistic forecasting framework developed for seasonal influenza forecasting in Nigeria.

The framework combines seasonal analogues, histogram-based gradient boosting, regime-aware priors, empirical residual simulation, adaptive ensemble weighting, and hierarchical reconciliation to generate probabilistic influenza forecasts across national, regional, and state levels.

NIHEF produces weekly forecasts for:

* Cases
* Hospitalisations
* Deaths
* National, regional, and state geographic levels
* Point forecasts
* Prediction intervals
* Quantile forecasts

The framework was designed specifically to address the temporal, geographic, seasonal, and hierarchical characteristics of influenza transmission in Nigeria.

---

## Key Contributions

NIHEF introduces four principal methodological innovations:

1. Harmattan–Wet Season Regime-Switching Prior (HWS-RSP)
2. Climate-informed regime mixing prior
3. Growth-conditioned analogue peak-quantile adjustment
4. Peak-window adaptive ensemble weighting

These components are integrated with seasonal-analogue forecasting, histogram gradient boosting, empirical residual simulation, and hierarchical reconciliation.

---

## Why NIHEF?

Seasonal influenza forecasting in Nigeria presents several challenges:

* Strong seasonal variation
* Geographic heterogeneity
* Differences between northern and southern transmission patterns
* Possible secondary epidemic waves
* Uneven surveillance information
* Dependence between cases, hospitalisations, and deaths
* The need to quantify forecast uncertainty
* The need for coherent forecasts across geographic levels

NIHEF addresses these challenges using a multi-component probabilistic ensemble rather than relying on a single forecasting model.

---

## Methodological Innovations

### 1. Harmattan–Wet Season Regime-Switching Prior (HWS-RSP)

HWS-RSP is the first forecasting component to model Nigeria's Harmattan-associated secondary influenza wave as a distinct transmission regime within a probabilistic forecasting framework.

The component introduces an explicit regime-switching prior representing distinct seasonal transmission behaviour and allows the predictive distribution to account for the possibility of a secondary wave associated with Nigeria's Harmattan and wet-season dynamics.

Rather than treating seasonal structure as a fixed historical pattern, HWS-RSP incorporates regime-specific prior information directly into the probabilistic ensemble.

This allows the model to represent changes in seasonal burden and uncertainty while retaining information from observed historical influenza seasons.

This formulation constitutes one of the principal methodological innovations of NIHEF.

### 2. Climate-Informed Regime Mixing Prior

Climate-zone information is incorporated into the regime prior through Bayesian-style mixing of empirical regime burden shares and climate-informed expectations.

This allows the framework to account for geographic differences in seasonal influenza behaviour while retaining information from observed historical seasons.

The climate-informed prior is applied as a probabilistic weighting mechanism rather than as a deterministic climate-to-incidence relationship.

### 3. Growth-Conditioned Analogue Peak-Quantile Adjustment

The seasonal analogue component is adjusted according to observed early-season growth.

This provides a mechanism for conditioning the upper-tail forecast on the trajectory of the current season, allowing the predictive distribution to respond to unusually rapid or slow epidemic growth.

The adjustment is designed to improve representation of peak magnitude while preserving the historical seasonal shape supplied by the analogue component.

### 4. Peak-Window Adaptive Ensemble Weighting

NIHEF allows the relative contribution of forecasting components to vary during the expected epidemic peak window.

The analogue component receives greater influence during the peak period, where historical seasonal shape can provide useful information about peak timing and magnitude.

This adaptive weighting is intended to improve peak-period forecast performance without requiring the same model contribution throughout the entire forecast horizon.

---

## Model Architecture

The NIHEF forecasting architecture can be summarised as:

```text
                         NIHEF
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
      Seasonal         HWS-RSP       Climate-informed
      analogue          prior          regime mixing
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                 Growth-conditioned
                 peak adjustment
                           │
                           ▼
                 Adaptive ensemble
                     weighting
                           │
                           ▼
              Empirical residual
                   simulation
                           │
                           ▼
             Hierarchical reconciliation
                           │
                           ▼
             Probabilistic forecasts
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
     National           Regional            State
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                Cases / Hospitalisations /
                       Deaths
```

The ensemble combines complementary forecasting mechanisms:

| Component                   | Primary role                           |
| --------------------------- | -------------------------------------- |
| Seasonal analogue           | Historical seasonal pattern            |
| Histogram Gradient Boosting | Nonlinear predictive modelling         |
| HWS-RSP                     | Seasonal regime representation         |
| Climate-informed prior      | Geographic regime information          |
| Growth adjustment           | Current-season trajectory conditioning |
| Adaptive ensemble weighting | Peak-period model combination          |
| Residual simulation         | Predictive uncertainty                 |
| Hierarchical reconciliation | Cross-level consistency                |

---

## Forecast Hierarchy

NIHEF produces forecasts across three geographic levels:

```text
                    Nigeria
                       │
              ┌────────┴────────┐
              │                 │
             North             South
              │                 │
       ┌──────┼──────┐    ┌─────┼─────┐
       │      │      │    │     │     │
     State  State  State State State State
       │      │      │    │     │     │
       └──────┴──────┴────┴─────┴─────┘
                       │
                       ▼
             Hierarchical reconciliation
```

The hierarchical forecasting process is designed to maintain consistency between national, regional, and state-level forecasts.

Forecasts are generated for:

* Nigeria
* Northern region
* Southern region
* Individual states

and for three target variables:

* Cases
* Hospitalisations
* Deaths

---

## Repository Structure

```text
NIHEF/
│
├── README.md
├── requirements.txt
│
├── nihef_v10_3.py
├── nihef_ablation.py
├── make_report.py
│
├── data/
│   ├── national.csv
│   ├── state.csv
│   └── metadata.csv
│
├── outputs/
│   ├── submission.csv
│   ├── seasonal_forecast_summary.csv
│   ├── validation_model_comparison.csv
│   ├── oof_calibration_metrics.csv
│   ├── oof_peak_metrics.csv
│   ├── oof_seasonal_metrics.csv
│   └── ablation_summary.csv
│
├── figures/
│   ├── fig01_national_weekly.png
│   ├── fig02_regional_curves.png
│   ├── fig03_north_vs_south.png
│   ├── fig04_state_heatmap.png
│   ├── fig05_attack_rate.png
│   ├── fig06_mortality.png
│   ├── fig07_peak_week.png
│   ├── fig08_uncertainty_bands.png
│   └── fig09_ablation_comparison.png
│
└── docs/
    ├── data_dictionary.md
    ├── methodology.md
    ├── functions.md
    └── reproduction.md
```

---

## Installation

NIHEF is implemented in Python.

A clean virtual environment is recommended.

### Create the environment

```bash
python -m venv .venv
```

### Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

The principal Python libraries used by NIHEF are:

| Package      | Version |
| ------------ | ------: |
| NumPy        |   2.1.3 |
| pandas       |   2.2.3 |
| scikit-learn |   1.5.2 |
| SciPy        |  1.14.1 |
| Matplotlib   |   3.9.2 |
| python-docx  |   1.1.2 |

The implementation is intended for Python 3.10+ subject to compatibility with the specified dependency versions.

---

## Input Data

The framework expects the following principal input files:

```text
data/
├── national.csv
├── state.csv
└── metadata.csv
```

### National data

The national dataset contains weekly observations for:

* Season
* Epidemiological week
* Cases
* Hospitalisations
* Deaths

### State data

The state-level dataset contains:

* Season
* Epidemiological week
* State
* Cases
* Hospitalisations
* Deaths

### Metadata

The metadata file contains geographic and contextual information including:

* State
* Geopolitical zone
* Population
* Climate classification
* Healthcare accessibility

A detailed description of all input and output fields is provided in:

`docs/data_dictionary.md`

If challenge data are subject to redistribution restrictions, the original challenge-provided files should be obtained through the appropriate challenge channel and placed in the `data/` directory before running the pipeline.

---

## Running the Main Forecasting Pipeline

After installing the dependencies and placing the required input data in the `data/` directory, run:

```bash
python nihef_v10_3.py
```

The main pipeline generates:

* Probabilistic forecasts
* Seasonal forecast summaries
* Validation metrics
* Out-of-fold calibration metrics
* Peak metrics
* Seasonal metrics
* Figures 1–8
* Submission-ready forecast files

The principal forecast output is:

```text
outputs/submission.csv
```

---

## Running the Ablation Analysis

The contribution of the principal NIHEF components can be evaluated using:

```bash
python nihef_ablation.py --ablation --parallel 4 --fast
```

The ablation framework evaluates the following configurations:

```text
full
no_hws
no_climate
no_growth
no_peak_weights
no_iterative_recon
baseline
```

The resulting summary is saved to:

```text
outputs/ablation_summary.csv
```

The corresponding visual comparison is:

```text
figures/fig09_ablation_comparison.png
```

The ablation analysis evaluates how individual components affect:

* Point forecast accuracy
* Probabilistic forecast accuracy
* Calibration
* Peak estimation
* Cumulative seasonal burden
* Forecast coherence

---

## Ablation Configurations

| Configuration        | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| `full`               | Complete NIHEF framework                                        |
| `no_hws`             | HWS-RSP removed                                                 |
| `no_climate`         | Climate-informed prior removed                                  |
| `no_growth`          | Growth-conditioned adjustment removed                           |
| `no_peak_weights`    | Peak-window adaptive weighting removed                          |
| `no_iterative_recon` | Iterative reconciliation removed                                |
| `baseline`           | Baseline configuration without the principal NIHEF enhancements |

The purpose of the ablation study is to determine how different methodological components influence different dimensions of forecasting performance.

---

## Generating the Technical Report

The report-generation script can be executed using:

```bash
python make_report.py
```

This generates the formatted technical report using the available forecast outputs, evaluation results, tables, and figures.

---

## Forecast Outputs

The principal submission file contains the following fields:

| Field    | Description                        |
| -------- | ---------------------------------- |
| `level`  | Geographic hierarchy level         |
| `group`  | Geographic group                   |
| `target` | Cases, hospitalisations, or deaths |
| `week`   | Forecast epidemiological week      |
| `mean`   | Mean forecast                      |
| `p05`    | 5th percentile                     |
| `p25`    | 25th percentile                    |
| `p50`    | 50th percentile                    |
| `p75`    | 75th percentile                    |
| `p95`    | 95th percentile                    |

The resulting forecast distribution therefore provides both central estimates and uncertainty information.

---

## Evaluation Framework

NIHEF is evaluated using both point-forecast and probabilistic metrics.

### Point Forecast Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Peak-week error
* Peak-incidence error
* Cumulative seasonal error

### Probabilistic Metrics

* Weighted Interval Score (WIS)
* Continuous Ranked Probability Score (CRPS)
* 50% prediction-interval coverage
* 90% prediction-interval coverage
* Quantile forecast evaluation

This evaluation framework assesses both the accuracy of the central forecast and the quality of the predictive distribution.

---

## Forecast Performance

For the evaluated Fold 2 forecast period, the ensemble achieved:

| Target           |   MAE | 90% PI Coverage | Peak-week Error | Cumulative Error |
| ---------------- | ----: | --------------: | --------------: | ---------------: |
| Cases            | 5,059 |           0.903 |         0 weeks |            -1.1% |
| Hospitalisations |  68.6 |           0.919 |         -1 week |            -1.1% |
| Deaths           |  7.76 |           0.903 |         0 weeks |            +4.2% |

These results demonstrate the framework's performance in the evaluated forecast period across point accuracy, uncertainty coverage, peak timing, and cumulative seasonal burden.

The pooled evaluation and out-of-fold results are available in:

```text
outputs/validation_model_comparison.csv
outputs/oof_calibration_metrics.csv
outputs/oof_peak_metrics.csv
outputs/oof_seasonal_metrics.csv
```

---

## Probabilistic Forecasting

A central objective of NIHEF is to produce calibrated predictive distributions rather than point forecasts alone.

For the evaluated Fold 2 period, the 90% prediction intervals achieved:

| Target           | 90% Coverage |
| ---------------- | -----------: |
| Cases            |        0.903 |
| Hospitalisations |        0.919 |
| Deaths           |        0.903 |

The framework therefore provides explicit uncertainty estimates around expected influenza burden.

The predictive distribution is generated through empirical residual simulation and quantile construction around the ensemble forecast.

---

## Visualisations

The repository contains nine principal figures.

| Figure   | Description                            |
| -------- | -------------------------------------- |
| Figure 1 | National weekly probabilistic forecast |
| Figure 2 | Regional epidemic curves               |
| Figure 3 | North versus South comparison          |
| Figure 4 | State-level forecast heatmap           |
| Figure 5 | Attack-rate comparison                 |
| Figure 6 | Mortality-rate comparison              |
| Figure 7 | Peak-week probability distributions    |
| Figure 8 | Forecast uncertainty bands             |
| Figure 9 | Ablation comparison                    |

The figures provide complementary views of:

* Temporal epidemic dynamics
* Geographic heterogeneity
* Regional differences
* State-level variation
* Per-capita burden
* Mortality burden
* Peak timing
* Predictive uncertainty
* Component-level performance

---

## Geographic Forecasting

NIHEF explicitly represents geographic differences in influenza dynamics.

The framework distinguishes between northern and southern transmission patterns and produces state-level forecasts.

This allows the model to capture differences in:

* Timing of seasonal peaks
* Peak magnitude
* Secondary-wave behaviour
* Absolute burden
* Population-adjusted burden
* Mortality patterns

The state-level outputs can be used to examine geographic heterogeneity beyond the national aggregate.

---

## Hierarchical Reconciliation

Forecasts are generated across multiple geographic levels.

Hierarchical reconciliation is applied to improve consistency between:

```text
State
  ↓
Regional
  ↓
National
```

The reconciliation process is performed iteratively to reduce discrepancies between aggregate and lower-level forecasts.

The reconciliation component is evaluated separately from the predictive model components so that improvements in hierarchy coherence are not incorrectly interpreted as direct improvements in out-of-fold forecast accuracy.

---

## Reproducibility

The NIHEF repository is designed to support computational reproduction of the forecasting workflow.

The standard reproduction sequence is:

```bash
pip install -r requirements.txt
python nihef_v10_3.py
python nihef_ablation.py --ablation --parallel 4 --fast
python make_report.py
```

The main pipeline generates the forecast outputs and Figures 1–8.

The ablation pipeline generates the component comparison and Figure 9.

The report-generation script assembles the available outputs into the technical report.

All stochastic procedures use the fixed random seed:

```text
20260909
```

The implementation also uses deterministic SHA-256 keyed random-number generation for reproducible stochastic operations.

Detailed reproduction instructions are available in:

`docs/reproduction.md`

Additional technical documentation is available in:

* `docs/methodology.md`
* `docs/data_dictionary.md`
* `docs/functions.md`

---

## Reproduction Workflow

```text
Install dependencies
        │
        ▼
Verify input data
        │
        ▼
Run main NIHEF pipeline
        │
        ├──────────────► Forecast outputs
        │
        └──────────────► Figures 1–8
        │
        ▼
Run ablation analysis
        │
        └──────────────► Figure 9
        │
        ▼
Generate technical report
        │
        ▼
Review outputs and metrics
```

---

## Computational Requirements

Approximate runtime for the current implementation:

| Task                      |              Approximate Runtime |
| ------------------------- | -------------------------------: |
| Main forecasting pipeline |                      ~30 minutes |
| Ablation analysis         | ~6 hours with parallel execution |
| Report generation         |                        ~1 minute |

Runtime depends on hardware, Python environment, number of parallel workers, and system load.

Approximately 8 GB of RAM is recommended for the main workflow, with 16 GB preferable for the complete ablation analysis.

---

## Randomness and Determinism

NIHEF uses a fixed random seed:

```text
20260909
```

Random operations are controlled using deterministic keyed random-number generation.

This design reduces variation between repeated runs and supports reproducible forecast generation.

Where machine-learning components use stochastic procedures, fixed random states are used where applicable.

---

## Documentation

Detailed documentation is provided in the `docs/` directory.

| Document                  | Purpose                                    |
| ------------------------- | ------------------------------------------ |
| `docs/methodology.md`     | Mathematical and computational methodology |
| `docs/data_dictionary.md` | Input and output data structures           |
| `docs/functions.md`       | Function-level documentation               |
| `docs/reproduction.md`    | Step-by-step reproduction instructions     |

---

## AI and Computational Tools Disclosure

Python and its scientific computing libraries were used for data processing, model development, forecasting, validation, analysis, and visualisation.

The principal libraries include:

* pandas
* NumPy
* scikit-learn
* SciPy
* Matplotlib
* python-docx

ChatGPT (OpenAI) was used as a supplementary tool for:

* Code debugging
* Troubleshooting programming errors
* Grammar

All code, results, analyses, and conclusions were reviewed and verified by the author.

AI was not used to generate or fabricate data, results, or model outputs.

---

## Reproducibility Checklist

Before releasing the repository, verify:

* [ ] `README.md` matches the final implementation
* [ ] `requirements.txt` is present
* [ ] Input data requirements are documented
* [ ] `nihef_v10_3.py` runs successfully
* [ ] `nihef_ablation.py` runs successfully
* [ ] `make_report.py` runs successfully
* [ ] Forecast outputs are generated
* [ ] Evaluation metrics are generated
* [ ] Figures 1–9 are generated
* [ ] Random seed is fixed
* [ ] Report values match generated outputs
* [ ] Documentation matches the final code
* [ ] Repository URL has been replaced with the final GitHub address
* [ ] License information has been completed
* [ ] Contact information has been completed

---

## Citation

If you use or reference NIHEF, please cite the associated MAEPiMS Challenge 2026 technical submission.

A formal `CITATION.cff` file may also be included in the repository to provide machine-readable citation information.

---

## Acknowledgements

This work was developed for the MAEPiMS Challenge 2026.

The challenge dataset was provided by the Adejimi Adeniji Foundation.

---

## License

This repository is intended for research and educational purposes.

The final project licence should be specified in the repository before public release.

---

## Contact

For questions regarding NIHEF, the forecasting methodology, reproduction, or technical documentation, please use the contact information associated with the final project repository.

---

## Project Status

NIHEF is a research forecasting framework developed for the MAEPiMS Challenge 2026 technical submission.

The repository contains:

* Forecasting source code
* Probabilistic forecasting methodology
* Hierarchical forecasting procedures
* Evaluation and validation tools
* Ablation analysis
* Visualisations
* Data documentation
* Function documentation
* Reproduction instructions
* Technical report generation

The repository is intended to provide a transparent and reproducible computational record of the NIHEF forecasting framework.
