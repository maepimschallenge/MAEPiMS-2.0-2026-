# MAEPiMS Challenge 2026: Nigeria influenza forecasting (SVEAITHQR)

An original nine-compartment mechanistic model (S, V, E, A, I, H, T, Q, R) with
spline-forced seasonal transmission, a negative binomial observation layer, a coherent
state/region/national spatial layer and a validated probabilistic ensemble.
Only the supplied synthetic dataset is used.

## Layout
```
data/                      supplied CSV files (unchanged)
src/data_io.py             loading, checks, Sunday-Saturday epi-week labels
src/eda.py                 exploratory analysis                      -> F1-F10
src/model.py               ODE model, spline forcing, R0, endemic quadratic
src/parameters.py          assumed values, fitted-parameter specification
src/check_theory.py        numerical verification of the analytical results
src/fit_national.py        joint ML fit (3 seasons; env KAPPA_YEARS, TRAIN_SEASONS)
src/report_fit.py          parameter table, derived quantities         -> F11-F12
src/fit_state_severity.py  state severity vs healthcare access         -> F13
src/sensitivity.py         LHS-PRCC uncertainty/sensitivity            -> F14-F18
src/plots_3d_R0.py         3-D R0 surfaces/volumes                     -> F19-F24
src/plots_2d.py            2-D parameter-variation simulations         -> F25-F36
src/scoring.py             MAE, RMSE, quantile loss, WIS, CRPS, coverage
src/forecast.py            forecasting engine (mechanistic + analogue, disaggregation)
src/validate.py            time-ordered validation on 2025/26          -> F37-F38
src/run_forecast.py        2026/27 forecasts and CSV files             -> F39-F45
outputs/forecast/          submitted forecast CSV files
```

## Reproduce everything
```
pip install -r requirements.txt
bash run_all.sh            # about 25 minutes in total
```

## Forecast files (outputs/forecast/)
* `forecast_2026_27_weekly_quantiles.csv`: 40 locations (Nigeria, North, South, 36 states + FCT)
  x 3 targets (cases, hospitalizations, deaths) x 52 epi weeks (Sunday 28 Jun 2026 - Saturday
  26 Jun 2027) x 23 quantiles (0.5 = median; 0.25/0.75 = 50% PI; 0.05/0.95 = 90% PI).
* `forecast_2026_27_seasonal_targets.csv`: peak week, peak incidence per 100k, cumulative
  cases/hospitalisations/deaths, attack rate, deaths per 100k (quantiles).
* `forecast_2026_27_probabilities.csv`: P(peak in 4-week windows), weekly P(increase)/P(decrease).
* `forecast_2026_27_summary.csv`: medians and 90% intervals of the seasonal targets.

## Conventions and assumptions
* Supplied weeks start on Monday; each is reported under the Sunday-Saturday epi week starting
  the preceding Sunday (`epiweek_start = week_start - 1 day`); MMWR week numbers are added.
* Week 1 of every season (~4x week 2) is an artefact: excluded from fitting, reproduced in forecasts.
* North = NC (incl. FCT), NE, NW zones; South = SE, SS, SW zones.

## Software citations
Harris et al. (2020) NumPy, Nature 585:357-362; McKinney (2010) pandas, Proc. 9th Python in Science
Conf. 56-61; Virtanen et al. (2020) SciPy 1.0, Nature Methods 17:261-272; Hunter (2007) Matplotlib,
CiSE 9(3):90-95; Branch, Coleman & Li (1999) SIAM J. Sci. Comput. 21(1):1-23; Eilers & Marx (1996)
Stat. Sci. 11(2):89-121; Bracher et al. (2021) PLoS Comput. Biol. 17(2):e1008618; Gneiting & Raftery
(2007) JASA 102(477):359-378; Gneiting, Balabdaoui & Raftery (2007) JRSS-B 69(2):243-268.
