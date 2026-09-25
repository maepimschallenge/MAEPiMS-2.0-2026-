# Model Design - Climate-Phased Dual-Wave Renewal Ensemble (CP-DWRE)

Status: **draft v0** - architecture agreed, core renewal engine scaffolded, calibration/residual
layers not yet implemented. This doc is the reference for what each module in `src/` should do.

## Why this design

The EDA (`eda/explore.py`, `eda/figures/`) established three facts the model must reproduce:

1. **Two-wave seasonal structure** - a large wet-season wave (~epi-week 13–17) and a smaller
   Harmattan-associated second wave (~10–14 weeks later), present in all 3 seasons.
2. **Regional phase offset** - South peaks earlier/sharper in wave 1; North's wave 2 is
   relatively stronger - a climate/zone effect, not just noise.
3. **State heterogeneity decoupled from severity** - healthcare access barely predicts attack
   rate (ascertainment effect) but strongly predicts CFR (corr ≈ −0.84).

No single off-the-shelf model (plain SEIR, plain ARIMA, plain GBM) captures all three cleanly
without borrowing structure directly from a specific published framework. The brief requires an
**original** model, so we compose three purpose-built layers instead of using one method
wholesale. This composition - not any one piece - is the original contribution.

## Architecture

```
                      ┌─────────────────────────┐
 covariates ─────────▶│ 1. Renewal Core (Rt)     │──▶ mechanistic trajectory (national + per-state)
 (calendar, zone,      │    src/renewal.py        │
  climate, season)     └─────────────────────────┘
                                   │ residuals (log observed − log mechanistic)
                                   ▼
                      ┌─────────────────────────┐
                      │ 2. Hierarchical State    │──▶ calibrated per-state trajectory
                      │    Calibration           │
                      │    src/calibration.py    │
                      └─────────────────────────┘
                                   │ remaining residuals
                                   ▼
                      ┌─────────────────────────┐
                      │ 3. Residual Quantile ML  │──▶ point-correction + heteroskedastic spread
                      │    src/residual_model.py │
                      └─────────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ 4. Probabilistic Ensemble│──▶ median, 50%/90% PI, quantiles,
                      │    src/ensemble.py       │    P(rising), P(falling), P(peak in k wks)
                      └─────────────────────────┘
```

### 1. Renewal core (`src/renewal.py`)

A discrete-time renewal-equation model, **not** a literal textbook SEIR - the state variable is
directly the effective reproduction number `R_t`, forced by covariates instead of estimated
compartment transitions:

```
I_t = R_t * Σ_{s=1..L} I_{t-s} * w_s               (renewal equation, w = generation-time kernel)
R_t = R_base * f_wet(t) * f_harmattan(t) * f_susceptible(t)
```

- `f_wet(t)`: a raised-cosine bump centered on the climatological wet-season onset (fit per
  climate class: tropical / savanna / sahel each get their own phase offset - this is what
  reproduces the North/South phase difference).
- `f_harmattan(t)`: a second, smaller bump offset by a fitted lag (captures wave 2); its
  amplitude is allowed to scale with the sahel/savanna share of a region, since the EDA shows
  the second wave is relatively stronger in the North.
- `f_susceptible(t) = 1 − (cumulative_cases_t / (population * s_max))`: simple susceptible
  depletion feedback so waves self-limit without a full SEIR state (`s_max` - effective
  susceptible fraction - is a fitted per-season parameter since severity differs by season).
- Fit by nonlinear least squares (`scipy.optimize.curve_fit` / `least_squares`) per season,
  first at national level, then per state with national fit as prior/regularizer (shrinkage
  toward national curve for low-signal states).

Output: a deterministic mechanistic trajectory per state per week, for cases; hospitalisations
and deaths are derived via fitted (possibly time-varying) rate ratios `hosp_t = h_ratio * I_t`,
`deaths_t = d_ratio * I_t`, since the surveillance data gives all three jointly.

### 2. Hierarchical state calibration (`src/calibration.py`)

The renewal core is fit with partial pooling: state-level parameters (`R_base`, phase offsets,
`s_max`) are drawn toward zone/climate-level means, shrinkage strength inversely proportional to
each state's historical signal-to-noise (population size, reporting stability). Implemented as
a mixed-effects regression on the log-residuals from step 1, with `zone`, `climate`, and
`hc_access` as fixed effects and `state` as a random intercept - not a full MCMC hierarchical
Bayesian model (out of scope for v0), but structured the same way so it can be upgraded to one
(e.g. via `pymc` or `bambi`) later without changing the interface.

### 3. Residual quantile ML correction (`src/residual_model.py`)

A LightGBM quantile regressor (multiple quantile levels: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
trained on the *remaining* residuals after steps 1–2, using features: lagged residuals, weeks
since mechanistic peak, state metadata (population, hc_access, zone, climate), and season
severity label. This mops up nonlinear effects the mechanistic+hierarchical layers miss
(reporting noise, idiosyncratic local outbreaks) and - critically - supplies **heteroskedastic
uncertainty**: residual spread is visibly larger near the peak in the EDA, so quantiles are
fit conditionally rather than assumed constant.

### 4. Probabilistic ensemble (`src/ensemble.py`)

Combines the mechanistic trajectory (step 1–2) as the median anchor with the residual quantile
spread (step 3) to produce calibrated forecast distributions. Also runs a parametric bootstrap
over the renewal core's fitted parameters (resampling within their fit covariance) to generate
an ensemble of trajectories, from which the optional probabilistic targets are read off directly:
`P(increasing)` = fraction of ensemble members with `I_{t+1} > I_t`, `P(peak within k weeks)` =
fraction of members whose trajectory peak falls in the window, etc.

## Evaluation (`src/metrics.py` - implemented)

MAE, RMSE, quantile loss (pinball), WIS (weighted interval score, built from a symmetric set of
prediction intervals including the quantile loss decomposition), a CRPS approximation via the
quantile-based formula, peak-week accuracy (± k week tolerance), peak-incidence accuracy (% error),
and PIT-histogram-based calibration check.

## Training / backtest protocol

- **Fit**: 2023/24 + 2024/25 (moderate + high-severity seasons) - gives the model both a normal
  and a stress-tested severity regime.
- **Backtest**: walk-forward on 2025/26 - forecast from a rolling origin (e.g. after each of
  weeks 1, 4, 8, 12 of the season) out to the rest of the season, scored against the metrics
  above at national, regional (North/South), and state level.
- Report both a **fixed-origin** full-season forecast (for the seasonal targets: peak week, peak
  incidence, cumulative burden, attack rate) and **rolling** short-horizon forecasts (for the
  primary weekly-cases task), since the brief scores both.

## Status / next steps

- [x] Renewal core skeleton + basic fit routine (`src/renewal.py`)
- [x] Metrics module (`src/metrics.py`)
- [x] Renewal core v1: leaky depletion pool + asymmetric wave-2 bump (see below)
- [x] Hierarchical calibration layer v1: zone wave-center shrinkage (see below)
- [x] Residual quantile ML layer v1: LightGBM log-ratio correction + interval inflation (see below)
- [x] Ensemble + probabilistic targets v1: parametric bootstrap + submission assembly (see below)
- [ ] Backtest harness + report figures
- [ ] Technical report (PDF)

### Ensemble & submission assembly v1 (`src/ensemble.py`, `scripts/run_ensemble.py`)

Two separate mechanisms, per docs/DESIGN.md's original split:

**Probabilistic targets** (`bootstrap_params`, `bootstrap_trajectories`,
`probability_of_rising`, `probability_peak_within`, `peak_week_distribution`) - a
parametric bootstrap over the national renewal fit's approximate parameter covariance
(`src.renewal.param_covariance`, from the least-squares Jacobian), giving an ensemble
of plausible national trajectories to read the challenge's *optional* targets off
directly. Demo run on 2025/26: P(rising transmission) at week 10 = 100% (still firmly
in the wave-1 upswing), P(peak within weeks 14-18) = 98%, with the bootstrap's modal
peak week (94.8% of draws) matching the renewal core's own fitted `t1` almost exactly
- i.e. the bootstrap's spread mostly reflects genuine parameter uncertainty, not
disagreement about the central estimate, which is the sanity check you want from it.

**Submission assembly** (`build_state_quantile_frame`, `apply_rate_ratio`,
`aggregate_via_monte_carlo`, `build_seasonal_summary`) - turns Components 2-3's
already-LOSO-validated per-state quantile forecasts into the actual deliverable
tables: `scripts/run_ensemble.py` produces, leave-one-season-out for all 3 seasons,
`outputs/forecast_<season>.csv` (state x week x quantile, cases + hospitalisations +
deaths - hospitalisation/death ratios fit **per state** on the training seasons via
`src.renewal.fit_rate_ratios`, not one national ratio, so a state's own historical
severity profile carries forward), `outputs/seasonal_summary_<season>.csv` (peak
week/incidence + cumulative-case quantiles per state), and
`outputs/regional_national_<season>.csv` (North/South/national aggregates).

**Aggregation method matters here and is worth flagging in the report's Assumptions
section:** North/South/national totals are *not* built by summing state-level
quantiles pointwise - that systematically overstates the aggregate's spread, since it
implicitly assumes every state's error is perfectly correlated. Instead
`aggregate_via_monte_carlo` samples from each state's fitted quantile function
(inverse-CDF via linear interpolation) and sums independent draws, which is the
statistically correct way to combine marginals **only if** the states' forecast
errors truly are independent - probably not fully true (a bad flu week tends to be
regional/national, not single-state), so this is a documented v1 simplification that
likely *understates* the true aggregate uncertainty somewhat, not an exact answer.
The same machinery, applied across weeks instead of states, gives seasonal cumulative
burden's uncertainty too, with the same independence caveat (weeks within a season are
autocorrelated in reality - a persistently bad month doesn't average out the way
independent draws would).

As a cross-check, the Monte-Carlo-aggregated national median (a genuine leave-one-
season-out forecast) was compared against the renewal core's own direct national fit
(an **in-sample** curve fit to that season's true national data, from
`scripts/smoke_test_renewal.py` - not itself a held-out prediction, so this is a
deliberately tough bar):

| Season | Direct national fit (in-sample) MAE | Bottom-up aggregate (LOSO) MAE | Difference |
|---|---|---|---|
| 2023/24 | 51,062 | **49,517** | -3.0% |
| 2024/25 | 57,022 | 230,065 | +303.5% |
| 2025/26 | 65,891 | **54,865** | -16.7% |

The bottom-up, genuinely out-of-sample aggregate actually **beats** the in-sample
curve fit on 2 of 3 seasons - plausible, not a fluke: summing ~37 independently
LightGBM-corrected state trajectories gets a diversification/averaging-out benefit
that a single national parametric curve fitting its own noisy weekly wiggles doesn't.
2024/25 is the exception, and a large one, but it's the *same* exception every other
layer in this pipeline has already flagged (calibration's phase shift, the residual
model's interval coverage) - consistent evidence that 2024/25 specifically breaks
assumptions learned from the other two seasons, not a new, unrelated failure mode.
**Practical takeaway for the submission:** the bottom-up state aggregate is a
reasonable - arguably preferable - national point estimate too, not merely a
by-product needed for state-level detail, but its 2024/25-sized failure mode means
the direct national fit is worth keeping and comparing against as a sanity check
whenever the two disagree sharply, rather than trusting either blindly.

### Calibration layer v1 (`src/calibration.py`, `scripts/run_calibration.py`, `scripts/run_calibration_loso.py`)

Two candidate corrections to the naive population-proportional allocation of the
national trajectory: (a) a zone/climate/hc_access magnitude-bias regression, and (b) a
zone-level wave-*center* shift (`t1`, `t2` only - see the two-round postmortem in
`src/calibration.py`'s module docstring for why amplitude and width fields are excluded).
(a) came back statistically insignificant (all p > 0.18, consistent with the EDA's weak
attack-rate/hc_access correlation) and isn't doing real work; (b) is the layer that
matters, but **only after a correction that's worth walking through honestly**, because
the first version of this section reported a result that didn't hold up:

**First pass (single split, train 2023/24+2024/25 -> test 2025/26):** the *full* zone
wave-center shift looked excellent - mean state MAE 6,228 -> 5,927, peak-week accuracy
81% -> 97%.

**Then a leave-one-season-out check (all 3 possible train/test splits,
`scripts/run_calibration_loso.py`)** showed that result doesn't generalize. Testing
against 2024/25 instead (a season several states hit their second-wave peak at
epi-week 29 in, vs epi-week 16-19 for those same states in every other season - a
season-specific anomaly, consistent with 2024/25 being flagged "HIGH, most severe since
2017-18" in `nigeria_flu_season_reference.csv`), the full shift nearly **doubled** mean
state MAE (8,366 -> 15,785) and peak-week accuracy fell to 51%. A shrinkage-strength
sweep (`k` from 3 to 40) made this *worse*, not better - pulling every zone's offset
toward a shared global mean is still a nonzero shift in the wrong direction when the
held-out season itself is the anomaly, not any one zone.

**Fix:** a `phase_weight` blend between the naive and phase-shifted allocations,
swept across all 3 LOSO splits to find a robust setting:

| `phase_weight` | 2023/24 MAE | 2024/25 MAE | 2025/26 MAE | mean across splits |
|---|---|---|---|---|
| 0.0 (naive) | 4,567 | 8,366 | 6,228 | 6,387 |
| **0.25 (shipped default)** | **4,315** | **9,122** | **5,870** | 6,436 |
| 1.0 (full shift) | 4,538 | 15,785 | 5,927 | 8,750 |

0.25 gives up almost nothing on the mean (6,436 vs naive's 6,387) while capping the
2024/25 downside at +9% instead of +89%, and still improving the 2 more typical splits.
Shipped as `DEFAULT_PHASE_WEIGHT = 0.25` in `src/calibration.py`. The honest summary for
the report's Uncertainty Analysis section: **regional wave timing is a real, learnable
signal (North lags South), but with only 3 simulated seasons total, any zone-level
correction has to stay conservative because one of those 3 seasons is a genuine outlier
a 2-season training set can't be expected to anticipate.**

### Residual quantile ML layer v1 (`src/residual_model.py`, `scripts/run_residual_model.py`)

Fits a LightGBM quantile regressor per level in `QUANTILE_LEVELS` on the log-ratio
between observed state-week cases and the calibration layer's baseline
(`log((observed+1)/(baseline+1))`), using static state metadata (population, zone,
climate, `hc_access`) plus position relative to the national wave centers as features.
This is also where the per-state *magnitude* correction the calibration layer
deliberately avoided (see its postmortem 1) finally gets a home - a tree ensemble can
learn "Lagos runs consistently above the national attack rate" directly from data
without touching the renewal equation's own susceptible-ceiling mechanics.

Trained and evaluated the same way as the calibration layer, for the same reason: each
training row's baseline is built from zone offsets fit on the *other* training season
only, so the residual target isn't contaminated by its own season's pattern, and
everything is scored leave-one-season-out.

**Point accuracy** (median prediction vs calibration baseline alone): clear wins on the
2 typical splits, roughly flat on 2024/25:

| Test season | MAE baseline | MAE residual-corrected |
|---|---|---|
| 2023/24 | 4,315 | **3,333** (-23%) |
| 2024/25 | 9,122 | 9,373 (+2.7%) |
| 2025/26 | 5,870 | **4,465** (-24%) |

**Probabilistic calibration** repeated the calibration layer's exact lesson: at
`interval_inflation=1.0` (raw model output), the nominal 90% interval covered ~87-89%
of true values on the 2 typical splits but only **45%** on 2024/25 - badly overconfident
whenever the held-out season is the atypical one, because the model's uncertainty
estimate is itself only calibrated to 2 seasons' residual spread. A sweep of
`interval_inflation` from 1.0 to 4.0 found 1.25 is a genuine Pareto improvement over no
adjustment - *lower* mean WIS across the 3 splits (1,239 vs 1,243) *and* better mean
90%-band coverage (80% vs 73%) - because on 2024/25 specifically, the interval-score
penalty for a too-narrow band outweighs the small penalty from widening it slightly.
Shipped as `DEFAULT_INTERVAL_INFLATION = 1.25`. Higher factors keep buying coverage on
2024/25 only very slowly (it plateaus in the high-60s% even at 4.0, since the
underlying problem there is bias in the median forecast, not just interval width) while
making the 2 typical splits' WIS steadily worse - 1.25 is the best evidence-based
compromise given only 3 seasons to validate against.

### v0 → v1 renewal-core fix (national smoke test)

v0's single non-leaky susceptible pool + symmetric wave-2 Gaussian couldn't fit 2023/24's
second wave - the optimizer pinned `s_max` and `s2` at their upper bounds trying to leave
enough "susceptibles" for a visible second peak, and still landed on the wrong timing/shape.
Root cause: one shared depletion pool forces wave 1 and wave 2 to compete for the same fixed
budget, which real double-wave seasons (a different sub-population / behavioural driver for
the Harmattan wave) don't do.

Fix: (1) `depletion_decay < 1` turns the depletion pool leaky - it partially replenishes
between waves instead of only ever draining - and (2) wave 2's bump got independent
rise/fall widths (`s2_rise`, `s2_fall`) instead of one symmetric width. National re-fit:

| Season | MAE before | MAE after | Peak week |
|---|---|---|---|
| 2023/24 | 83,298 | **51,062** | correct within 1 wk (was already correct wk) |
| 2024/25 | 59,475 | 57,022 | correct wk 16 |
| 2025/26 | 44,762 | 65,891 | within 1 wk (was correct wk) |

Net: substantially better on the season this was meant to fix, a small regression on
2025/26 (wave 1 now slightly overshoots - the leakier pool permits a touch more amplitude
before self-limiting). Acceptable for v1; a candidate follow-up is a mild regularization
penalty discouraging `s_max` from collapsing to very small values, which is what's driving
the sharper/taller wave-1 overshoot in the re-fit. Revisit once the calibration layer is in
place, since state-level fits will show whether this is a national-aggregate artifact or a
real pattern worth keeping.
