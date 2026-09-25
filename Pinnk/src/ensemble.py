"""
Probabilistic ensemble & submission builder - Component 4 of the CP-DWRE.
See docs/DESIGN.md.

Two distinct jobs:

1. **Parametric bootstrap of the national renewal fit** (`bootstrap_national_trajectories`
   + `probability_of_rising` / `probability_peak_within`) - resamples renewal parameters
   from the fit's approximate covariance (`src.renewal.param_covariance`) to answer the
   challenge's *optional* probabilistic targets directly from an ensemble of plausible
   national trajectories, rather than from the calibrated/residual-corrected quantiles
   (which are per-state magnitude/timing corrections, not a trajectory ensemble).

2. **Submission assembly** (`build_submission_table`, `aggregate_via_monte_carlo`,
   `build_seasonal_summary`) - stitches Components 1-3's already-validated outputs
   (a calibrated + residual-corrected per-state quantile forecast, produced exactly as
   in `scripts/run_residual_model.py`) into the long-format forecast tables the
   challenge's "Forecast dataset (CSV)" deliverable needs: state-level weekly
   quantiles, hospitalisation/death quantiles (via `src.renewal.fit_rate_ratios`),
   national/regional (North/South) aggregates, and seasonal summary targets (peak
   week distribution, cumulative burden).

   National/regional aggregation sums *samples*, not quantiles directly - summing
   independent states' quantiles pointwise is not statistically valid (it systematically
   overstates the aggregate's spread). `aggregate_via_monte_carlo` instead draws samples
   from each state's fitted quantile function (inverse-CDF via linear interpolation,
   flat-extrapolated past the outermost fitted quantile - an approximation, since the
   true tail beyond the 5th/95th percentile is unknown, but a standard and documented
   one), sums across states per draw, and re-derives quantiles of the sum. The same
   machinery, applied across weeks instead of states, gives the seasonal cumulative
   burden's uncertainty too - at the cost of treating weeks as independent draws, which
   understates autocorrelated uncertainty (a real epidemic's bad weeks cluster); flagged
   as a v1 simplification in docs/DESIGN.md rather than quietly assumed away.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.renewal import RenewalParams, simulate, LOWER, UPPER, N_SEED

DEFAULT_N_BOOTSTRAP = 500
DEFAULT_N_MC_SAMPLES = 2000


def bootstrap_params(
    params: RenewalParams, cov: np.ndarray, n_draws: int = DEFAULT_N_BOOTSTRAP, rng: np.random.Generator | None = None
) -> list[RenewalParams]:
    """`n_draws` parameter vectors sampled from N(params, cov), clipped to the
    renewal model's own valid bounds (LOWER/UPPER from src.renewal)."""
    rng = rng or np.random.default_rng()
    mean = params.as_array()
    # cov can be near-singular for well-constrained parameters; nearest-PSD nudge.
    cov = cov + np.eye(len(mean)) * 1e-10
    draws = rng.multivariate_normal(mean, cov, size=n_draws)
    draws = np.clip(draws, LOWER, UPPER)
    return [RenewalParams.from_array(d) for d in draws]


def bootstrap_trajectories(
    params_draws: list[RenewalParams],
    weeks: np.ndarray,
    seed_cases: np.ndarray,
    population: float,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """(n_draws, n_weeks) array: `src.renewal.simulate` re-run for each bootstrap draw,
    all sharing the same observed seed weeks (only the fitted dynamics vary)."""
    return np.stack([simulate(p, weeks, seed_cases[:N_SEED], population, kernel) for p in params_draws], axis=0)


def probability_of_rising(trajectories: np.ndarray, t_index: int) -> float:
    """Fraction of bootstrap trajectories with next week's cases > this week's, at
    `t_index`. `1 - probability_of_rising(...)` (excluding exact ties) is P(decreasing)."""
    return float(np.mean(trajectories[:, t_index + 1] > trajectories[:, t_index]))


def probability_peak_within(trajectories: np.ndarray, weeks: np.ndarray, week_lo: float, week_hi: float) -> float:
    """Fraction of bootstrap trajectories whose peak (argmax) falls in `[week_lo, week_hi]`."""
    peak_weeks = weeks[trajectories.argmax(axis=1)]
    return float(np.mean((peak_weeks >= week_lo) & (peak_weeks <= week_hi)))


def peak_week_distribution(trajectories: np.ndarray, weeks: np.ndarray) -> pd.Series:
    """Empirical probability of each week being the season peak, across bootstrap draws
    - a full distribution rather than the single point `probability_peak_within` reduces."""
    peak_weeks = weeks[trajectories.argmax(axis=1)]
    counts = pd.Series(peak_weeks).value_counts(normalize=True).sort_index()
    return counts.reindex(weeks, fill_value=0.0)


# --------------------------------------------------------------------------------
# Submission assembly
# --------------------------------------------------------------------------------

def build_state_quantile_frame(
    state_quantiles: dict[str, dict[float, np.ndarray]],
    weeks: np.ndarray,
    week_start_dates: np.ndarray,
    season: str,
    target: str = "cases",
) -> pd.DataFrame:
    """Long-format table: one row per (state, week, quantile level).

    `state_quantiles`: {state: {quantile_level: array of length len(weeks)}}, as
    produced by `src.residual_model.reconstruct_quantile_forecast` per state.
    """
    rows = []
    for state, qdict in state_quantiles.items():
        for q, values in qdict.items():
            for wk, wk_start, val in zip(weeks, week_start_dates, values):
                rows.append({
                    "season": season, "state": state, "epi_week_of_season": wk, "week_start": wk_start,
                    "target": target, "quantile_level": q, "value": float(val),
                })
    return pd.DataFrame(rows)


def apply_rate_ratio(state_quantiles: dict[str, dict[float, np.ndarray]], ratio: float) -> dict[str, dict[float, np.ndarray]]:
    """Scale every quantile trajectory by a constant rate ratio (hosp/case or
    death/case, from `src.renewal.fit_rate_ratios`) - monotonicity is preserved since
    scaling by a positive constant doesn't change the ordering across quantile levels."""
    return {state: {q: values * ratio for q, values in qdict.items()} for state, qdict in state_quantiles.items()}


def _sample_from_quantiles(
    qdict: dict[float, np.ndarray], n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_samples, n_weeks) samples per week, via inverse-CDF linear interpolation over
    the fitted quantile levels, flat-extrapolated below the lowest / above the highest
    fitted level (see module docstring's caveat on tail behaviour)."""
    levels = sorted(qdict)
    grid = np.stack([qdict[q] for q in levels], axis=0)  # (n_levels, n_weeks)
    u = rng.uniform(0, 1, size=n_samples)
    n_weeks = grid.shape[1]
    samples = np.empty((n_samples, n_weeks))
    for t in range(n_weeks):
        samples[:, t] = np.interp(u, levels, grid[:, t])
    return samples


def aggregate_via_monte_carlo(
    group_quantiles: dict[str, dict[float, np.ndarray]],
    quantile_levels: list[float],
    n_samples: int = DEFAULT_N_MC_SAMPLES,
    rng: np.random.Generator | None = None,
) -> dict[float, np.ndarray]:
    """Sum-of-samples aggregation across a group of series (states, for a regional or
    national total) - see module docstring for why this isn't just summed quantiles.
    Members are sampled independently; real states share season-wide drivers
    (documented v1 simplification, likely *understates* aggregate uncertainty)."""
    rng = rng or np.random.default_rng()
    members = list(group_quantiles.values())
    total = None
    for qdict in members:
        s = _sample_from_quantiles(qdict, n_samples, rng)
        total = s if total is None else total + s
    return {q: np.quantile(total, q, axis=0) for q in quantile_levels}


def build_seasonal_summary(
    state_quantiles: dict[str, dict[float, np.ndarray]],
    weeks: np.ndarray,
    quantile_levels: list[float],
    n_samples: int = DEFAULT_N_MC_SAMPLES,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-state seasonal targets: peak week & peak incidence (from the median
    trajectory) and cumulative-cases quantiles (via the same Monte Carlo summation as
    `aggregate_via_monte_carlo`, applied across weeks instead of states - same
    independence caveat, see module docstring)."""
    rng = rng or np.random.default_rng()
    rows = []
    for state, qdict in state_quantiles.items():
        median = qdict[0.5]
        peak_idx = int(np.argmax(median))
        samples = _sample_from_quantiles(qdict, n_samples, rng)  # (n_samples, n_weeks)
        cum_samples = samples.sum(axis=1)
        row = {
            "state": state, "peak_week": float(weeks[peak_idx]), "peak_incidence": float(median[peak_idx]),
        }
        for q in quantile_levels:
            row[f"cum_cases_q{q}"] = float(np.quantile(cum_samples, q))
        rows.append(row)
    return pd.DataFrame(rows)
