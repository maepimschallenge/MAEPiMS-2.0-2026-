"""
Renewal core - Component 1 of the Climate-Phased Dual-Wave Renewal Ensemble.
See docs/DESIGN.md for the full architecture and rationale.

R_t is modelled directly (not derived from SEIR compartments) as a baseline
transmissibility multiplicatively forced by two "wave" bumps (wet-season +
Harmattan, the second allowed to rise and fall asymmetrically) and a
susceptible-depletion feedback term:

    R_t   = R_base * (1 + a1*bump(t;t1,s1) + a2*asym_bump(t;t2,s2_rise,s2_fall)) * susceptible(t)
    I_t   = R_t * sum_{k=1..L} w_k * I_{t-k}          (discrete renewal equation)

`susceptible(t)` is a *leaky* depletion pool rather than plain cumulative-incidence
depletion: `depletion_decay < 1` lets it partially replenish between waves (new local
mixing pools, e.g. school terms restarting, rather than literal waning immunity within
one season). This matters because a single non-leaky pool forces wave 1 and wave 2 to
compete for the same fixed susceptible budget - v0 hit this directly (`s_max` and `s2`
pinned at their upper bounds trying to leave room for a second wave); see docs/DESIGN.md
"Status / next steps" for the diagnosis.

Fit per season (and per state, with optional shrinkage toward the national fit)
by nonlinear least squares against observed weekly cases.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares

N_SEED = 3          # number of initial weeks taken directly from observed data
KERNEL_LEN = 4       # generation-time kernel length, in weeks


def generation_kernel(length: int = KERNEL_LEN, mean: float = 1.0, shape: float = 2.0) -> np.ndarray:
    """Discretized gamma-shaped generation-time kernel, weekly resolution, normalized to sum 1.

    Influenza's true generation interval (~2-3 days) is sub-weekly; at weekly resolution
    this collapses to a kernel dominated by lag 1 with a short tail, which is what this
    gamma discretization gives for shape=2, mean=1 week.
    """
    from scipy.stats import gamma
    scale = mean / shape
    edges = np.arange(0, length + 1)
    cdf = gamma.cdf(edges, a=shape, scale=scale)
    w = np.diff(cdf)
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def gaussian_bump(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / max(width, 1e-3)) ** 2)


def asym_bump(t: np.ndarray, center: float, width_rise: float, width_fall: float) -> np.ndarray:
    """Gaussian bump with a different width before vs after its center - lets a wave's
    rise and decline have different steepness instead of forcing symmetry."""
    t = np.asarray(t, dtype=float)
    width = np.where(t <= center, max(width_rise, 1e-3), max(width_fall, 1e-3))
    return np.exp(-0.5 * ((t - center) / width) ** 2)


@dataclass
class RenewalParams:
    r_base: float
    a1: float
    t1: float
    s1: float
    a2: float
    t2: float
    s2_rise: float
    s2_fall: float
    s_max: float          # effective susceptible fraction ceiling (0, 1]
    depletion_decay: float  # 1.0 = permanent depletion; <1 lets the susceptible pool leak/replenish

    def as_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RenewalParams":
        return cls(*arr)


# (name, lower bound, upper bound, initial guess)
_PARAM_SPEC = [
    ("r_base", 0.3, 2.5, 1.0),
    ("a1", 0.0, 15.0, 3.0),
    ("t1", 5.0, 25.0, 14.0),
    ("s1", 0.5, 6.0, 2.0),
    ("a2", 0.0, 15.0, 1.0),
    ("t2", 15.0, 40.0, 27.0),
    ("s2_rise", 0.5, 6.0, 2.0),
    ("s2_fall", 0.5, 6.0, 2.0),
    ("s_max", 0.05, 0.95, 0.5),
    ("depletion_decay", 0.85, 1.0, 0.98),
]
LOWER = np.array([p[1] for p in _PARAM_SPEC])
UPPER = np.array([p[2] for p in _PARAM_SPEC])
INIT = np.array([p[3] for p in _PARAM_SPEC])


def simulate(
    params: RenewalParams,
    weeks: np.ndarray,
    seed_cases: np.ndarray,
    population: float,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate the renewal equation forward from `seed_cases` (length >= len(kernel)).

    Returns the simulated weekly case series over the full `weeks` horizon (seed weeks
    included, reproduced exactly from `seed_cases`).
    """
    if kernel is None:
        kernel = generation_kernel()
    L = len(kernel)
    n = len(weeks)
    I = np.zeros(n)
    n_seed = min(len(seed_cases), n)
    I[:n_seed] = seed_cases[:n_seed]

    # Leaky depletion pool: depletion[t] = decay*depletion[t-1] + I[t]. With
    # decay=1 this is plain cumulative incidence (the old, non-leaky behaviour);
    # decay<1 lets the pool partially replenish, so wave 2 isn't starved of
    # "susceptibles" by wave 1's depletion (see module docstring).
    depletion = np.zeros(n)
    for i in range(n_seed):
        prev = depletion[i - 1] if i > 0 else 0.0
        depletion[i] = params.depletion_decay * prev + I[i]

    for t in range(n_seed, n):
        lag_window = I[max(0, t - L):t][::-1]
        w = kernel[: len(lag_window)]
        w = w / w.sum() if len(lag_window) < L else kernel
        infectious_pressure = np.dot(w, lag_window)

        wave1 = gaussian_bump(weeks[t], params.t1, params.s1)
        wave2 = asym_bump(weeks[t], params.t2, params.s2_rise, params.s2_fall)
        susceptible_frac = max(1.0 - depletion[t - 1] / (population * params.s_max), 0.02)
        r_t = params.r_base * (1 + params.a1 * wave1 + params.a2 * wave2) * susceptible_frac

        I[t] = max(r_t * infectious_pressure, 0.0)
        depletion[t] = params.depletion_decay * depletion[t - 1] + I[t]

    return I


def _residuals(x: np.ndarray, weeks, observed, population, kernel, prior=None, prior_weight=0.0) -> np.ndarray:
    params = RenewalParams.from_array(x)
    sim = simulate(params, weeks, observed[:N_SEED], population, kernel)
    # sqrt-transform stabilizes variance across the huge seed-vs-peak magnitude range
    resid = np.sqrt(np.maximum(sim, 0)) - np.sqrt(np.maximum(observed, 0))
    if prior is not None and prior_weight > 0:
        reg = prior_weight * (x - prior) / (UPPER - LOWER)
        return np.concatenate([resid, reg])
    return resid


def fit_series(
    weeks: np.ndarray,
    observed: np.ndarray,
    population: float,
    kernel: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> tuple[RenewalParams, np.ndarray]:
    """Fit renewal-model parameters to one season's weekly case series (national or one state).

    `prior` / `prior_weight`: optional shrinkage toward another fit's parameters (e.g. a
    state's fit shrinking toward the national fit) - the hierarchical-calibration idea from
    docs/DESIGN.md, implemented here as a simple ridge-style penalty rather than a full
    mixed-effects model for v0.
    """
    if kernel is None:
        kernel = generation_kernel()
    x0 = prior.copy() if prior is not None else INIT.copy()
    result = least_squares(
        _residuals,
        x0=x0,
        bounds=(LOWER, UPPER),
        args=(weeks, observed, population, kernel, prior, prior_weight),
        method="trf",
    )
    params = RenewalParams.from_array(result.x)
    sim = simulate(params, weeks, observed[:N_SEED], population, kernel)
    return params, sim


def param_covariance(result) -> np.ndarray:
    """Approximate parameter covariance from a `least_squares` result's Jacobian:
    `cov = sigma^2 * (J^T J)^-1`, the standard asymptotic normal approximation for
    nonlinear least squares (e.g. Seber & Wild). Used by `src.ensemble`'s parametric
    bootstrap for the optional probabilistic targets (P(rising), P(peak within k
    weeks)) - this is *not* used by the point forecast itself, so a rough covariance
    approximation is an acceptable trade-off. When `fit_series` was called with a
    `prior`/`prior_weight` regularizer, its residual rows are still in `result.fun` /
    `result.jac`, so the returned covariance is already appropriately tightened by
    that shrinkage (a regularizer is, from a Bayesian view, informative about the
    posterior too) rather than needing separate handling here.
    """
    J = result.jac
    dof = max(len(result.fun) - J.shape[1], 1)
    sigma_sq = float(np.sum(result.fun ** 2) / dof)
    try:
        cov = sigma_sq * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = sigma_sq * np.linalg.pinv(J.T @ J)
    return cov


def fit_series_with_covariance(
    weeks: np.ndarray,
    observed: np.ndarray,
    population: float,
    kernel: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> tuple[RenewalParams, np.ndarray, np.ndarray]:
    """Same fit as `fit_series`, additionally returning the approximate parameter
    covariance matrix (see `param_covariance`) for `src.ensemble`'s bootstrap."""
    if kernel is None:
        kernel = generation_kernel()
    x0 = prior.copy() if prior is not None else INIT.copy()
    result = least_squares(
        _residuals,
        x0=x0,
        bounds=(LOWER, UPPER),
        args=(weeks, observed, population, kernel, prior, prior_weight),
        method="trf",
    )
    params = RenewalParams.from_array(result.x)
    sim = simulate(params, weeks, observed[:N_SEED], population, kernel)
    return params, sim, param_covariance(result)


def fit_rate_ratios(cases: np.ndarray, hospitalizations: np.ndarray, deaths: np.ndarray) -> dict:
    """Simple constant hosp/case and death/case ratios (least-squares through the origin).

    A v0 stand-in for a time-varying severity model; revisit if residuals show the ratio
    drifting across the season (e.g. healthcare-capacity saturation near the peak).
    """
    cases = np.maximum(np.asarray(cases, dtype=float), 1e-6)
    h_ratio = float(np.sum(cases * hospitalizations) / np.sum(cases ** 2))
    d_ratio = float(np.sum(cases * deaths) / np.sum(cases ** 2))
    return {"hosp_ratio": h_ratio, "death_ratio": d_ratio}
