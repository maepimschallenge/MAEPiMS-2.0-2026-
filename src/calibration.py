"""
Hierarchical state calibration - Component 2 of the CP-DWRE. See docs/DESIGN.md.

Three pieces:

1. **Per-state renewal fits on training seasons** (`fit_all_states`) - each state's
   own renewal-core parameters, shrunk toward the national fit via the
   `prior`/`prior_weight` mechanism already in `src.renewal.fit_series`.

2. **Magnitude bias regression** (`fit_bias_regression`) - OLS of each state-season's
   total-count log-bias (observed vs a naive population-proportional split of the
   national trajectory) on zone, climate and `hc_access`. **Result on this dataset:
   no significant effect** (all p > 0.18 - consistent with the EDA's weak
   attack-rate/hc_access correlation, eda/figures/04). Kept in the pipeline since it's
   cheap and does no harm, but it is not the layer doing the real work - see (3).

3. **Zone-level phase shrinkage** (`fit_zone_shape_offsets` / `allocate_with_phase_shift`)
   - this is what actually matters. A scalar magnitude correction applied to the
   *national-shaped* curve can never get a state's peak timing right, and the EDA
   (eda/figures/02_north_vs_south.png) shows North and South clearly peak at different
   weeks. So: take each training state's fitted wave-center parameters (`t1`, `t2` -
   deliberately just the centers; see the two-round postmortem below) minus the
   national fit's, average the deltas by zone, shrink each zone's mean toward the
   global mean (empirical-Bayes-style, weight = n_zone / (n_zone + k) - the "plain
   shrinkage estimator" docs/DESIGN.md flagged as the v0 fallback for a full
   mixed-effects model), and re-simulate the renewal core at *national scale*
   (national seed weeks, national population) with only the wave centers nudged
   toward the zone's typical phase. That phase-shifted national curve is then split
   across states by population share, same as the naive baseline - the phase shift
   changes *when* each region's curve moves, population share still decides *how much*.
   On the single split tested first (train 2023/24+2024/25, test 2025/26), the *full*
   shift (`phase_weight=1.0`) looked excellent: mean state MAE 6,228 -> 5,927,
   peak-week accuracy (±1 week) 81% -> 97%. **That result does not generalize** - see
   `DEFAULT_PHASE_WEIGHT` below and `scripts/run_calibration_loso.py` for the full
   leave-one-season-out picture, which is why this ships blended at `phase_weight=0.25`
   rather than the full 1.0.

   **Postmortem, round 1 - why amplitude/ceiling fields are excluded:** an earlier
   version shifted `a1`/`a2` too and re-simulated each state from its own seed weeks
   with its own population, reusing the *national* `s_max`. Much worse than the naive
   baseline (mean MAE ~12.8k, 0/37 states improved). Cause: `s_max` sets the
   susceptible ceiling as `population * s_max`; states whose true attack rate is above
   the national average (e.g. Lagos, ~12.7% vs ~9.5% national in 2025/26) had their
   ceiling capped below their actual case count by construction. That's a per-state
   magnitude problem, which is what Component 3 (`src/residual_model.py`) is for - this
   layer stays magnitude-agnostic and leaves magnitude to population-proportional
   allocation (and later, the residual ML layer).

   **Postmortem, round 2 - why widths (`s1`, `s2_rise`, `s2_fall`) are also excluded:**
   with the ceiling issue fixed, shifting the wave *widths* alongside the centers still
   made the South zones worse (e.g. SW mean MAE 8.4k -> 14.3k) even though North
   improved. Cause: in the renewal equation, widening the window where R_t is elevated
   lets cases compound for more weeks before hitting the susceptible ceiling, which
   inflates the peak even though the amplitude parameter (`a2`) never moved - width
   isn't a "shape-only" change here, it leaks into magnitude through the model's own
   nonlinear dynamics. Restricting the shift to `t1`/`t2` alone (a pure time
   translation) avoids that coupling and is what's implemented below.

   **Postmortem, round 3 - why the full shift (`phase_weight=1.0`) still isn't shipped:**
   leave-one-season-out testing (all 3 train/test splits, not just the one above) shows
   the full shift only wins clearly when the held-out season's per-state timing looks
   like a typical average of the other two. It fails badly when held out on 2024/25:
   several states (e.g. Kano, Katsina, Taraba) have their second wave peak at epi-week 29
   in 2024/25 specifically - vs epi-week 16-19 for the same states in every other season -
   a season-specific anomaly (consistent with 2024/25 being flagged "HIGH, most severe
   since 2017-18" in nigeria_flu_season_reference.csv), not a stable zone characteristic.
   A zone offset trained on the other two (more typical) seasons has no way to see that
   coming, and even a small shift in the wrong direction hurts once a season breaks the
   pattern. With only 3 simulated seasons total, that's a real epistemic limit, not
   something more shrinkage can fix (a k-sweep in scripts/run_calibration_loso.py's
   sibling test made the 2024/25 failure *worse*, not better, since pulling every zone's
   offset toward a shared global mean is still a nonzero shift in the same wrong
   direction). Blending at `phase_weight=0.25` is the honest compromise given that.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.renewal import RenewalParams, fit_series, simulate, generation_kernel, N_SEED, _PARAM_SPEC

# Wave-center fields only - NOT widths (s1/s2_rise/s2_fall) or amplitude/ceiling
# fields (a1/a2/r_base/s_max). See module docstring's two-round postmortem: widths
# leak into magnitude through the renewal equation's compounding, and amplitude/
# ceiling fields need a per-state magnitude model this layer deliberately lacks.
SHAPE_FIELDS = ["t1", "t2"]
_FIELD_NAMES = [p[0] for p in _PARAM_SPEC]
_FIELD_BOUNDS = {p[0]: (p[1], p[2]) for p in _PARAM_SPEC}


def fit_state_with_shrinkage(
    weeks: np.ndarray,
    observed: np.ndarray,
    population: float,
    national_params: RenewalParams,
    shrinkage: float = 2.0,
) -> tuple[RenewalParams, np.ndarray]:
    """Fit one state's renewal params, shrunk toward the national fit.

    `shrinkage`: prior_weight passed to src.renewal.fit_series - higher pulls the
    state fit harder toward the national curve. Use `default_shrinkage` to derive
    it from population so small, noisier states shrink harder by default.
    """
    prior = national_params.as_array()
    return fit_series(weeks, observed, population, prior=prior, prior_weight=shrinkage)


def default_shrinkage(population: float, reference_population: float = 6_000_000, base: float = 1.5) -> float:
    """Smaller states get pulled harder toward the national fit: shrinkage scales
    with sqrt(reference_population / population), i.e. inverse to their expected
    signal-to-noise as a rough stand-in for a proper variance-based estimate."""
    return float(base * np.sqrt(reference_population / max(population, 1.0)))


def fit_all_states(
    by_state_season: pd.DataFrame,
    national_params: RenewalParams,
    shrinkage_fn=default_shrinkage,
) -> dict[str, tuple[RenewalParams, np.ndarray]]:
    """Fit every state in one season, each shrunk toward `national_params`."""
    results = {}
    for state, d in by_state_season.groupby("state"):
        d = d.sort_values("epi_week_of_season")
        weeks = d["epi_week_of_season"].to_numpy(dtype=float)
        observed = d["cases"].to_numpy(dtype=float)
        population = float(d["population"].iloc[0])
        shrinkage = shrinkage_fn(population)
        results[state] = fit_state_with_shrinkage(weeks, observed, population, national_params, shrinkage)
    return results


def population_proportional_allocation(
    national_sim: np.ndarray, state_population: float, national_population: float
) -> np.ndarray:
    """Naive baseline: split the national trajectory by population share alone."""
    return national_sim * (state_population / national_population)


def build_bias_table(
    by_state_seasons: pd.DataFrame,
    national_sims: dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (season, state): total-count log-bias of the naive population-
    proportional allocation against what was actually observed.

    `national_sims`: {season: national renewal-fit trajectory array}, weeks in the
    same order as `by_state_seasons`'s `epi_week_of_season` for that season.
    """
    meta_idx = meta.set_index("state")
    rows = []
    for season, d_season in by_state_seasons.groupby("season"):
        national_sim = national_sims[season]
        national_population = float(d_season.groupby("state")["population"].first().sum())
        for state, d in d_season.groupby("state"):
            d = d.sort_values("epi_week_of_season")
            observed_total = d["cases"].sum()
            population = float(d["population"].iloc[0])
            naive = population_proportional_allocation(national_sim, population, national_population)
            naive_total = naive.sum()
            log_bias = np.log((observed_total + 1) / (naive_total + 1))
            m = meta_idx.loc[state]
            rows.append({
                "season": season, "state": state, "zone": m["zone"], "climate": m["climate"],
                "hc_access": m["hc_access"], "population": population,
                "observed_total": observed_total, "naive_total": naive_total, "log_bias": log_bias,
            })
    return pd.DataFrame(rows)


def fit_bias_regression(bias_table: pd.DataFrame, formula: str = "log_bias ~ C(zone) + C(climate) + hc_access"):
    """OLS bias model: state-season log-bias ~ zone + climate + healthcare access.

    v0 stand-in for a full hierarchical/mixed-effects model (see module docstring).
    """
    return smf.ols(formula, data=bias_table).fit()


def predict_bias(model, meta_row: pd.Series) -> float:
    """Predicted log-bias for one state from its static metadata."""
    row = pd.DataFrame([{"zone": meta_row["zone"], "climate": meta_row["climate"], "hc_access": meta_row["hc_access"]}])
    return float(model.predict(row).iloc[0])


def allocate_national_to_states(
    national_sim: np.ndarray,
    meta: pd.DataFrame,
    bias_model=None,
) -> pd.DataFrame:
    """Calibrated per-state weekly trajectories from a national renewal trajectory.

    Population-proportional split, corrected by `bias_model`'s predicted log-bias
    per state if one is given (falls back to the naive population-only split
    otherwise). Works for any season - including held-out ones - since it only
    needs static state metadata, not that season's own state-level observations.
    """
    national_population = float(meta["population"].sum())
    out = {}
    for _, m in meta.iterrows():
        naive = population_proportional_allocation(national_sim, m["population"], national_population)
        if bias_model is not None:
            bias = predict_bias(bias_model, m)
            out[m["state"]] = naive * np.exp(bias)
        else:
            out[m["state"]] = naive
    return pd.DataFrame(out)


def build_shape_delta_table(
    state_fits: dict[str, dict[str, tuple[RenewalParams, np.ndarray]]],
    national_params: dict[str, RenewalParams],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Long-format table of (season, state, zone, field, delta) for the shape fields
    in SHAPE_FIELDS - each state's fitted value minus that season's national fit.

    `state_fits`: {season: {state: (params, sim)}} as returned by `fit_all_states`,
    called once per training season. `national_params`: {season: RenewalParams}.
    """
    zone_map = meta.set_index("state")["zone"].to_dict()
    rows = []
    for season, fits in state_fits.items():
        nat = national_params[season]
        for state, (params, _sim) in fits.items():
            for field in SHAPE_FIELDS:
                rows.append({
                    "season": season, "state": state, "zone": zone_map[state],
                    "field": field, "delta": getattr(params, field) - getattr(nat, field),
                })
    return pd.DataFrame(rows)


def fit_zone_shape_offsets(delta_table: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Empirical-Bayes-shrunk mean shape-parameter offset per (zone, field).

    Each zone's raw mean delta is shrunk toward the global mean delta for that field,
    weight = n_zone / (n_zone + k): zones with more training observations (more
    state-seasons) keep more of their own signal; zones with fewer are pulled harder
    toward the population-wide average. This is the "plain shrinkage estimator" v0
    fallback named in docs/DESIGN.md for the full hierarchical model.
    """
    global_mean = delta_table.groupby("field")["delta"].mean().rename("global_mean")
    zone_stats = delta_table.groupby(["zone", "field"])["delta"].agg(["mean", "count"]).reset_index()
    zone_stats = zone_stats.merge(global_mean, on="field")
    weight = zone_stats["count"] / (zone_stats["count"] + k)
    zone_stats["shrunk_delta"] = weight * zone_stats["mean"] + (1 - weight) * zone_stats["global_mean"]
    return zone_stats[["zone", "field", "shrunk_delta"]]


def zone_offsets_dict(shrunk_offsets: pd.DataFrame) -> dict[str, dict[str, float]]:
    """`fit_zone_shape_offsets` output reshaped to {zone: {field: shrunk_delta}}."""
    out: dict[str, dict[str, float]] = {}
    for _, row in shrunk_offsets.iterrows():
        out.setdefault(row["zone"], {})[row["field"]] = row["shrunk_delta"]
    return out


def apply_zone_offset(base_params: RenewalParams, zone: str, offsets: dict[str, dict[str, float]]) -> RenewalParams:
    """`base_params` (typically a season's national fit) with each SHAPE_FIELDS entry
    shifted by that zone's shrunk offset, clipped back into the field's valid bounds."""
    values = {f: getattr(base_params, f) for f in _FIELD_NAMES}
    for field, delta in offsets.get(zone, {}).items():
        lo, hi = _FIELD_BOUNDS[field]
        values[field] = float(np.clip(values[field] + delta, lo, hi))
    return RenewalParams(**values)


def simulate_zone_phase(
    national_params: RenewalParams,
    zone: str,
    offsets: dict[str, dict[str, float]],
    weeks: np.ndarray,
    national_seed_cases: np.ndarray,
    national_population: float,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """National-scale trajectory with this zone's timing/width offsets applied.

    Re-simulated from the *national* seed weeks and population (not the state's) -
    amplitude and susceptible-ceiling fields are untouched, so this stays at national
    magnitude; only when/how-wide the two waves are moves toward the zone's typical
    phase. `allocate_with_phase_shift` is what turns this into a per-state forecast.
    """
    if kernel is None:
        kernel = generation_kernel()
    zone_params = apply_zone_offset(national_params, zone, offsets)
    return simulate(zone_params, weeks, national_seed_cases[:N_SEED], national_population, kernel)


DEFAULT_PHASE_WEIGHT = 0.25  # see PHASE_WEIGHT note below


def allocate_with_phase_shift(
    national_params: RenewalParams,
    offsets: dict[str, dict[str, float]],
    meta: pd.DataFrame,
    weeks: np.ndarray,
    national_seed_cases: np.ndarray,
    national_population: float,
    kernel: np.ndarray | None = None,
    bias_model=None,
    phase_weight: float = DEFAULT_PHASE_WEIGHT,
) -> pd.DataFrame:
    """Per-state trajectories: a `phase_weight`-blend of the naive population-share
    allocation and the zone-phase-shifted one (optionally further corrected by
    `bias_model`'s magnitude bias). Needs nothing state-specific from the target
    season - zone/metadata and the national fit/seed are enough, so this works for
    held-out seasons exactly as it would for a genuine forecast.

    PHASE_WEIGHT NOTE - why this defaults to 0.25, not 1.0: `scripts/run_calibration.py`
    originally reported the *full* phase shift (`phase_weight=1.0`) hitting 97% peak-week
    accuracy vs the naive baseline's 81%, on the single split (train 2023/24+2024/25,
    test 2025/26). Running all 3 leave-one-season-out splits
    (`scripts/run_calibration_loso.py`) showed that result does not generalize: testing
    against the 2024/25 (high-severity, anomalously-timed) season instead, the full phase
    shift nearly *doubled* mean state MAE (8,366 -> 15,785) and peak-week accuracy fell to
    51%. A weight sweep at `phase_weight` in {0, .25, .5, .75, 1} across all 3 splits found
    0.25 keeps most of the upside on the 2 splits where the shift helps while capping the
    downside on 2024/25 to a ~9% MAE increase instead of ~89%. With only 3 seasons of
    training data, a zone offset fit on 2 of them can't be expected to capture a
    genuinely anomalous third season, so this stays deliberately conservative - see
    docs/DESIGN.md's "Calibration layer v1" section for the full sweep table and the
    epidemiological read on *why* 2024/25 breaks the naive zone-offset assumption.
    """
    if kernel is None:
        kernel = generation_kernel()
    zone_sims = {
        zone: simulate_zone_phase(national_params, zone, offsets, weeks, national_seed_cases, national_population, kernel)
        for zone in meta["zone"].unique()
    }
    national_sim_full = simulate(national_params, weeks, national_seed_cases[:N_SEED], national_population, kernel)
    out = {}
    for _, m in meta.iterrows():
        naive_share = population_proportional_allocation(national_sim_full, m["population"], national_population)
        phase_share = population_proportional_allocation(zone_sims[m["zone"]], m["population"], national_population)
        share = (1 - phase_weight) * naive_share + phase_weight * phase_share
        if bias_model is not None:
            share = share * np.exp(predict_bias(bias_model, m))
        out[m["state"]] = share
    return pd.DataFrame(out)
