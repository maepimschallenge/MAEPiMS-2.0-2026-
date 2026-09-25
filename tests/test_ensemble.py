import numpy as np
from src.ensemble import (
    probability_of_rising, probability_peak_within, peak_week_distribution,
    apply_rate_ratio, aggregate_via_monte_carlo, _sample_from_quantiles,
)


def test_probability_of_rising_all_rising():
    trajectories = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)
    assert probability_of_rising(trajectories, 0) == 1.0


def test_probability_of_rising_mixed():
    trajectories = np.array([[1, 2], [1, 0], [1, 2], [1, 0]], dtype=float)
    assert probability_of_rising(trajectories, 0) == 0.5


def test_probability_peak_within_matches_argmax_window():
    weeks = np.array([1, 2, 3, 4, 5], dtype=float)
    trajectories = np.array([
        [0, 0, 5, 0, 0],   # peak at week 3
        [0, 0, 0, 0, 5],   # peak at week 5
        [0, 5, 0, 0, 0],   # peak at week 2
    ], dtype=float)
    assert np.isclose(probability_peak_within(trajectories, weeks, 2, 3), 2 / 3)


def test_peak_week_distribution_sums_to_one_and_covers_all_weeks():
    weeks = np.array([1, 2, 3], dtype=float)
    trajectories = np.array([[0, 5, 0], [5, 0, 0], [0, 5, 0]], dtype=float)
    dist = peak_week_distribution(trajectories, weeks)
    assert np.isclose(dist.sum(), 1.0)
    assert set(dist.index) == {1.0, 2.0, 3.0}
    assert np.isclose(dist.loc[2.0], 2 / 3)


def test_apply_rate_ratio_scales_and_preserves_order():
    state_q = {"Lagos": {0.1: np.array([10.0, 20.0]), 0.5: np.array([20.0, 40.0]), 0.9: np.array([30.0, 60.0])}}
    scaled = apply_rate_ratio(state_q, 0.02)
    assert np.allclose(scaled["Lagos"][0.5], [0.4, 0.8])
    assert scaled["Lagos"][0.1][0] <= scaled["Lagos"][0.5][0] <= scaled["Lagos"][0.9][0]


def test_sample_from_quantiles_median_recovered_approximately():
    rng = np.random.default_rng(0)
    qdict = {0.1: np.array([80.0]), 0.5: np.array([100.0]), 0.9: np.array([120.0])}
    samples = _sample_from_quantiles(qdict, n_samples=20000, rng=rng)
    assert abs(np.median(samples[:, 0]) - 100.0) < 2.0


def test_aggregate_via_monte_carlo_mean_close_to_sum_of_medians():
    rng = np.random.default_rng(0)
    group = {
        "A": {0.1: np.array([8.0]), 0.5: np.array([10.0]), 0.9: np.array([12.0])},
        "B": {0.1: np.array([18.0]), 0.5: np.array([20.0]), 0.9: np.array([22.0])},
    }
    out = aggregate_via_monte_carlo(group, [0.1, 0.5, 0.9], n_samples=20000, rng=rng)
    assert abs(out[0.5][0] - 30.0) < 1.5
    assert out[0.1][0] < out[0.5][0] < out[0.9][0]
