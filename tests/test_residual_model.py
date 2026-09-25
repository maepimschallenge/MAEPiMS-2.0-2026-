import numpy as np
from src.residual_model import enforce_monotonic_quantiles, inflate_quantiles, reconstruct_quantile_forecast


def test_enforce_monotonic_quantiles_sorts_crossed_predictions():
    # q=0.25's raw prediction (5) incorrectly exceeds q=0.75's (3) at this point
    crossed = {0.25: np.array([5.0, 1.0]), 0.5: np.array([2.0, 2.0]), 0.75: np.array([3.0, 4.0])}
    fixed = enforce_monotonic_quantiles(crossed)
    for i in range(2):
        assert fixed[0.25][i] <= fixed[0.5][i] <= fixed[0.75][i]


def test_inflate_quantiles_keeps_median_fixed_and_widens_symmetrically():
    q = {0.1: np.array([8.0]), 0.5: np.array([10.0]), 0.9: np.array([12.0])}
    inflated = inflate_quantiles(q, factor=2.0)
    assert inflated[0.5][0] == 10.0
    assert inflated[0.1][0] == 6.0   # 10 - 2*(10-8)
    assert inflated[0.9][0] == 14.0  # 10 + 2*(12-10)


def test_inflate_quantiles_factor_one_is_identity():
    q = {0.1: np.array([8.0]), 0.5: np.array([10.0]), 0.9: np.array([12.0])}
    assert inflate_quantiles(q, factor=1.0) is q


def test_inflate_quantiles_never_negative():
    q = {0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}
    inflated = inflate_quantiles(q, factor=10.0)
    assert inflated[0.1][0] >= 0


def test_reconstruct_quantile_forecast_median_reproduces_baseline_when_log_ratio_zero():
    baseline = np.array([100.0, 200.0])
    log_ratio_q = {0.25: np.zeros(2), 0.5: np.zeros(2), 0.75: np.zeros(2)}
    out = reconstruct_quantile_forecast(baseline, log_ratio_q, interval_inflation=1.0)
    assert np.allclose(out[0.5], baseline)


def test_reconstruct_quantile_forecast_is_nonnegative_and_nondecreasing_in_quantile():
    baseline = np.array([50.0, 75.0])
    log_ratio_q = {0.1: np.array([-0.5, 2.0]), 0.5: np.array([0.0, 0.0]), 0.9: np.array([0.5, -1.0])}
    out = reconstruct_quantile_forecast(baseline, log_ratio_q, interval_inflation=1.0)
    for i in range(2):
        assert out[0.1][i] <= out[0.5][i] <= out[0.9][i]
        assert out[0.1][i] >= 0
