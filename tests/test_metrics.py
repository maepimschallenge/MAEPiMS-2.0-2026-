import numpy as np
from src.metrics import (
    mae, rmse, quantile_loss, weighted_interval_score, crps_from_quantiles,
    peak_week_accuracy, peak_incidence_pct_error, pit_values,
)


def test_mae_rmse_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0
    assert rmse(y, y) == 0


def test_quantile_loss_zero_for_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    assert quantile_loss(y, y, q=0.5) == 0
    assert quantile_loss(y, y, q=0.9) == 0


def test_quantile_loss_penalizes_correct_direction():
    y_true = np.array([10.0])
    # underforecasting should cost more at high quantiles than at low quantiles
    under = np.array([5.0])
    loss_q10 = quantile_loss(y_true, under, q=0.1)
    loss_q90 = quantile_loss(y_true, under, q=0.9)
    assert loss_q90 > loss_q10


def test_wis_zero_for_perfect_forecast():
    y = np.array([100.0, 200.0])
    median = y.copy()
    qpreds = {0.1: y.copy(), 0.9: y.copy(), 0.25: y.copy(), 0.75: y.copy()}
    assert weighted_interval_score(y, median, qpreds) == 0


def test_crps_nonnegative():
    y = np.array([5.0, 10.0])
    qpreds = {0.1: np.array([3.0, 8.0]), 0.5: np.array([5.0, 9.0]), 0.9: np.array([7.0, 12.0])}
    assert crps_from_quantiles(y, qpreds) >= 0


def test_peak_week_accuracy():
    true_wk = np.array([14, 16, 15])
    pred_wk = np.array([14, 15, 18])
    assert peak_week_accuracy(true_wk, pred_wk, tolerance=1) == 2 / 3
    assert peak_week_accuracy(true_wk, pred_wk, tolerance=0) == 1 / 3


def test_peak_incidence_pct_error_zero_when_exact():
    assert peak_incidence_pct_error(np.array([100.0]), np.array([100.0])) == 0


def test_pit_values_in_unit_interval():
    y = np.array([5.0, 15.0])
    qpreds = {0.1: np.array([1.0, 10.0]), 0.5: np.array([5.0, 15.0]), 0.9: np.array([9.0, 20.0])}
    pit = pit_values(y, qpreds)
    assert np.all((pit >= 0) & (pit <= 1))
