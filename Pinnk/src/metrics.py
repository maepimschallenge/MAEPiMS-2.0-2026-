"""
Evaluation metrics for the MAEPiMS Challenge 2026.

Covers every metric named in the challenge brief's "Evaluation Criteria":
MAE, RMSE, Weighted Interval Score (WIS), a quantile-based CRPS approximation,
quantile (pinball) loss, peak-week accuracy, peak-incidence accuracy, and a
PIT-based calibration check for the probabilistic forecasts.

All functions take plain numpy arrays / pandas Series so they can score any
layer of the pipeline (mechanistic-only, calibrated, or full ensemble) the
same way.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def quantile_loss(y_true, y_pred, q: float) -> float:
    """Pinball loss for a single quantile level q in (0, 1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def mean_quantile_loss(y_true, quantile_preds: dict[float, np.ndarray]) -> float:
    """Average pinball loss across a dict of {quantile_level: predictions}."""
    losses = [quantile_loss(y_true, preds, q) for q, preds in quantile_preds.items()]
    return float(np.mean(losses))


def interval_score(y_true, lower, upper, alpha: float) -> float:
    """Score for a single (1 - alpha) central prediction interval (Gneiting & Raftery 2007).

    alpha=0.5 -> 50% PI, alpha=0.1 -> 90% PI.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    penalty_lo = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_hi = (2 / alpha) * np.maximum(y_true - upper, 0)
    return float(np.mean(width + penalty_lo + penalty_hi))


def weighted_interval_score(y_true, median, quantile_preds: dict[float, np.ndarray]) -> float:
    """WIS via its standard decomposition into a set of central prediction intervals.

    `quantile_preds` must contain symmetric pairs around 0.5, e.g. keys
    {0.05, 0.1, 0.25, 0.75, 0.9, 0.95} plus the median prediction passed separately.
    Reduces to the mean pinball loss formulation of Bracher et al. (2021).
    """
    y_true = np.asarray(y_true, dtype=float)
    median = np.asarray(median, dtype=float)
    levels = sorted(q for q in quantile_preds if q < 0.5)
    K = len(levels)
    if K == 0:
        return float(np.mean(np.abs(y_true - median)))

    total = 0.5 * np.abs(y_true - median)
    for lo_q in levels:
        hi_q = round(1 - lo_q, 10)
        if hi_q not in quantile_preds:
            continue
        alpha = 2 * lo_q
        lower, upper = quantile_preds[lo_q], quantile_preds[hi_q]
        width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
        penalty_lo = np.maximum(np.asarray(lower, dtype=float) - y_true, 0)
        penalty_hi = np.maximum(y_true - np.asarray(upper, dtype=float), 0)
        iscore = (alpha / 2) * width + penalty_lo + penalty_hi
        total = total + (alpha / 2) * iscore
    return float(np.mean(total / (K + 0.5)))


def crps_from_quantiles(y_true, quantile_preds: dict[float, np.ndarray]) -> float:
    """CRPS approximated as 2x the mean pinball loss over a dense quantile grid
    (exact in the limit of infinitely many quantile levels; see Gneiting & Raftery 2007,
    eq. 8, and the pinball-loss/CRPS equivalence used by e.g. the CDC FluSight scoring)."""
    return 2 * mean_quantile_loss(y_true, quantile_preds)


def peak_week_accuracy(true_peak_week, pred_peak_week, tolerance: int = 1) -> float:
    """Fraction of series whose predicted peak week is within `tolerance` weeks of truth."""
    true_peak_week = np.asarray(true_peak_week)
    pred_peak_week = np.asarray(pred_peak_week)
    return float(np.mean(np.abs(true_peak_week - pred_peak_week) <= tolerance))


def peak_incidence_pct_error(true_peak_value, pred_peak_value) -> float:
    """Mean absolute percentage error on peak incidence magnitude."""
    true_peak_value = np.asarray(true_peak_value, dtype=float)
    pred_peak_value = np.asarray(pred_peak_value, dtype=float)
    return float(np.mean(np.abs(pred_peak_value - true_peak_value) / np.maximum(true_peak_value, 1e-9)) * 100)


def pit_values(y_true, quantile_preds: dict[float, np.ndarray]) -> np.ndarray:
    """Probability-integral-transform-ish calibration diagnostic: for each observation,
    the fraction of the fitted quantile grid that falls below the true value. A
    well-calibrated model gives PIT values ~ Uniform(0, 1) in aggregate."""
    y_true = np.asarray(y_true, dtype=float)
    levels = sorted(quantile_preds)
    grid = np.stack([np.asarray(quantile_preds[q], dtype=float) for q in levels], axis=0)
    return np.mean(grid <= y_true[None, :], axis=0)


def score_all(y_true, median, quantile_preds: dict[float, np.ndarray] | None = None) -> dict:
    """Convenience wrapper: point-forecast metrics, plus probabilistic ones if quantiles given."""
    out = {"MAE": mae(y_true, median), "RMSE": rmse(y_true, median)}
    if quantile_preds:
        out["WIS"] = weighted_interval_score(y_true, median, quantile_preds)
        out["CRPS_approx"] = crps_from_quantiles(y_true, quantile_preds)
        out["mean_quantile_loss"] = mean_quantile_loss(y_true, quantile_preds)
        if 0.25 in quantile_preds and 0.75 in quantile_preds:
            out["IS_50"] = interval_score(y_true, quantile_preds[0.25], quantile_preds[0.75], alpha=0.5)
        if 0.05 in quantile_preds and 0.95 in quantile_preds:
            out["IS_90"] = interval_score(y_true, quantile_preds[0.05], quantile_preds[0.95], alpha=0.1)
    return out
