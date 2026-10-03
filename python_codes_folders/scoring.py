"""
scoring.py - evaluation metrics named in the MAEPiMS 2026 guidelines.

Point:       MAE, RMSE (on the forecast median)
Quantile:    quantile (pinball) loss, interval score, Weighted Interval Score
             (WIS; Bracher et al., 2021, PLoS Comput Biol 17(2):e1008618)
Sample:      CRPS via the energy form E|X-y| - 0.5 E|X-X'|
             (Gneiting & Raftery, 2007, JASA 102(477):359-378)
Seasonal:    peak-week error, peak-incidence and cumulative-burden relative error
Calibration: empirical coverage of central prediction intervals, PIT

All functions take plain NumPy arrays so they can be reused for national,
regional and state series.
"""
from __future__ import annotations

import numpy as np

# 23 quantile levels (includes median, 50% and 90% central intervals)
QUANTILES = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                      0.95, 0.975, 0.99])


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def quantile_loss(y, qpred, taus=QUANTILES):
    """Mean pinball loss. y: (T,), qpred: (T, K) in the order of taus."""
    y = np.asarray(y)[:, None]
    q = np.asarray(qpred)
    u = y - q
    return float(np.mean(np.maximum(taus * u, (taus - 1) * u)))


def interval_score(y, lower, upper, alpha):
    y, lo, up = map(np.asarray, (y, lower, upper))
    return (up - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - up) * (y > up)


def wis(y, qpred, taus=QUANTILES):
    """Weighted interval score per time point (Bracher et al., 2021).
    Requires symmetric quantile levels around the median."""
    y = np.asarray(y)
    q = np.asarray(qpred)
    taus = np.asarray(taus)
    med = q[:, np.argmin(np.abs(taus - 0.5))]
    lower_taus = taus[taus < 0.5]
    K = len(lower_taus)
    total = 0.5 * np.abs(y - med)
    for tl in lower_taus:
        alpha = 2 * tl
        il = np.argmin(np.abs(taus - tl))
        iu = np.argmin(np.abs(taus - (1 - tl)))
        total = total + (alpha / 2) * interval_score(y, q[:, il], q[:, iu], alpha)
    return total / (K + 0.5)


def crps_samples(y, samples):
    """CRPS per time point from predictive samples. samples: (T, M)."""
    y = np.asarray(y)[:, None]
    x = np.asarray(samples)
    term1 = np.mean(np.abs(x - y), axis=1)
    xs = np.sort(x, axis=1)
    M = xs.shape[1]
    # E|X-X'| computed in O(M log M) with sorted samples
    i = np.arange(1, M + 1)
    term2 = np.sum((2 * i - M - 1) * xs, axis=1) * 2 / (M * M)
    return term1 - 0.5 * term2


def coverage(y, lower, upper):
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))


def seasonal_errors(y_obs, y_pred_median, population=None):
    """Peak week (0-based index difference), relative peak-incidence error,
    relative cumulative-burden error, attack-rate error (if population)."""
    y_obs, y_pred = np.asarray(y_obs), np.asarray(y_pred_median)
    out = {
        "peak_week_error": int(np.argmax(y_pred) - np.argmax(y_obs)),
        "peak_incidence_rel_error": float((y_pred.max() - y_obs.max()) / y_obs.max()),
        "cumulative_rel_error": float((y_pred.sum() - y_obs.sum()) / y_obs.sum()),
    }
    if population is not None:
        out["attack_rate_error_pct_points"] = float(100 * (y_pred.sum() - y_obs.sum()) / population)
    return out


def score_quantile_forecast(y, qpred, taus=QUANTILES):
    """Convenience bundle for one series."""
    taus = np.asarray(taus)
    med = qpred[:, np.argmin(np.abs(taus - 0.5))]
    i25, i75 = np.argmin(np.abs(taus - 0.25)), np.argmin(np.abs(taus - 0.75))
    i05, i95 = np.argmin(np.abs(taus - 0.05)), np.argmin(np.abs(taus - 0.95))
    return {
        "MAE": mae(y, med), "RMSE": rmse(y, med),
        "QuantileLoss": quantile_loss(y, qpred, taus),
        "WIS": float(np.mean(wis(y, qpred, taus))),
        "Coverage50": coverage(y, qpred[:, i25], qpred[:, i75]),
        "Coverage90": coverage(y, qpred[:, i05], qpred[:, i95]),
    }


if __name__ == "__main__":
    # self-test: a perfectly calibrated Gaussian forecast
    rng = np.random.default_rng(1)
    from scipy.stats import norm
    T = 20000
    y = rng.normal(size=T)
    q = np.tile(norm.ppf(QUANTILES), (T, 1))
    s = score_quantile_forecast(y, q)
    samp = rng.normal(size=(200, 4000))
    crps_mc = crps_samples(y[:200], samp).mean()
    crps_exact = np.mean(y[:200] * (2 * norm.cdf(y[:200]) - 1) + 2 * norm.pdf(y[:200]) - 1 / np.sqrt(np.pi))
    print(s)
    print(f"CRPS sample={crps_mc:.4f} exact={crps_exact:.4f}")
    assert abs(s["Coverage50"] - 0.5) < 0.02 and abs(s["Coverage90"] - 0.9) < 0.02
    assert abs(crps_mc - crps_exact) < 0.01
    print("scoring self-test passed")
