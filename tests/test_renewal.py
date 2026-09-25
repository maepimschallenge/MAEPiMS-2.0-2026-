import numpy as np
from src.renewal import generation_kernel, gaussian_bump, fit_series


def test_generation_kernel_sums_to_one():
    k = generation_kernel()
    assert np.isclose(k.sum(), 1.0)
    assert np.all(k > 0)


def test_gaussian_bump_peaks_at_center():
    t = np.arange(0, 30)
    bump = gaussian_bump(t, center=14, width=2)
    assert t[bump.argmax()] == 14
    assert bump.max() == 1.0


def test_fit_series_recovers_a_double_peak_shape():
    """Sanity check on synthetic data: the renewal core should hit the
    approximate location and scale of two well-separated Gaussian-like peaks."""
    weeks = np.arange(1, 53, dtype=float)
    synthetic = (
        50 + 5000 * np.exp(-0.5 * ((weeks - 15) / 2) ** 2)
        + 2000 * np.exp(-0.5 * ((weeks - 28) / 2) ** 2)
    )
    params, sim = fit_series(weeks, synthetic, population=1_000_000)
    true_peak_week = weeks[synthetic.argmax()]
    fitted_peak_week = weeks[sim.argmax()]
    assert abs(true_peak_week - fitted_peak_week) <= 2
    # fitted peak magnitude within 30% of true peak magnitude
    assert abs(sim.max() - synthetic.max()) / synthetic.max() < 0.3
