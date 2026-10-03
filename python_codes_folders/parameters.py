"""
parameters.py - fixed (assumed) natural-history parameters and the
specification of estimated parameters. Time unit: days.

Fixed values are MODELLING ASSUMPTIONS (report section "Assumptions"); they
are not taken from any influenza dataset. Each one is varied in the
sensitivity analysis (src/sensitivity.py). Demographic values are set so that
the population is constant, which matches the supplied data (one population
figure for all three seasons).
"""
N0 = 229_501_935            # national population in the supplied data
LIFE_EXPECTANCY_YEARS = 55.0

FIXED = {
    # demography (constant population: Lambda = mu N0)
    "mu": 1.0 / (LIFE_EXPECTANCY_YEARS * 365.0),
    # natural history
    "sigma": 1.0 / 2.0,      # mean latent period 2 days
    "theta": 0.30,           # fraction of infections that are asymptomatic
    "gamma_A": 1.0 / 4.0,    # asymptomatic infectious period 4 days
    "gamma_I": 1.0 / 5.0,    # symptomatic community recovery, 5 days
    "gamma_H": 1.0 / 7.0,    # hospital recovery, 7 days
    "gamma_T": 1.0 / 4.0,    # recovery on antiviral therapy, 4 days
    "gamma_Q": 1.0 / 5.0,    # recovery in isolation, 5 days
    # interventions (low antiviral and isolation coverage)
    "alpha_A": 0.001, "alpha": 0.02, "alpha_H": 0.10,
    "phi_A": 0.001, "phi_I": 0.05, "phi_H": 0.05,
    # relative infectiousness (A is the reference: coefficient 1 in lambda)
    "eta_I": 1.0, "eta_H": 0.2, "eta_T": 0.5, "eta_Q": 0.1,
    # vaccination and immunity
    "eps_V": 0.60,           # vaccine efficacy
    "omega": 1.0e-5,         # vaccination rate (very low coverage)
    "xi": 1.0 / 180.0,       # waning of vaccine protection, 6 months
    "kappa": 1.0 / 365.0,    # waning of infection-acquired immunity, 1 year
    # delta_H and delta_I are not separately identifiable from national data
    # (estimated correlation -0.94, unbounded interval in the first fit), so the
    # in-hospital fatality delta_H / K_H is fixed at 5% and delta_I is estimated.
    "delta_H": 0.05 * (1.0 / 7.0 + 0.10 + 1.0 / (LIFE_EXPECTANCY_YEARS * 365.0)) / 0.95,
}
import os as _os
if _os.environ.get("KAPPA_YEARS"):          # sensitivity refit: immunity duration in years
    FIXED["kappa"] = 1.0 / (float(_os.environ["KAPPA_YEARS"]) * 365.0)
FIXED["Lambda"] = FIXED["mu"] * N0

# ---------------------------------------------------------------------------
# Seasonal forcing (P2, option B): beta(t) = beta_0 f(t), f(t) = exp(g(t))/mean,
# g a periodic cubic spline with N_KNOTS equally spaced knots (about monthly,
# knot 0 = season start = first Sunday of July). g_0 = 0 fixes the level
# (the level is carried by beta_0). Season s adds h_s to the knots inside the
# Harmattan window (early Dec - early Feb), so the Harmattan intensity can
# differ between seasons; h = 0 for the reference season.
# ---------------------------------------------------------------------------
N_KNOTS = 12
KNOT_DAYS = [365.0 * k / N_KNOTS for k in range(N_KNOTS)]
HARMATTAN_KNOTS = [5, 6, 7]          # days ~152, 183, 213 after 1 July
_G_GUESS = [0.003, -0.23, -0.087, -0.043, 0.475, 1.6, 0.482, -1.221, -2.461, -0.97, 0.208]

# Estimated parameters: name -> (initial guess, lower, upper, scale)
SHARED_SPEC = {
    "beta_0":  (1.262, 0.05, 5.0, "log"),     # mean transmission rate (per day), reference season
    **{f"g_{k}": (_G_GUESS[k - 1], -4.0, 4.0, "lin") for k in range(1, N_KNOTS)},
    "p_c":     (0.1357, 0.01, 1.0, "log"),     # reporting fraction of symptomatic infections
    "rho":     (0.0004919, 1e-5, 0.2, "log"),     # hospitalisation rate of I (per day)
    "delta_I": (2.347e-05, 1e-7, 0.05, "log"),    # disease death rate, community (per day)
    # seed size and early-season forcing are almost perfectly confounded
    # (|corr| ~ 0.98 when E0 is season-specific), so one E0 is shared by all seasons
    "E0":      (20.0, 1.0, 1e6, "log"),      # exposed seeds at season start
}
# P-spline smoothness penalty (Eilers & Marx, 1996, Stat. Sci. 11:89-121):
# residual rows SPLINE_PENALTY * (g_{k-1} - 2 g_k + g_{k+1}), periodic in k.
SPLINE_PENALTY = 2.0
SEASON_SPEC = {
    "m":    (0.65, 0.3, 3.0, "log"),         # season transmission multiplier (1 for reference)
    "h":    (0.00, -3.0, 3.0, "lin"),        # season Harmattan shift of knots 5-7 (0 for reference)
    "s0":   (0.75, 0.02, 1.0, "lin"),        # susceptible fraction at season start
    "B_c":  (2e4, 1.0, 2e5, "log"),          # background weekly reported cases
    "B_h":  (300.0, 0.1, 5e3, "log"),        # background weekly admissions
    "B_d":  (30.0, 0.01, 5e2, "log"),        # background weekly deaths
}
REFERENCE_ONLY_FIXED = {"m": 1.0, "h": 0.0}  # values for the reference season
SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
ALL_SEASONS = list(SEASONS)
if _os.environ.get("TRAIN_SEASONS"):       # time-ordered validation: fit on a subset
    SEASONS = [x.strip() for x in _os.environ["TRAIN_SEASONS"].split(",")]
REFERENCE_SEASON = "2023/2024"   # its multiplier m is fixed at 1 (identifiability)

