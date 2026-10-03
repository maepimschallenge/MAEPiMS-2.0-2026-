"""
model.py - SVEAITHQR influenza model (nine state variables) with the
observation bookkeeping integrals approved in proposal P1.

Time unit: days. t = 0 is the Sunday that starts epi week 1 of a season.

Force of infection (as established, coefficient 1 on A, i.e. eta_A = 1):
    lambda(t) = beta(t)/N * (A + eta_I I + eta_H H + eta_T T + eta_Q Q)
Seasonal forcing (proposal P2, option A: second harmonic added to beta only):
    beta(t) = beta_0 [1 + delta_c cos(2 pi (t - phi)/365) + delta_2 cos(4 pi (t - phi_2)/365)]
beta_0 is the time-average of beta(t), so R_0 and all stability results of the
autonomous model are unchanged; delta_2 = 0 recovers the original forcing.

Bookkeeping integrals (not compartments; they do not feed back):
    C_I' = (1 - theta) sigma E      new symptomatic infections
    C_H' = rho I                    new hospital admissions
    C_D' = delta_I I + delta_H H    new influenza deaths

Libraries: NumPy (Harris et al., 2020), SciPy solve_ivp (Virtanen et al., 2020).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

STATE_NAMES = ["S", "V", "E", "A", "I", "H", "T", "Q", "R"]
AUX_NAMES = ["C_I", "C_H", "C_D"]
YEAR = 365.0


def make_spline_forcing(knot_values, n_grid=3651):
    """Periodic forcing f(t) = exp(g(t)) / mean(exp(g)), with g a periodic cubic
    spline through equally spaced knots over one year (P2, option B).
    f > 0 everywhere and its annual mean is exactly 1, so beta_0 remains the
    time-average of beta(t) and all autonomous threshold results still apply."""
    from scipy.interpolate import CubicSpline
    K = len(knot_values)
    tk = np.linspace(0.0, YEAR, K + 1)
    g = CubicSpline(tk, np.append(knot_values, knot_values[0]), bc_type="periodic")
    grid = np.linspace(0.0, YEAR, n_grid)
    vals = np.exp(g(grid))
    vals = vals / (np.trapezoid(vals, grid) / YEAR)
    # tabulated on a 0.1-day grid and linearly interpolated (speed; error < 1e-4)
    return lambda t: np.interp(np.mod(t, YEAR), grid, vals)


def beta_t(t, p):
    if "forcing_fn" in p:                         # spline forcing (option B)
        return p["beta_0"] * p["forcing_fn"](t)
    return p["beta_0"] * (1.0
                          + p["delta_c"] * np.cos(2 * np.pi * (t - p["phi"]) / YEAR)
                          + p.get("delta_2", 0.0) * np.cos(4 * np.pi * (t - p.get("phi_2", 0.0)) / YEAR))


def min_forcing_factor(p, n=2000):
    """Minimum of beta(t)/beta_0 over one year (must be >= 0)."""
    t = np.linspace(0, YEAR, n)
    return float(np.min(beta_t(t, p) / p["beta_0"]))


def rhs(t, y, p):
    S, V, E, A, I, H, T, Q, R = y[:9]
    N = S + V + E + A + I + H + T + Q + R
    lam = beta_t(t, p) / N * (A + p["eta_I"] * I + p["eta_H"] * H + p["eta_T"] * T + p["eta_Q"] * Q)
    mu = p["mu"]
    K_A = p["gamma_A"] + p["alpha_A"] + p["phi_A"] + mu
    K_I = p["gamma_I"] + p["rho"] + p["alpha"] + p["phi_I"] + p["delta_I"] + mu
    K_H = p["gamma_H"] + p["alpha_H"] + p["delta_H"] + mu
    K_T = p["gamma_T"] + mu
    K_Q = p["gamma_Q"] + mu
    dS = p["Lambda"] + p["kappa"] * R + p["xi"] * V - (lam + p["omega"] + mu) * S
    dV = p["omega"] * S - ((1 - p["eps_V"]) * lam + p["xi"] + mu) * V
    dE = lam * S + (1 - p["eps_V"]) * lam * V - (p["sigma"] + mu) * E
    dA = p["theta"] * p["sigma"] * E - K_A * A
    dI = (1 - p["theta"]) * p["sigma"] * E - K_I * I
    dH = p["rho"] * I - K_H * H
    dT = p["alpha_A"] * A + p["alpha"] * I + p["alpha_H"] * H - K_T * T
    dQ = p["phi_A"] * A + p["phi_I"] * I + p["phi_H"] * H - K_Q * Q
    dR = (p["gamma_A"] * A + p["gamma_I"] * I + p["gamma_H"] * H + p["gamma_T"] * T
          + p["gamma_Q"] * Q - (p["kappa"] + mu) * R)
    dCI = (1 - p["theta"]) * p["sigma"] * E
    dCH = p["rho"] * I
    dCD = p["delta_I"] * I + p["delta_H"] * H
    return [dS, dV, dE, dA, dI, dH, dT, dQ, dR, dCI, dCH, dCD]


def dfe(p):
    m0 = p["omega"] + p["xi"] + p["mu"]
    S0 = p["Lambda"] * (p["xi"] + p["mu"]) / (p["mu"] * m0)
    V0 = p["Lambda"] * p["omega"] / (p["mu"] * m0)
    return S0, V0


def initial_state(p, N0, s0, E0):
    """Season start: a fraction s0 of the non-vaccinated population is
    susceptible (the rest carries immunity from earlier seasons, placed in R);
    V at its disease-free proportion; E0 exposed seeds; other infected = 0."""
    S_dfe, V_dfe = dfe(p)
    v_frac = V_dfe / (S_dfe + V_dfe)
    V = v_frac * N0
    S = s0 * (N0 - V - E0)
    R = (1 - s0) * (N0 - V - E0)
    return np.array([S, V, E0, 0, 0, 0, 0, 0, R, 0, 0, 0], dtype=float)


def simulate_weekly(p, y0, n_weeks=52):
    """Weekly increments of the bookkeeping integrals: returns dict of arrays
    (length n_weeks) for symptomatic incidence, admissions and deaths, plus the
    solution at week boundaries."""
    tb = 7.0 * np.arange(n_weeks + 1)
    sol = solve_ivp(rhs, (0, tb[-1]), y0, t_eval=tb, args=(p,), method="LSODA",
                    rtol=1e-7, atol=1e-6)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y
    return {"inc_I": np.diff(Y[9]), "adm_H": np.diff(Y[10]), "deaths": np.diff(Y[11]), "Y": Y, "t": tb}


# ---------------------------------------------------------------------------
# Threshold quantities (autonomous model, beta = beta_0)
# ---------------------------------------------------------------------------
def rates(p):
    mu = p["mu"]
    return dict(
        K_A=p["gamma_A"] + p["alpha_A"] + p["phi_A"] + mu,
        K_I=p["gamma_I"] + p["rho"] + p["alpha"] + p["phi_I"] + p["delta_I"] + mu,
        K_H=p["gamma_H"] + p["alpha_H"] + p["delta_H"] + mu,
        K_T=p["gamma_T"] + mu, K_Q=p["gamma_Q"] + mu,
        m0=p["omega"] + p["xi"] + mu,
        m1=p["xi"] + mu + (1 - p["eps_V"]) * p["omega"])


def R0_closed_form(p, eta_A=1.0):
    """Equation eq:flu_R0 with eta_A = 1 (lambda as written)."""
    r = rates(p)
    K_A, K_I, K_H, K_T, K_Q = r["K_A"], r["K_I"], r["K_H"], r["K_T"], r["K_Q"]
    th, s, mu = p["theta"], p["sigma"], p["mu"]
    asym = eta_A / K_A + p["eta_T"] * p["alpha_A"] / (K_A * K_T) + p["eta_Q"] * p["phi_A"] / (K_A * K_Q)
    sym = (p["eta_I"] / K_I + p["eta_H"] * p["rho"] / (K_I * K_H)
           + (p["eta_T"] / K_T) * (p["alpha"] / K_I + p["alpha_H"] * p["rho"] / (K_I * K_H))
           + (p["eta_Q"] / K_Q) * (p["phi_I"] / K_I + p["phi_H"] * p["rho"] / (K_I * K_H)))
    return p["beta_0"] * r["m1"] / r["m0"] * s / (s + mu) * (th * asym + (1 - th) * sym)


def ngm_matrices(p, eta_A=1.0):
    """F and V for X = (E, A, I, H, T, Q) at the DFE (eq:R0_rank1)."""
    r = rates(p)
    b0 = p["beta_0"] * r["m1"] / r["m0"]
    c = np.array([0, eta_A, p["eta_I"], p["eta_H"], p["eta_T"], p["eta_Q"]])
    F = np.zeros((6, 6))
    F[0, :] = b0 * c
    V = np.diag([p["sigma"] + p["mu"], r["K_A"], r["K_I"], r["K_H"], r["K_T"], r["K_Q"]])
    V[1, 0] = -p["theta"] * p["sigma"]
    V[2, 0] = -(1 - p["theta"]) * p["sigma"]
    V[3, 2] = -p["rho"]
    V[4, 1], V[4, 2], V[4, 3] = -p["alpha_A"], -p["alpha"], -p["alpha_H"]
    V[5, 1], V[5, 2], V[5, 3] = -p["phi_A"], -p["phi_I"], -p["phi_H"]
    return F, V, c


def R0_ngm(p):
    F, V, _ = ngm_matrices(p)
    return float(np.max(np.abs(np.linalg.eigvals(F @ np.linalg.inv(V)))))


def R0bar(p):
    """GAS threshold: R0bar = R0 * m0/m1 >= R0."""
    r = rates(p)
    return R0_closed_form(p) * r["m0"] / r["m1"]


def endemic_quadratic(p, eta_A=1.0):
    """Coefficients a2, a1, a0 of the equation for lambda* (endemic equilibrium)."""
    r = rates(p)
    s, mu, th = p["sigma"], p["mu"], p["theta"]
    pA = th * s / r["K_A"]
    pI = (1 - th) * s / r["K_I"]
    pH = p["rho"] * pI / r["K_H"]
    pT = (p["alpha_A"] * pA + p["alpha"] * pI + p["alpha_H"] * pH) / r["K_T"]
    pQ = (p["phi_A"] * pA + p["phi_I"] * pI + p["phi_H"] * pH) / r["K_Q"]
    pR = (p["gamma_A"] * pA + p["gamma_I"] * pI + p["gamma_H"] * pH + p["gamma_T"] * pT
          + p["gamma_Q"] * pQ) / (p["kappa"] + mu)
    Phi = eta_A * pA + p["eta_I"] * pI + p["eta_H"] * pH + p["eta_T"] * pT + p["eta_Q"] * pQ
    nu = 1 + pA + pI + pH + pT + pQ + pR
    epsV = p["eps_V"]
    a2 = nu * (1 - epsV)
    a1 = nu * r["m1"] + (1 - epsV) * ((s + mu) - p["beta_0"] * Phi)
    a0 = (s + mu) * r["m0"] * (1 - R0_closed_form(p))
    return dict(a2=a2, a1=a1, a0=a0, disc=a1 ** 2 - 4 * a2 * a0,
                zeta=p["kappa"] * pR / (s + mu), Phi=Phi, nu=nu)
