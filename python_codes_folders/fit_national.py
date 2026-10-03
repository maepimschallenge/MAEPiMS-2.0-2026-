"""
fit_national.py - maximum-likelihood fit of the SVEAITHQR model to the national
weekly cases, hospitalisations and deaths (all three seasons jointly).

Observation model (proposal P1), week w of season s:
    cases_w  ~ NB(mean = p_c * [C_I]_w + B_c,s , size k_c)
    hosp_w   ~ NB(mean = [C_H]_w + B_h,s ,        size k_h)   (p_h = 1)
    deaths_w ~ NB(mean = [C_D]_w + B_d,s ,        size k_d)   (p_d = 1)
where [C]_w is the increment of a bookkeeping integral over the epi week.
For NB2 counts with large means, log(y + 1) has approximately constant
variance psi_1(k) ~ 1/k, so maximising the NB likelihood is approximated by
weighted least squares on log(y + 1); the weights are 1/sd of each series'
log-residuals (two passes). Week 1 of each season is excluded (artefact).

Seasons are fitted jointly: shared epidemiological and forcing parameters,
season-specific susceptible fraction s0, multiplier m, Harmattan shift h and
backgrounds; one seed E0 shared by all seasons (seed and early forcing are
confounded). Forcing: periodic log-spline (P2 option B) with P-spline penalty.
s0 plays the role of the approved per-season severity multiplier: in R_eff
the two are interchangeable (R_eff = R_0(t) * S/N), so only s0 is estimated.

Optimiser: scipy.optimize.least_squares (TRF; Branch et al., 1999) on
transformed parameters, multi-start. Uncertainty: asymptotic covariance from
the Jacobian, s^2 (J^T J)^{-1}, mapped to 95% intervals on the natural scale.

Run: python src/fit_national.py      (about 10-20 minutes)
Outputs: outputs/fit/national_fit.json, tables and figures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_all  # noqa: E402
from model import initial_state, make_spline_forcing, simulate_weekly  # noqa: E402
from parameters import (FIXED, HARMATTAN_KNOTS, N0, N_KNOTS, REFERENCE_ONLY_FIXED, SPLINE_PENALTY,
                        REFERENCE_SEASON, SEASON_SPEC, SEASONS, SHARED_SPEC)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import os
_tag = ("" if not os.environ.get("KAPPA_YEARS") else f"_kappa{os.environ['KAPPA_YEARS']}y")
_tag += ("" if not os.environ.get("TRAIN_SEASONS") else "_train" + os.environ["TRAIN_SEASONS"].replace("/", "").replace(",", "_"))
OUT = ROOT / "outputs" / ("fit" + _tag)
OUT.mkdir(parents=True, exist_ok=True)
SERIES = ["cases", "hospitalizations", "deaths"]


# ----------------------------------------------------------------- packing
def build_spec():
    names, guess, lo, hi, scale = [], [], [], [], []
    for k, (g, l, h, sc) in SHARED_SPEC.items():
        names.append(k); guess.append(g); lo.append(l); hi.append(h); scale.append(sc)
    for s in SEASONS:
        for k, (g, l, h, sc) in SEASON_SPEC.items():
            if k in REFERENCE_ONLY_FIXED and s == REFERENCE_SEASON:
                continue
            names.append(f"{k}|{s}"); guess.append(g); lo.append(l); hi.append(h); scale.append(sc)
    return names, np.array(guess), np.array(lo), np.array(hi), np.array(scale)


NAMES, GUESS, LO, HI, SCALE = build_spec()
IS_LOG = SCALE == "log"


def to_x(v):
    x = np.array(v, dtype=float)
    x[IS_LOG] = np.log(x[IS_LOG])
    return x


def to_v(x):
    v = np.array(x, dtype=float)
    v[IS_LOG] = np.exp(v[IS_LOG])
    return v


X_LO, X_HI = to_x(LO), to_x(HI)


def unpack(v):
    d = dict(zip(NAMES, v))
    shared = {k: d[k] for k in SHARED_SPEC}
    season = {s: {k: d.get(f"{k}|{s}", REFERENCE_ONLY_FIXED.get(k)) for k in SEASON_SPEC} for s in SEASONS}
    return shared, season


# -------------------------------------------------------------- prediction
def predict(v):
    shared, season = unpack(v)
    p = {**FIXED, **shared}
    out = {}
    for s in SEASONS:
        q = season[s]
        g = np.array([0.0] + [shared[f"g_{k}"] for k in range(1, N_KNOTS)])
        g[HARMATTAN_KNOTS] += q["h"]
        ps = {**p, "beta_0": p["beta_0"] * q["m"], "forcing_fn": make_spline_forcing(g), "knots": g}
        y0 = initial_state(ps, N0, q["s0"], shared["E0"])
        sim = simulate_weekly(ps, y0)
        out[s] = {"cases": p["p_c"] * sim["inc_I"] + q["B_c"],
                  "hospitalizations": sim["adm_H"] + q["B_h"],
                  "deaths": sim["deaths"] + q["B_d"],
                  "epi_cases": p["p_c"] * sim["inc_I"],
                  "true_symptomatic": sim["inc_I"],
                  "Y": sim["Y"], "p": ps}
    return out, p


def load_obs():
    nat = load_all()["national"]
    obs = {}
    for s in SEASONS:
        g = nat[nat["season"] == s].sort_values("epi_week_of_season")
        obs[s] = {c: g[c].to_numpy(float) for c in SERIES}
        obs[s]["epiweek_start"] = g["epiweek_start"].to_numpy()
    return obs


OBS = load_obs()
FIT_MASK = np.arange(52) >= 1        # exclude week 1 (artefact)


def residuals(x, w):
    v = to_v(x)
    try:
        pred, p = predict(v)
    except Exception:
        return np.full(len(SEASONS) * 3 * FIT_MASK.sum() + N_KNOTS, 50.0)
    res = []
    for s in SEASONS:
        for j, c in enumerate(SERIES):
            r = np.log1p(OBS[s][c][FIT_MASK]) - np.log1p(np.clip(pred[s][c][FIT_MASK], 0, None))
            res.append(w[j] * r)
    # P-spline smoothness penalty on the (reference) log-forcing knots
    g = np.array([0.0] + [v[NAMES.index(f"g_{k}")] for k in range(1, N_KNOTS)])
    res.append(SPLINE_PENALTY * (np.roll(g, 1) - 2 * g + np.roll(g, -1)))
    out = np.concatenate(res)
    return np.where(np.isfinite(out), out, 50.0)


def fit(x0, w, max_nfev):
    return least_squares(residuals, x0, args=(w,), bounds=(X_LO, X_HI), method="trf",
                         x_scale="jac", diff_step=1e-3, max_nfev=max_nfev, verbose=0)


def series_sd(x):
    v = to_v(x)
    pred, _ = predict(v)
    sds = []
    for c in SERIES:
        r = np.concatenate([np.log1p(OBS[s][c][FIT_MASK]) - np.log1p(pred[s][c][FIT_MASK]) for s in SEASONS])
        sds.append(np.std(r, ddof=1))
    return np.array(sds)


def main(n_starts=1, seed=2026):
    rng = np.random.default_rng(seed)
    w = np.ones(3)
    t0 = time.time()
    # ---- stage 1: multi-start (phases on a grid, others jittered)
    starts = []
    for i in range(n_starts):
        g = to_x(GUESS)
        if i > 0:
            g = g + np.where(IS_LOG, rng.normal(0, 0.3, len(g)), rng.normal(0, 0.2, len(g)))
        starts.append(np.clip(g, X_LO + 1e-6, X_HI - 1e-6))
    results = []
    for i, x0 in enumerate(starts):
        r = fit(x0, w, max_nfev=120)
        results.append(r)
        print(f"start {i + 1}/{n_starts}: cost={r.cost:9.3f} nfev={r.nfev} ({time.time() - t0:5.0f}s)", flush=True)
    best = min(results, key=lambda r: r.cost)

    # ---- stage 2: reweight by series sd and polish (two passes)
    for _ in range(2):
        sd = series_sd(best.x)
        w = 1.0 / sd
        best = fit(best.x, w, max_nfev=400)
        print(f"polish: cost={best.cost:.3f} weights={np.round(w, 2)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- uncertainty: asymptotic covariance in transformed space
    # penalised fit: covariance (J^T J)^-1 using ALL rows (data + penalty),
    # i.e. the penalty acts as a Gaussian smoothness prior on the knots
    J = best.jac
    n_obs, n_par = J.shape[0] - N_KNOTS, J.shape[1]
    r_data = best.fun[:-N_KNOTS]
    s2 = float(r_data @ r_data) / (n_obs - n_par)  # ~1 after reweighting
    JTJ = J.T @ J
    cov = s2 * np.linalg.pinv(JTJ)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(se, se)
    v = to_v(best.x)
    lo95 = np.where(IS_LOG, np.exp(best.x - 1.96 * se), best.x - 1.96 * se)
    hi95 = np.where(IS_LOG, np.exp(best.x + 1.96 * se), best.x + 1.96 * se)
    at_bound = (np.abs(best.x - X_LO) < 1e-4) | (np.abs(best.x - X_HI) < 1e-4)
    sd = series_sd(best.x)
    result = {
        "names": NAMES, "x": best.x.tolist(), "value": v.tolist(), "lo95": lo95.tolist(),
        "hi95": hi95.tolist(), "se_transformed": se.tolist(), "scale": SCALE.tolist(),
        "at_bound": at_bound.tolist(), "cost": float(best.cost), "n_obs": int(n_obs),
        "n_par": int(n_par), "series_sd_log": dict(zip(SERIES, sd.tolist())),
        "nb_size_k": dict(zip(SERIES, (1 / sd ** 2).tolist())),
        "weights": w.tolist(), "cov_transformed": cov.tolist(), "corr": corr.tolist(),
        "fixed": FIXED, "N0": N0, "multistart_costs": [float(r.cost) for r in results],
    }
    (OUT / "national_fit.json").write_text(json.dumps(result, indent=1))
    print(f"done in {time.time() - t0:.0f}s; results -> {OUT / 'national_fit.json'}")
    return result


if __name__ == "__main__":
    main()
