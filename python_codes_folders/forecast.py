"""
forecast.py - probabilistic forecasting engine (proposal P5).

Ensemble members
  M1  mechanistic SVEAITHQR model, immunity 1 year   (outputs/fit)
  M2  mechanistic SVEAITHQR model, immunity 4 years  (outputs/fit_kappa4y)
  B   analogue-season statistical benchmark (resampled past seasons)

Mechanistic sample paths combine three sources of uncertainty:
  (i)   parameter uncertainty: draws from the asymptotic normal distribution of
        the fitted (transformed) parameters, N(x_hat, cov);
  (ii)  season-to-season variability: the unknown season-specific effects
        (m, h, s0, backgrounds) of the target season are drawn from those
        fitted for one of the training seasons, chosen at random;
  (iii) model discrepancy and reporting noise: a multiplicative AR(1)
        log-normal error per week (stationary sd = residual sd of the fit,
        lag-1 correlation 0.7) applied nationally, followed by state-level
        negative binomial noise with size k = 12 (EDA plateau estimate).

Spatial layer (top-down, coherent): national epidemic counts are allocated
to the 37 states with week-specific state shares estimated from the training
seasons (capturing the south-to-north lag and the stronger northern Harmattan
wave); hospitalisations and deaths use state-specific severity ratios (which
reflect healthcare accessibility), rescaled so that states sum to the
national mechanistic series. North, South and national values are sums of
state draws, so all levels are coherent in every sample path.

Week 1 of each season shows a start-of-season artefact (about 4x week 2) in
the supplied data; it is reproduced statistically by multiplying week 1 by a
state-specific ratio resampled from the training seasons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_all  # noqa: E402
from model import initial_state, make_spline_forcing, simulate_weekly  # noqa: E402
from parameters import (FIXED, HARMATTAN_KNOTS, N0, N_KNOTS, REFERENCE_ONLY_FIXED,  # noqa: E402
                        SEASON_SPEC, SHARED_SPEC)
from scoring import QUANTILES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["cases", "hospitalizations", "deaths"]
K_STATE = 12.0          # NB size at state level (EDA, Figure F9)
AR_PHI = 0.7            # lag-1 correlation of the model-discrepancy error


# --------------------------------------------------------------- data views
def state_arrays(seasons):
    """Return state list, meta, and arrays Y[series] with shape (n_seasons, 37, 52)."""
    d = load_all()
    st, meta = d["state"], d["meta"].set_index("state")
    states = sorted(st["state"].unique())
    Y = {}
    for c in SERIES:
        arr = np.zeros((len(seasons), len(states), 52))
        for i, s in enumerate(seasons):
            piv = st[st["season"] == s].pivot(index="state", columns="epi_week_of_season", values=c)
            arr[i] = piv.loc[states].to_numpy(float)
        Y[c] = arr
    return states, meta.loc[states], Y


def spatial_profiles(Y):
    """Week-specific state shares of national epidemic (above-background) cases,
    state background shares, state severity weights and week-1 ratios."""
    base = {c: np.median(Y[c][:, :, 35:], axis=2) for c in SERIES}            # (S, 37)
    ex = np.clip(Y["cases"] - base["cases"][:, :, None], 0, None)             # (S, 37, 52)
    ex[:, :, 0] = 0
    num = ex.sum(0)                                                           # (37, 52)
    den = num.sum(0)                                                          # (52,)
    overall = num.sum(1) / num.sum()
    share = np.where(den[None, :] > 0.01 * den.max(), num / np.maximum(den, 1e-9)[None, :],
                     overall[:, None])
    share = share / share.sum(0, keepdims=True)
    bshare = {c: base[c].sum(0) / base[c].sum() for c in SERIES}               # (37,)
    sev = {}
    for c in ["hospitalizations", "deaths"]:
        exc = np.clip(Y[c] - base[c][:, :, None], 0, None)[:, :, 1:].sum((0, 2))
        sev[c] = exc / np.maximum(ex.sum((0, 2)), 1)                           # per epidemic case
    w1 = {c: Y[c][:, :, 0] / np.maximum(Y[c][:, :, 1], 1) for c in SERIES}     # (S, 37)
    return share, bshare, sev, w1


# ------------------------------------------------------ mechanistic member
def _spec_bounds(names):
    lo, hi, log = [], [], []
    for n in names:
        b = n.split("|")[0]
        g, l, h, sc = (SHARED_SPEC.get(b) or SEASON_SPEC.get(b))
        lo.append(l); hi.append(h); log.append(sc == "log")
    lo, hi, log = map(np.array, (lo, hi, log))
    return np.where(log, np.log(lo), lo), np.where(log, np.log(hi), hi), log


def mechanistic_national(fit_dir, kappa_years, n_paths, rng):
    """Sample national mean trajectories (epidemic + background) for a new season.
    Returns dict series -> (n_paths, 52) of epidemic means and backgrounds (n_paths,)."""
    res = json.loads((Path(fit_dir) / "national_fit.json").read_text())
    names = res["names"]
    x = np.array(res["x"])
    cov = np.array(res["cov_transformed"])
    w, V = np.linalg.eigh((cov + cov.T) / 2)
    L = V * np.sqrt(np.clip(w, 0, None))
    xlo, xhi, islog = _spec_bounds(names)
    seasons = sorted({n.split("|")[1] for n in names if "|" in n})
    out = {c: np.zeros((n_paths, 52)) for c in SERIES}
    back = {c: np.zeros(n_paths) for c in SERIES}
    p_fixed = {**FIXED, "kappa": 1.0 / (kappa_years * 365.0)}
    i = 0
    while i < n_paths:
        xd = np.clip(x + L @ rng.standard_normal(len(x)), xlo, xhi)
        v = np.where(islog, np.exp(xd), xd)
        d = dict(zip(names, v))
        s = seasons[rng.integers(len(seasons))]
        q = {k: d.get(f"{k}|{s}", REFERENCE_ONLY_FIXED.get(k)) for k in SEASON_SPEC}
        g = np.array([0.0] + [d[f"g_{k}"] for k in range(1, N_KNOTS)])
        g[HARMATTAN_KNOTS] += q["h"]
        p = {**p_fixed, **{k: d[k] for k in ("p_c", "rho", "delta_I")},
             "beta_0": d["beta_0"] * q["m"], "forcing_fn": make_spline_forcing(g)}
        try:
            sim = simulate_weekly(p, initial_state(p, N0, q["s0"], d["E0"]))
        except Exception:
            continue
        out["cases"][i] = p["p_c"] * sim["inc_I"]
        out["hospitalizations"][i] = sim["adm_H"]
        out["deaths"][i] = sim["deaths"]
        back["cases"][i], back["hospitalizations"][i], back["deaths"][i] = q["B_c"], q["B_h"], q["B_d"]
        i += 1
    sd = res["series_sd_log"]
    return out, back, sd


def to_states(nat, back, sd, prof, rng, discrepancy=True):
    """Allocate national means to states, add discrepancy, week-1 artefact and
    NB noise. Returns dict series -> (n_paths, 37, 52) integer draws."""
    share, bshare, sev, w1 = prof
    n = nat["cases"].shape[0]
    out = {}
    # common AR(1) discrepancy per series and path
    eps = {}
    for c in SERIES:
        e = np.zeros((n, 52))
        if discrepancy:
            s = sd[c]
            e[:, 0] = rng.normal(0, s, n)
            for t in range(1, 52):
                e[:, t] = AR_PHI * e[:, t - 1] + rng.normal(0, s * np.sqrt(1 - AR_PHI ** 2), n)
        eps[c] = np.exp(e - 0.5 * sd[c] ** 2 if discrepancy else e)
    cases_state = share[None] * nat["cases"][:, None, :]                       # (n,37,52)
    j = rng.integers(w1["cases"].shape[0], size=n)
    for c in SERIES:
        if c == "cases":
            epi = cases_state
        else:
            wts = share * sev[c][:, None]
            wts = wts / np.maximum(wts.sum(0, keepdims=True), 1e-12)
            epi = wts[None] * nat[c][:, None, :]
        mu = (epi + bshare[c][None, :, None] * back[c][:, None, None]) * eps[c][:, None, :]
        mu[:, :, 0] *= w1[c][j]                                                 # week-1 artefact
        mu = np.clip(mu, 1e-9, None)
        out[c] = rng.negative_binomial(K_STATE, K_STATE / (K_STATE + mu))
    return out


# --------------------------------------------------------- analogue member
def analogue_states(Y, n_paths, rng):
    """Benchmark: resample a training season, shift it by -1..+1 weeks (weeks
    2-52), rescale its epidemic part by a log-normal factor (sd = between-season
    sd of log totals) and add NB noise."""
    S = Y["cases"].shape[0]
    tot = Y["cases"].sum((1, 2))
    s_tot = max(0.2, float(np.std(np.log(tot), ddof=1)) if S > 1 else 0.3)
    out = {c: np.zeros((n_paths,) + Y[c].shape[1:], dtype=np.int64) for c in SERIES}
    for i in range(n_paths):
        j = rng.integers(S)
        sh = rng.integers(-1, 2)
        f = np.exp(rng.normal(0, s_tot))
        for c in SERIES:
            y = Y[c][j].copy()
            body = np.roll(y[:, 1:], sh, axis=1)
            if sh > 0:
                body[:, :sh] = y[:, 1:1 + sh]
            elif sh < 0:
                body[:, sh:] = y[:, sh:]
            base = np.median(y[:, 35:], axis=1, keepdims=True)
            body = base + np.clip(body - base, 0, None) * f
            y = np.concatenate([y[:, :1], body], axis=1)
            out[c][i] = rng.negative_binomial(K_STATE, K_STATE / (K_STATE + np.clip(y, 1e-9, None)))
    return out


# ------------------------------------------------------------ aggregation
def aggregate(draws, meta):
    """State draws -> dict location -> series -> (n_paths, 52)."""
    north = (meta["zone"].isin(["NC", "NE", "NW"])).to_numpy()
    loc = {"Nigeria": {c: draws[c].sum(1) for c in SERIES},
           "North": {c: draws[c][:, north].sum(1) for c in SERIES},
           "South": {c: draws[c][:, ~north].sum(1) for c in SERIES}}
    for k, s in enumerate(meta.index):
        loc[s] = {c: draws[c][:, k] for c in SERIES}
    return loc


def mixture(members, weights, rng, n_total):
    """Combine member draws (dict series -> (n,37,52)) by resampling paths."""
    counts = np.round(np.array(weights) / np.sum(weights) * n_total).astype(int)
    out = {}
    for c in SERIES:
        parts = []
        for m, k in zip(members, counts):
            if k > 0:
                idx = rng.integers(m[c].shape[0], size=k)
                parts.append(m[c][idx])
        out[c] = np.concatenate(parts, axis=0)
    return out


def quantiles(samples, taus=QUANTILES):
    """samples (n, T) -> (T, K)."""
    return np.quantile(samples, taus, axis=0).T
