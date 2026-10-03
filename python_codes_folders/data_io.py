"""
data_io.py - loading, validation and epidemiological-week alignment for the
MAEPiMS Challenge 2026 synthetic influenza dataset.

Libraries: pandas (McKinney, 2010; The pandas development team, 2020),
NumPy (Harris et al., 2020).

Epidemiological-week convention
-------------------------------
The supplied files index weeks by `week_start`, which falls on a MONDAY for
every row (ISO-8601 weeks). The challenge requires forecasts aligned with
epidemiological weeks running Sunday to Saturday (MMWR convention).
Without daily data the counts cannot be re-binned, so we adopt the documented
convention that each supplied Monday-Sunday week is reported under the
Sunday-Saturday epi week that starts the day before (6 of the 7 days overlap).
The mapping is a pure relabelling: `epiweek_start = week_start - 1 day`.
This is an assumption of the report and is isolated in one function so it can
be changed if the organisers clarify otherwise.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

NORTH_ZONES = {"NC", "NE", "NW"}   # FCT is in the NC zone
SOUTH_ZONES = {"SE", "SS", "SW"}


# ---------------------------------------------------------------------------
# MMWR epidemiological week (Sunday-Saturday). Week 1 is the first week of
# the year that contains at least four days of that calendar year.
# ---------------------------------------------------------------------------
def _mmwr_week1_start(year: int) -> date:
    jan1 = date(year, 1, 1)
    # Python weekday(): Monday=0 ... Sunday=6 ; convert to Sunday=0
    dow_sun0 = (jan1.weekday() + 1) % 7
    first_sunday_on_or_before = jan1 - timedelta(days=dow_sun0)
    # if Jan 1 is Wed/Thu/Fri/Sat (dow 3..6) the week holds < 4 days of the year
    return first_sunday_on_or_before + timedelta(days=7) if dow_sun0 >= 4 else first_sunday_on_or_before


def mmwr_week(d: date) -> tuple[int, int]:
    """Return (mmwr_year, mmwr_week) for a calendar date."""
    for yr in (d.year + 1, d.year, d.year - 1):
        start = _mmwr_week1_start(yr)
        if d >= start:
            return yr, (d - start).days // 7 + 1
    raise ValueError(d)


def add_epiweek_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Sunday-Saturday epi-week labels (see module docstring)."""
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out["epiweek_start"] = out["week_start"] - pd.Timedelta(days=1)   # Sunday
    out["epiweek_end"] = out["epiweek_start"] + pd.Timedelta(days=6)  # Saturday
    yw = out["epiweek_start"].dt.date.map(mmwr_week)
    out["mmwr_year"] = [y for y, _ in yw]
    out["mmwr_week"] = [w for _, w in yw]
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_all(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    nat = pd.read_csv(data_dir / "nigeria_flu_weekly_national.csv")
    st = pd.read_csv(data_dir / "nigeria_flu_weekly_by_state.csv")
    meta = pd.read_csv(data_dir / "nigeria_flu_state_metadata.csv")
    summ = pd.read_csv(data_dir / "nigeria_flu_season_summary.csv")
    ref = pd.read_csv(data_dir / "nigeria_flu_season_reference.csv")

    nat = add_epiweek_columns(nat)
    st = add_epiweek_columns(st)
    st["region"] = np.where(st["zone"].isin(NORTH_ZONES), "North", "South")
    meta["region"] = np.where(meta["zone"].isin(NORTH_ZONES), "North", "South")

    # Flag the first week of each season: in every season and every state the
    # week-1 count is ~4-6x week 2 (a start-of-season artefact, not
    # transmission). We keep the value but flag it for exclusion when fitting.
    for df in (nat, st):
        df["flag_week1_artefact"] = df["epi_week_of_season"].eq(1)

    return {"national": nat, "state": st, "meta": meta, "summary": summ, "reference": ref}


def validate(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Internal-consistency checks; returns a table of check -> result."""
    nat, st, meta = d["national"], d["state"], d["meta"]
    agg = st.groupby(["season", "epi_week_of_season"])[["cases", "hospitalizations", "deaths"]].sum()
    natk = nat.set_index(["season", "epi_week_of_season"])[["cases", "hospitalizations", "deaths"]]
    rows = [
        ("states + FCT", meta["state"].nunique() == 37),
        ("weeks per state", bool((st.groupby("state").size() == 156).all())),
        ("sum of states == national (all 3 series)", bool((agg - natk).abs().to_numpy().max() == 0)),
        ("national population == sum of state populations", int(nat["population"].iloc[0]) == int(meta["population"].sum())),
        ("no missing values", int(st.isna().sum().sum()) == 0),
        ("no negative counts", bool((st[["cases", "hospitalizations", "deaths"]] >= 0).all().all())),
        ("all week_start are Mondays", bool((nat["week_start"].dt.dayofweek == 0).all())),
        # informational: deaths are NOT a subset of hospitalisations (community
        # deaths exist), consistent with deaths from both I and H in the model
        ("state-weeks with deaths > hospitalisations", int((st["deaths"] > st["hospitalizations"]).sum())),
    ]
    return pd.DataFrame(rows, columns=["check", "result"])


def regional_series(st: pd.DataFrame, by: str = "region") -> pd.DataFrame:
    """Aggregate state series to a grouping (region, zone or climate)."""
    g = (st.groupby(["season", "epi_week_of_season", "epiweek_start", by])
           [["cases", "hospitalizations", "deaths", "population"]].sum().reset_index())
    for c in ["cases", "hospitalizations", "deaths"]:
        g[f"{c}_per_100k"] = g[c] / g["population"] * 1e5
    return g
