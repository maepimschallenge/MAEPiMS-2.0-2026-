"""
Data loading for the MAEPiMS Challenge 2026 dataset.

All paths are relative to the repo root and point at the supplied CSVs under
MAEPiMS_Challenge_Data/MAEPiMS_Challenge_Data/. Per the competition rules, only
this supplied simulated dataset may be used for model development and evaluation.
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "MAEPiMS_Challenge_Data" / "MAEPiMS_Challenge_Data"

SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
SEASON_SEVERITY = {"2023/2024": "Moderate", "2024/2025": "High", "2025/2026": "Moderate-High"}
NORTH_ZONES = {"NW", "NC", "NE"}
SOUTH_ZONES = {"SW", "SE", "SS"}


def load_national(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Weekly national cases/hospitalisations/deaths, 3 seasons."""
    df = pd.read_csv(data_dir / "nigeria_flu_weekly_national.csv", parse_dates=["week_start"])
    return df


def load_by_state(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Weekly state-level surveillance data for all 36 states + FCT, 3 seasons."""
    df = pd.read_csv(data_dir / "nigeria_flu_weekly_by_state.csv", parse_dates=["week_start"])
    return df


def load_season_summary(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-state, per-season seasonal summary (attack rate, peak week, CFR, ...)."""
    df = pd.read_csv(data_dir / "nigeria_flu_season_summary.csv", parse_dates=["peak_week_start"])
    return df


def load_state_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Static state metadata: population, zone, climate, healthcare-access index."""
    return pd.read_csv(data_dir / "nigeria_flu_state_metadata.csv")


def load_season_reference(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Mapping of each synthetic season to its real-world US-season analogue."""
    return pd.read_csv(data_dir / "nigeria_flu_season_reference.csv")


def load_all(data_dir: Path = DATA_DIR) -> dict:
    """Convenience loader returning every table keyed by name."""
    return {
        "national": load_national(data_dir),
        "by_state": load_by_state(data_dir),
        "season_summary": load_season_summary(data_dir),
        "state_metadata": load_state_metadata(data_dir),
        "season_reference": load_season_reference(data_dir),
    }


def add_region(df: pd.DataFrame, meta: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    """Attach a North/South `region` column derived from each state's geopolitical zone."""
    zone_map = meta.set_index("state")["zone"].to_dict()
    out = df.copy()
    out["region"] = out[state_col].map(
        lambda s: "North" if zone_map.get(s) in NORTH_ZONES
        else ("South" if zone_map.get(s) in SOUTH_ZONES else None)
    )
    return out


def train_test_split_by_season(
    df: pd.DataFrame,
    train_seasons=("2023/2024", "2024/2025"),
    test_season: str = "2025/2026",
):
    """Split any season-indexed table into train/test by whole season(s).

    Matches the protocol in docs/DESIGN.md: fit on the moderate + high-severity
    seasons, backtest on the held-out moderate-high season.
    """
    train = df[df["season"].isin(train_seasons)].copy()
    test = df[df["season"] == test_season].copy()
    return train, test
