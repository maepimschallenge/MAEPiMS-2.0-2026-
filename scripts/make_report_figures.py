"""
Two additional figures for the technical report that the brief's "Recommended
Visualisations" list asks for and the earlier EDA/diagnostic figures don't cover:
forecast uncertainty bands, and a peak-week summary across states. Reuses the
already-generated outputs/ CSVs from scripts/run_ensemble.py - run that first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data import load_national, load_state_metadata

OUT = Path(__file__).resolve().parents[1] / "eda" / "figures"
BLUE, RED, ORANGE, MUTED, GRID = "#2a78d6", "#e34948", "#eb6834", "#898781", "#e1e0d9"
SEQ_BLUE_LIGHT = "#b7d3f6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})

national = load_national()
meta = load_state_metadata()

# ---- 1. National forecast uncertainty bands, held-out 2025/26 -----------------
season = "2025/2026"
regional = pd.read_csv(OUT.parent.parent / "outputs" / f"regional_national_{season.replace('/', '_')}.csv",
                        parse_dates=["week_start"])
nat_forecast = regional[regional["region"] == "National"].sort_values("epi_week_of_season")
truth = national[national["season"] == season].sort_values("epi_week_of_season")

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(nat_forecast["week_start"], nat_forecast["q0.05"], nat_forecast["q0.95"],
                 color=SEQ_BLUE_LIGHT, alpha=0.5, label="90% interval")
ax.fill_between(nat_forecast["week_start"], nat_forecast["q0.25"], nat_forecast["q0.75"],
                 color=BLUE, alpha=0.35, label="50% interval")
ax.plot(nat_forecast["week_start"], nat_forecast["q0.5"], color=BLUE, linewidth=2, label="Median forecast")
ax.plot(truth["week_start"], truth["cases"], color=MUTED, linewidth=2, linestyle="--", label="Observed")
ax.set_title(f"National Weekly Cases - Forecast Uncertainty Bands, Held-Out {season}",
             loc="left", fontsize=12, fontweight="bold")
ax.set_ylabel("Weekly cases")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "07_forecast_uncertainty_bands.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 2. Peak-week summary across states, 2024/25 (the anomalous season) -------
season = "2024/2025"
seasonal = pd.read_csv(OUT.parent.parent / "outputs" / f"seasonal_summary_{season.replace('/', '_')}.csv")
seasonal = seasonal.sort_values("peak_week")
zone_order = ["NW", "NE", "NC", "SW", "SE", "SS"]
zone_color = dict(zip(zone_order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]))

fig, ax = plt.subplots(figsize=(9, 11))
y = range(len(seasonal))
ax.scatter(seasonal["peak_week"], y, c=[zone_color[z] for z in seasonal["zone"]], s=55, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(seasonal["state"], fontsize=8)
ax.set_xlabel("Forecast peak epidemiological week")
ax.set_title(f"Peak-Week Summary by State - Held-Out {season}\n(forecast from models trained on the other 2 seasons)",
             loc="left", fontsize=11, fontweight="bold")
ax.grid(axis="y", visible=False)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=zone_color[z], label=z) for z in zone_order]
ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8, title="Zone", title_fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "08_peak_week_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(f"  {OUT / '07_forecast_uncertainty_bands.png'}")
print(f"  {OUT / '08_peak_week_summary.png'}")
