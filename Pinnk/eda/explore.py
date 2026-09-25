"""
MAEPiMS Challenge 2026 - Exploratory Data Analysis
Reads the four challenge CSVs and produces:
  1. National weekly cases / hospitalisations / deaths (3 seasons)
  2. North vs South regional epidemic curves per season
  3. State-level weekly incidence heatmap (per season)
  4. Attack rate vs healthcare-access scatter, coloured by zone
Figures saved to eda/figures/. A text summary is printed to stdout.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

DATA = "MAEPiMS_Challenge_Data/MAEPiMS_Challenge_Data"
OUT = "eda/figures"

# ---- palette (validated dataviz default) ----------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK = "#0b0b0b"
INK_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SEC,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#1c5cab"])

NORTH_ZONES = {"NW", "NC", "NE"}
SOUTH_ZONES = {"SW", "SE", "SS"}

# ---- load -------------------------------------------------------------------
national = pd.read_csv(f"{DATA}/nigeria_flu_weekly_national.csv", parse_dates=["week_start"])
by_state = pd.read_csv(f"{DATA}/nigeria_flu_weekly_by_state.csv", parse_dates=["week_start"])
summary = pd.read_csv(f"{DATA}/nigeria_flu_season_summary.csv", parse_dates=["peak_week_start"])
meta = pd.read_csv(f"{DATA}/nigeria_flu_state_metadata.csv")
ref = pd.read_csv(f"{DATA}/nigeria_flu_season_reference.csv")

seasons = ["2023/2024", "2024/2025", "2025/2026"]
season_labels = {"2023/2024": "2023/24 (Moderate)", "2024/2025": "2024/25 (High)", "2025/2026": "2025/26 (Mod-High)"}
season_color = {"2023/2024": BLUE, "2024/2025": RED, "2025/2026": ORANGE}

print("=" * 70)
print("BASIC SHAPE CHECKS")
print("=" * 70)
print("national:", national.shape, national["season"].unique())
print("by_state:", by_state.shape, by_state["state"].nunique(), "states x",
      by_state.groupby("state").size().unique(), "weeks each")
print("date range:", national["week_start"].min(), "->", national["week_start"].max())
print()

# =============================================================================
# 1. National weekly cases / hospitalisations / deaths (small multiples, one axis each)
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
metrics = [("cases", "Weekly cases"), ("hospitalizations", "Weekly hospitalisations"), ("deaths", "Weekly deaths")]

for ax, (col, title) in zip(axes, metrics):
    for s in seasons:
        d = national[national["season"] == s].sort_values("week_start")
        ax.plot(d["week_start"], d[col], color=season_color[s], linewidth=2,
                solid_capstyle="round", label=season_labels[s])
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.6)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

axes[0].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Nigeria National Influenza Surveillance - 3 Simulated Seasons", fontsize=13, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/01_national_weekly.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 2. North vs South regional epidemic curves per season
# =============================================================================
state_zone = meta.set_index("state")["zone"].to_dict()
by_state["region"] = by_state["state"].map(lambda s: "North" if state_zone.get(s) in NORTH_ZONES else "South")
region_weekly = by_state.groupby(["season", "region", "week_start"], as_index=False).agg(
    cases=("cases", "sum"), population=("population", "sum")
)
region_weekly["cases_per_100k"] = region_weekly["cases"] / region_weekly["population"] * 100000

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
region_color = {"North": ORANGE, "South": BLUE}
for ax, s in zip(axes, seasons):
    d = region_weekly[region_weekly["season"] == s]
    for region in ["North", "South"]:
        dd = d[d["region"] == region].sort_values("week_start")
        ax.plot(dd["week_start"], dd["cases_per_100k"], color=region_color[region], linewidth=2,
                 solid_capstyle="round", label=region)
    ax.set_title(season_labels[s], loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Cases per 100,000")
axes[0].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("North vs South Nigeria - Weekly Incidence per 100k", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/02_north_vs_south.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 3. State-level weekly incidence heatmap (2024/25 high-severity season)
# =============================================================================
season_pick = "2024/2025"
d = by_state[by_state["season"] == season_pick].copy()
d = d.sort_values(["state", "week_start"])
pivot = d.pivot(index="state", columns="epi_week_of_season", values="cases_per_100k")
# order states by zone then name for readability
zone_order = ["NW", "NE", "NC", "SW", "SE", "SS"]
state_order = meta.assign(zorder=meta["zone"].map({z: i for i, z in enumerate(zone_order)})) \
                   .sort_values(["zorder", "state"])["state"].tolist()
pivot = pivot.loc[state_order]

fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(pivot.values, aspect="auto", cmap=SEQ_BLUE)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_xticks(range(0, pivot.shape[1], 4))
ax.set_xticklabels(pivot.columns[::4], fontsize=8)
ax.set_xlabel("Epidemiological week of season")
ax.set_title(f"State-Level Weekly Incidence per 100k - {season_labels[season_pick]}",
             fontsize=12, fontweight="bold", color=INK, loc="left")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Cases per 100,000", color=INK_SEC)
# zone separator lines
zones_seq = meta.set_index("state").loc[state_order, "zone"]
boundaries = [i for i in range(1, len(zones_seq)) if zones_seq.iloc[i] != zones_seq.iloc[i - 1]]
for b in boundaries:
    ax.axhline(b - 0.5, color=SURFACE, linewidth=2)
fig.tight_layout()
fig.savefig(f"{OUT}/03_state_heatmap_2024_25.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 4. Attack rate vs healthcare access, coloured by zone (all seasons)
# =============================================================================
merged = summary.merge(meta[["state", "hc_access"]], on="state", how="left")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, sharex=True)
zones = sorted(merged["zone"].unique())
zone_color = dict(zip(zones, CAT))
for ax, s in zip(axes, seasons):
    d = merged[merged["season"] == s]
    for z in zones:
        dz = d[d["zone"] == z]
        ax.scatter(dz["hc_access"], dz["attack_rate_pct"], s=45, color=zone_color[z],
                   edgecolor=SURFACE, linewidth=0.5, label=z, zorder=3)
    ax.set_title(season_labels[s], loc="left", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("Healthcare accessibility index")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Attack rate (%)")
axes[-1].legend(frameon=False, loc="upper right", fontsize=8, title="Zone", title_fontsize=8)
fig.suptitle("Attack Rate vs Healthcare Accessibility, by Geopolitical Zone", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/04_attack_rate_vs_access.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# Text summary
# =============================================================================
print("=" * 70)
print("SEASON SUMMARY (national)")
print("=" * 70)
for s in seasons:
    d = national[national["season"] == s]
    peak_row = d.loc[d["cases"].idxmax()]
    print(f"\n{season_labels[s]}:")
    print(f"  total cases: {d['cases'].sum():,}  | total deaths: {d['deaths'].sum():,}")
    print(f"  peak week: epi-week {peak_row['epi_week_of_season']} ({peak_row['week_start'].date()}), "
          f"{peak_row['cases']:,} cases, {peak_row['cases_per_100k']:.1f} per 100k")
    cfr = d["deaths"].sum() / d["cases"].sum() * 100
    print(f"  implied national CFR: {cfr:.3f}%")

print("\n" + "=" * 70)
print("STATE EXTREMES (2024/25 high-severity season)")
print("=" * 70)
s24 = summary[summary["season"] == "2024/2025"].merge(meta[["state", "hc_access"]], on="state")
print("Highest attack rate:", s24.nlargest(3, "attack_rate_pct")[["state", "attack_rate_pct", "hc_access"]].to_string(index=False))
print("\nLowest attack rate:", s24.nsmallest(3, "attack_rate_pct")[["state", "attack_rate_pct", "hc_access"]].to_string(index=False))
print("\nHighest CFR:", s24.nlargest(3, "cfr_pct")[["state", "cfr_pct", "hc_access"]].to_string(index=False))
print("\nCorrelation(attack_rate, hc_access):", s24["attack_rate_pct"].corr(s24["hc_access"]).round(3))
print("Correlation(cfr, hc_access):", s24["cfr_pct"].corr(s24["hc_access"]).round(3))

print("\n" + "=" * 70)
print("SEASON REFERENCE MAPPING")
print("=" * 70)
print(ref.to_string(index=False))

print("\nSaved figures to eda/figures/:")
print("  01_national_weekly.png")
print("  02_north_vs_south.png")
print("  03_state_heatmap_2024_25.png")
print("  04_attack_rate_vs_access.png")
