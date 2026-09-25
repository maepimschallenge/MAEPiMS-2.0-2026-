"""
Smoke test for the renewal core (src/renewal.py): fit each season's national
series, report fit quality, and plot fitted vs observed. Not part of the final
pipeline - a sanity check that the model shape (two Gaussian-forced waves +
susceptible depletion) can actually reproduce the observed curves before the
calibration/residual layers are built on top of it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_national
from src.renewal import fit_series, generation_kernel
from src.metrics import mae, rmse

BLUE, RED, ORANGE = "#2a78d6", "#e34948", "#eb6834"
SEASON_COLOR = {"2023/2024": BLUE, "2024/2025": RED, "2025/2026": ORANGE}

national = load_national()
kernel = generation_kernel()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)

for ax, season in zip(axes, ["2023/2024", "2024/2025", "2025/2026"]):
    d = national[national["season"] == season].sort_values("epi_week_of_season")
    weeks = d["epi_week_of_season"].to_numpy(dtype=float)
    observed = d["cases"].to_numpy(dtype=float)
    population = float(d["population"].iloc[0])

    params, sim = fit_series(weeks, observed, population, kernel)
    m, r = mae(observed, sim), rmse(observed, sim)
    peak_true_wk, peak_pred_wk = weeks[observed.argmax()], weeks[sim.argmax()]

    print(f"\n{season}:  MAE={m:,.0f}  RMSE={r:,.0f}  "
          f"true peak wk={peak_true_wk:.0f} (val {observed.max():,.0f})  "
          f"fitted peak wk={peak_pred_wk:.0f} (val {sim.max():,.0f})")
    print(f"  params: {params}")

    ax.plot(weeks, observed, color="#898781", linewidth=2, label="Observed")
    ax.plot(weeks, sim, color=SEASON_COLOR[season], linewidth=2, linestyle="--", label="Renewal fit")
    ax.set_title(season, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epi week of season")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

axes[0].set_ylabel("Weekly national cases")
fig.suptitle("Renewal-core fit vs observed - national weekly cases", fontsize=13, fontweight="bold")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "eda" / "figures" / "05_renewal_fit_smoke_test.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
