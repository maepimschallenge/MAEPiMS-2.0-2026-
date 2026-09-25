"""
Calibration layer end-to-end run.

1. Fit the national renewal core for all 3 seasons.
2. Fit every state's renewal core (shrunk toward national) for the 2 TRAINING
   seasons only (2023/24, 2024/25) -> measure each state's systematic timing/width
   offset from the national fit, and its systematic total-count magnitude bias
   against a naive population-proportional split of the national trajectory.
3. Fit (a) the zone/climate/hc_access magnitude-bias regression and (b) the
   zone-level timing/width shrinkage on that training data only.
4. Evaluate on the HELD-OUT season (2025/26): naive population-only allocation vs
   magnitude-bias-corrected vs zone-phase-shifted (both the shipped default blend,
   phase_weight=0.25, and the full shift, phase_weight=1.0, shown side by side), state
   by state, against what was actually observed. The held-out season's own state-level
   data is used only for scoring, never for fitting.

   IMPORTANT: this script alone only tests ONE train/test split, and the full phase
   shift looks great on it (97% peak-week accuracy) - but that does NOT generalize; see
   scripts/run_calibration_loso.py for all 3 leave-one-season-out splits and
   src/calibration.py's module docstring for why the shipped default is the conservative
   0.25 blend, not the full shift this single split would suggest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_national, load_by_state, load_state_metadata
from src.renewal import fit_series, generation_kernel, N_SEED
from src.calibration import (
    fit_all_states, build_bias_table, fit_bias_regression, allocate_national_to_states,
    build_shape_delta_table, fit_zone_shape_offsets, zone_offsets_dict, allocate_with_phase_shift,
)
from src.metrics import mae, rmse, peak_week_accuracy

TRAIN_SEASONS = ["2023/2024", "2024/2025"]
TEST_SEASON = "2025/2026"
BLUE, RED, ORANGE, MUTED = "#2a78d6", "#e34948", "#eb6834", "#898781"

national = load_national()
by_state = load_by_state()
meta = load_state_metadata()
kernel = generation_kernel()

# ---- 1. national fits, all seasons -----------------------------------------
print("Fitting national renewal core (3 seasons)...")
national_params, national_sims, national_weeks, national_seeds, national_pops = {}, {}, {}, {}, {}
for season in TRAIN_SEASONS + [TEST_SEASON]:
    d = national[national["season"] == season].sort_values("epi_week_of_season")
    weeks = d["epi_week_of_season"].to_numpy(dtype=float)
    observed = d["cases"].to_numpy(dtype=float)
    population = float(d["population"].iloc[0])
    params, sim = fit_series(weeks, observed, population, kernel)
    national_params[season] = params
    national_sims[season] = sim
    national_weeks[season] = weeks
    national_seeds[season] = observed[:N_SEED]
    national_pops[season] = population
    print(f"  {season}: national MAE={mae(observed, sim):,.0f}")

# ---- 2. per-state fits, training seasons only -------------------------------
print("\nFitting per-state renewal cores (training seasons, shrunk toward national)...")
state_fits = {}
for season in TRAIN_SEASONS:
    d_season = by_state[by_state["season"] == season]
    state_fits[season] = fit_all_states(d_season, national_params[season])
    print(f"  {season}: fit {len(state_fits[season])} states")

# ---- 3a. magnitude bias regression -------------------------------------------
train_by_state = by_state[by_state["season"].isin(TRAIN_SEASONS)]
bias_table = build_bias_table(train_by_state, {s: national_sims[s] for s in TRAIN_SEASONS}, meta)
bias_model = fit_bias_regression(bias_table)
print("\nMagnitude bias regression (training seasons only):")
print(bias_model.summary().tables[1])

# ---- 3b. zone-level timing/width shrinkage (the layer that actually matters) --
delta_table = build_shape_delta_table(state_fits, national_params, meta)
zone_offsets = fit_zone_shape_offsets(delta_table)
offsets_dict = zone_offsets_dict(zone_offsets)
print("\nShrunk zone timing/width offsets vs national fit (weeks):")
print(zone_offsets.pivot(index="zone", columns="field", values="shrunk_delta").round(2).to_string())

# ---- 4. held-out evaluation: naive vs magnitude vs phase (default blend + full) -
test_by_state = by_state[by_state["season"] == TEST_SEASON]
naive_alloc = allocate_national_to_states(national_sims[TEST_SEASON], meta, bias_model=None)
magnitude_alloc = allocate_national_to_states(national_sims[TEST_SEASON], meta, bias_model=bias_model)
phase_alloc = allocate_with_phase_shift(
    national_params[TEST_SEASON], offsets_dict, meta,
    national_weeks[TEST_SEASON], national_seeds[TEST_SEASON], national_pops[TEST_SEASON], kernel,
)  # default phase_weight=0.25 (conservative, see docstring above)
phase_full_alloc = allocate_with_phase_shift(
    national_params[TEST_SEASON], offsets_dict, meta,
    national_weeks[TEST_SEASON], national_seeds[TEST_SEASON], national_pops[TEST_SEASON], kernel,
    phase_weight=1.0,  # shown only for comparison -- NOT what's shipped, see LOSO caveat above
)
meta_idx = meta.set_index("state")

rows = []
for state, d in test_by_state.groupby("state"):
    d = d.sort_values("epi_week_of_season")
    weeks = d["epi_week_of_season"].to_numpy(dtype=float)
    observed = d["cases"].to_numpy(dtype=float)
    naive, magnitude = naive_alloc[state].to_numpy(), magnitude_alloc[state].to_numpy()
    phase, phase_full = phase_alloc[state].to_numpy(), phase_full_alloc[state].to_numpy()
    rows.append({
        "state": state, "zone": meta_idx.loc[state, "zone"],
        "mae_naive": mae(observed, naive), "mae_magnitude": mae(observed, magnitude),
        "mae_phase": mae(observed, phase), "mae_phase_full": mae(observed, phase_full),
        "rmse_naive": rmse(observed, naive), "rmse_magnitude": rmse(observed, magnitude), "rmse_phase": rmse(observed, phase),
        "true_peak_wk": weeks[observed.argmax()], "naive_peak_wk": weeks[naive.argmax()],
        "magnitude_peak_wk": weeks[magnitude.argmax()], "phase_peak_wk": weeks[phase.argmax()],
        "phase_full_peak_wk": weeks[phase_full.argmax()],
    })
eval_df = pd.DataFrame(rows)

print(f"\n{'='*70}\nHELD-OUT SEASON {TEST_SEASON} - state-level evaluation\n{'='*70}")
for method in ["naive", "magnitude", "phase", "phase_full"]:
    peak_acc = peak_week_accuracy(eval_df["true_peak_wk"], eval_df[f"{method}_peak_wk"])
    print(f"{method:>11}:  mean MAE={eval_df[f'mae_{method}'].mean():>9,.0f}   peak-wk acc (+-1wk)={peak_acc:.2%}"
          + ("   <- shipped default" if method == "phase" else "   <- single-split only, see LOSO" if method == "phase_full" else ""))
n_improved = (eval_df["mae_phase"] < eval_df["mae_naive"]).sum()
print(f"States where shipped (0.25-blend) phase calibration improved MAE over naive: {n_improved} / {len(eval_df)}")
print("\nBy zone (mean MAE, naive vs shipped default):")
print(eval_df.groupby("zone")[["mae_naive", "mae_phase"]].mean().round(0).to_string())

eval_df.sort_values("mae_naive", ascending=False).to_csv(
    Path(__file__).resolve().parents[1] / "outputs" / "calibration_eval_2025_26.csv", index=False
)

# ---- diagnostic figure: 4 representative states ------------------------------
picks = ["Kebbi", "Lagos", "Oyo", "Bayelsa"]  # low-access North, high-access South, biggest outlier, smallest state
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True)
for ax, state in zip(axes, picks):
    d = test_by_state[test_by_state["state"] == state].sort_values("epi_week_of_season")
    weeks = d["epi_week_of_season"].to_numpy(dtype=float)
    observed = d["cases"].to_numpy(dtype=float)
    zone = meta_idx.loc[state, "zone"]
    ax.plot(weeks, observed, color=MUTED, linewidth=2, label="Observed")
    ax.plot(weeks, naive_alloc[state], color=BLUE, linewidth=1.5, linestyle=":", label="Naive (pop. share)")
    ax.plot(weeks, phase_alloc[state], color=RED, linewidth=2, linestyle="--", label="Phase-calibrated")
    ax.set_title(f"{state} ({zone})", loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Weekly cases")
fig.suptitle(f"Held-out {TEST_SEASON}: naive vs zone-phase-calibrated state allocation", fontsize=13, fontweight="bold")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "eda" / "figures" / "06_calibration_holdout.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
print(f"Saved outputs/calibration_eval_2025_26.csv")
