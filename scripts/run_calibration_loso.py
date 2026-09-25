"""
Leave-one-season-out robustness check for the calibration layer.

scripts/run_calibration.py validates the zone wave-center shift on exactly one
split (train on 2023/24+2024/25, test on 2025/26). Before trusting that result,
check it holds up for all 3 possible train/test splits - otherwise the earlier
81% -> 97% peak-week-accuracy jump could just be a fluke of that one split.

RESULT (this is why src/calibration.py ships phase_weight=0.25, not 1.0): the full
shift (phase_weight=1.0) wins on 2 of 3 splits but nearly doubles mean state MAE
(8,366 -> 15,785) when tested against 2024/25, an anomalously-timed high-severity
season (see src/calibration.py's module docstring, "Postmortem, round 3"). This
script reports BOTH the full shift and the shipped 0.25-blend side by side so that
trade-off stays visible rather than getting quietly averaged away.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import load_national, load_by_state, load_state_metadata
from src.renewal import fit_series, generation_kernel, N_SEED
from src.calibration import (
    fit_all_states, allocate_national_to_states,
    build_shape_delta_table, fit_zone_shape_offsets, zone_offsets_dict, allocate_with_phase_shift,
)
from src.metrics import mae, peak_week_accuracy

SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
national = load_national()
by_state = load_by_state()
meta = load_state_metadata()
kernel = generation_kernel()

# ---- fit national + all states for every season once, reused across splits ---
print("Fitting national + per-state renewal cores for all 3 seasons (reused across splits)...")
national_params, national_sims, national_weeks, national_seeds, national_pops = {}, {}, {}, {}, {}
state_fits = {}
for season in SEASONS:
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

    state_fits[season] = fit_all_states(by_state[by_state["season"] == season], params)
    print(f"  {season}: national MAE={mae(observed, sim):,.0f}, {len(state_fits[season])} states fit")

meta_idx = meta.set_index("state")
summary_rows = []

for test_season in SEASONS:
    train_seasons = [s for s in SEASONS if s != test_season]
    train_state_fits = {s: state_fits[s] for s in train_seasons}
    train_national_params = {s: national_params[s] for s in train_seasons}

    delta_table = build_shape_delta_table(train_state_fits, train_national_params, meta)
    zone_offsets = fit_zone_shape_offsets(delta_table)
    offsets_dict = zone_offsets_dict(zone_offsets)

    test_by_state = by_state[by_state["season"] == test_season]
    naive_alloc = allocate_national_to_states(national_sims[test_season], meta, bias_model=None)
    shipped_alloc = allocate_with_phase_shift(  # phase_weight=0.25, the shipped default
        national_params[test_season], offsets_dict, meta,
        national_weeks[test_season], national_seeds[test_season], national_pops[test_season], kernel,
    )
    full_alloc = allocate_with_phase_shift(  # phase_weight=1.0, shown only for contrast
        national_params[test_season], offsets_dict, meta,
        national_weeks[test_season], national_seeds[test_season], national_pops[test_season], kernel,
        phase_weight=1.0,
    )

    rows = []
    for state, d in test_by_state.groupby("state"):
        d = d.sort_values("epi_week_of_season")
        weeks = d["epi_week_of_season"].to_numpy(dtype=float)
        observed = d["cases"].to_numpy(dtype=float)
        naive, shipped, full = naive_alloc[state].to_numpy(), shipped_alloc[state].to_numpy(), full_alloc[state].to_numpy()
        rows.append({
            "state": state,
            "mae_naive": mae(observed, naive), "mae_shipped": mae(observed, shipped), "mae_full": mae(observed, full),
            "true_pk": weeks[observed.argmax()], "naive_pk": weeks[naive.argmax()],
            "shipped_pk": weeks[shipped.argmax()], "full_pk": weeks[full.argmax()],
        })
    edf = pd.DataFrame(rows)
    naive_acc = peak_week_accuracy(edf["true_pk"], edf["naive_pk"])
    shipped_acc = peak_week_accuracy(edf["true_pk"], edf["shipped_pk"])
    full_acc = peak_week_accuracy(edf["true_pk"], edf["full_pk"])
    n_improved = (edf["mae_shipped"] < edf["mae_naive"]).sum()

    print(f"\nTest season = {test_season}  (trained on {train_seasons})")
    print(f"  mean MAE   naive={edf['mae_naive'].mean():>9,.0f}   shipped(0.25)={edf['mae_shipped'].mean():>9,.0f}   full(1.0)={edf['mae_full'].mean():>9,.0f}")
    print(f"  peak-wk acc (+-1wk)   naive={naive_acc:.2%}   shipped(0.25)={shipped_acc:.2%}   full(1.0)={full_acc:.2%}")
    print(f"  states improved (shipped vs naive): {n_improved}/{len(edf)}")

    summary_rows.append({
        "test_season": test_season, "mae_naive": edf["mae_naive"].mean(),
        "mae_shipped": edf["mae_shipped"].mean(), "mae_full": edf["mae_full"].mean(),
        "peak_acc_naive": naive_acc, "peak_acc_shipped": shipped_acc, "peak_acc_full": full_acc,
        "states_improved": n_improved, "n_states": len(edf),
    })

summary = pd.DataFrame(summary_rows)
print(f"\n{'='*70}\nLEAVE-ONE-SEASON-OUT SUMMARY (all 3 splits)\n{'='*70}")
print(summary.to_string(index=False))
out = Path(__file__).resolve().parents[1] / "outputs" / "calibration_loso_summary.csv"
summary.to_csv(out, index=False)
print(f"\nSaved {out}")
