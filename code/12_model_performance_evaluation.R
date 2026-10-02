# MAEPiMS Challenge 2026
# Model Performance Evaluation
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# MAEPiMS MODEL PERFORMANCE EVALUATION
# Verification and manuscript tables
# ============================================================

library(tidyverse)

# ------------------------------------------------------------
# 1. File containing the two chronological holdout results
# ------------------------------------------------------------

metrics_file <- "/mnt/data/maepims_eval/season_holdout_metrics.csv"

holdout <- read_csv(
  metrics_file,
  show_col_types = FALSE
)

print(holdout)

# ------------------------------------------------------------
# 2. Basic checks
# ------------------------------------------------------------

required_columns <- c(
  "holdout",
  "model",
  "MAE",
  "RMSE",
  "WIS",
  "CRPS",
  "PeakWeekError",
  "CumulativeErrorPct"
)

stopifnot(
  all(required_columns %in% names(holdout))
)

# Two chronological holdouts
expected_holdouts <- c(
  "2024/2025",
  "2025/2026"
)

stopifnot(
  setequal(
    unique(holdout$holdout),
    expected_holdouts
  )
)

# M0-M4 for case forecasting
expected_models <- c(
  "M0",
  "M1",
  "M2",
  "M3",
  "M4"
)

stopifnot(
  setequal(
    unique(holdout$model),
    expected_models
  )
)

# Each holdout must contain five case models
model_counts <- holdout %>%
  count(holdout)

print(model_counts)

stopifnot(
  all(model_counts$n == 5)
)

# ------------------------------------------------------------
# 3. Recalculate the average across the two holdouts
# ------------------------------------------------------------

model_performance <- holdout %>%
  group_by(model) %>%
  summarise(
    MAE =
      mean(MAE),

    RMSE =
      mean(RMSE),

    WIS =
      mean(WIS),

    CRPS =
      mean(CRPS),

    PeakWeekError =
      mean(PeakWeekError),

    CumulativeErrorPct =
      mean(CumulativeErrorPct),

    .groups = "drop"
  )

print(model_performance)

# ------------------------------------------------------------
# 4. Round to manuscript precision
# ------------------------------------------------------------

model_performance_table <- model_performance %>%
  mutate(
    MAE =
      round(MAE, 2),

    RMSE =
      round(RMSE, 2),

    WIS =
      round(WIS, 2),

    CRPS =
      round(CRPS, 2),

    PeakWeekError =
      round(PeakWeekError, 2),

    CumulativeErrorPct =
      round(CumulativeErrorPct, 2)
  )

print(model_performance_table)

# ------------------------------------------------------------
# 5. Expected manuscript values
# ------------------------------------------------------------

expected_case_results <- tribble(

  ~model, ~MAE, ~RMSE, ~WIS, ~CRPS,
  ~PeakWeekError, ~CumulativeErrorPct,

  "M0",
  8690.96,
  24801.99,
  9213.58,
  7328.48,
  1.61,
  28.93,

  "M1",
  12517.21,
  35512.47,
  14787.73,
  11401.50,
  1.73,
  69.55,

  "M2",
  12547.63,
  35323.05,
  14653.61,
  11330.10,
  1.89,
  66.83,

  "M3",
  11259.22,
  32127.37,
  12417.22,
  9742.75,
  1.84,
  57.58,

  "M4",
  11418.56,
  32369.16,
  12658.28,
  9919.55,
  1.96,
  57.21
)

# ------------------------------------------------------------
# 6. Compare calculated and manuscript values
# ------------------------------------------------------------

comparison <- model_performance_table %>%
  left_join(
    expected_case_results,
    by = "model",
    suffix = c(
      "_calculated",
      "_expected"
    )
  )

print(comparison)

# ------------------------------------------------------------
# 7. Verify every table value
# ------------------------------------------------------------

for (i in seq_len(nrow(expected_case_results))) {

  model_i <- expected_case_results$model[i]

  calculated_i <-
    model_performance %>%
    filter(model == model_i)

  expected_i <-
    expected_case_results[i, ]

  stopifnot(
    abs(
      calculated_i$MAE -
        expected_i$MAE
    ) < 0.01
  )

  stopifnot(
    abs(
      calculated_i$RMSE -
        expected_i$RMSE
    ) < 0.01
  )

  stopifnot(
    abs(
      calculated_i$WIS -
        expected_i$WIS
    ) < 0.01
  )

  stopifnot(
    abs(
      calculated_i$CRPS -
        expected_i$CRPS
    ) < 0.01
  )

  stopifnot(
    abs(
      calculated_i$PeakWeekError -
        expected_i$PeakWeekError
    ) < 0.01
  )

  stopifnot(
    abs(
      calculated_i$CumulativeErrorPct -
        expected_i$CumulativeErrorPct
    ) < 0.01
  )
}

cat(
  "\nAll case-model performance values verified.\n"
)

# ------------------------------------------------------------
# 8. Calculate percentage change M2 -> M3
# ------------------------------------------------------------

m2 <- model_performance %>%
  filter(model == "M2")

m3 <- model_performance %>%
  filter(model == "M3")

m2_m3_change <- tibble(

  metric = c(
    "MAE",
    "RMSE",
    "WIS",
    "CRPS",
    "Peak-week error",
    "Cumulative error"
  ),

  M2 = c(
    m2$MAE,
    m2$RMSE,
    m2$WIS,
    m2$CRPS,
    m2$PeakWeekError,
    m2$CumulativeErrorPct
  ),

  M3 = c(
    m3$MAE,
    m3$RMSE,
    m3$WIS,
    m3$CRPS,
    m3$PeakWeekError,
    m3$CumulativeErrorPct
  )
) %>%
  mutate(
    percent_change =
      100 * (M3 - M2) / M2,

    percent_reduction =
      100 * (M2 - M3) / M2
  )

print(m2_m3_change)

# ------------------------------------------------------------
# 9. Calculate M3 -> M4 change
# ------------------------------------------------------------

m4 <- model_performance %>%
  filter(model == "M4")

m3_m4_change <- tibble(

  metric = c(
    "MAE",
    "RMSE",
    "WIS",
    "CRPS",
    "Peak-week error",
    "Cumulative error"
  ),

  M3 = c(
    m3$MAE,
    m3$RMSE,
    m3$WIS,
    m3$CRPS,
    m3$PeakWeekError,
    m3$CumulativeErrorPct
  ),

  M4 = c(
    m4$MAE,
    m4$RMSE,
    m4$WIS,
    m4$CRPS,
    m4$PeakWeekError,
    m4$CumulativeErrorPct
  )
) %>%
  mutate(
    percent_change =
      100 * (M4 - M3) / M3
  )

print(m3_m4_change)

# ------------------------------------------------------------
# 10. Identify the lowest-error model
# ------------------------------------------------------------

best_by_metric <- tibble(

  metric = c(
    "MAE",
    "RMSE",
    "WIS",
    "CRPS",
    "Peak-week error",
    "Cumulative error"
  ),

  selected_model = c(
    model_performance$model[
      which.min(model_performance$MAE)
    ],

    model_performance$model[
      which.min(model_performance$RMSE)
    ],

    model_performance$model[
      which.min(model_performance$WIS)
    ],

    model_performance$model[
      which.min(model_performance$CRPS)
    ],

    model_performance$model[
      which.min(model_performance$PeakWeekError)
    ],

    model_performance$model[
      which.min(
        model_performance$CumulativeErrorPct
      )
    ]
  )
)

print(best_by_metric)

# ------------------------------------------------------------
# 11. Verify that M0 is lowest for all reported metrics
# ------------------------------------------------------------

stopifnot(
  best_by_metric$selected_model[1] == "M0"
)

stopifnot(
  best_by_metric$selected_model[2] == "M0"
)

stopifnot(
  best_by_metric$selected_model[3] == "M0"
)

stopifnot(
  best_by_metric$selected_model[4] == "M0"
)

stopifnot(
  best_by_metric$selected_model[5] == "M0"
)

stopifnot(
  best_by_metric$selected_model[6] == "M0"
)

cat(
  "\nM0 has the lowest value for all six case metrics ",
  "in the current preliminary evaluation.\n"
)

# ------------------------------------------------------------
# 12. Save tables
# ------------------------------------------------------------

write_csv(
  holdout,
  "case_model_holdout_results.csv"
)

write_csv(
  model_performance_table,
  "case_model_performance_table.csv"
)

write_csv(
  m2_m3_change,
  "M2_to_M3_component_change.csv"
)

write_csv(
  m3_m4_change,
  "M3_to_M4_component_change.csv"
)

write_csv(
  best_by_metric,
  "best_model_by_metric.csv"
)

# ============================================================
# M5 SEVERITY PERFORMANCE
# ============================================================

severity_file <-
  "/mnt/data/maepims_eval/m5_severity_metrics.csv"

severity <- read_csv(
  severity_file,
  show_col_types = FALSE
)

print(severity)

# ------------------------------------------------------------
# Expected structure
# ------------------------------------------------------------

required_severity <- c(
  "outcome",
  "MAE",
  "RMSE",
  "WIS",
  "CRPS",
  "PeakWeekError",
  "CumulativeErrorPct"
)

stopifnot(
  all(
    required_severity %in%
      names(severity)
  )
)

stopifnot(
  nrow(severity) == 2
)

# ------------------------------------------------------------
# Round for manuscript
# ------------------------------------------------------------

severity_table <- severity %>%
  mutate(
    MAE =
      round(MAE, 2),

    RMSE =
      round(RMSE, 2),

    WIS =
      round(WIS, 2),

    CRPS =
      round(CRPS, 2),

    PeakWeekError =
      round(PeakWeekError, 2),

    CumulativeErrorPct =
      round(CumulativeErrorPct, 2)
  )

print(severity_table)

# ------------------------------------------------------------
# Expected values
# ------------------------------------------------------------

expected_severity <- tribble(

  ~outcome, ~MAE, ~RMSE, ~WIS, ~CRPS,
  ~PeakWeekError, ~CumulativeErrorPct,

  "Hospitalisations",
  165.26,
  460.22,
  189.11,
  147.23,
  1.92,
  57.87,

  "Deaths",
  15.97,
  41.49,
  17.48,
  13.81,
  1.53,
  57.26
)

# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

severity_comparison <- severity_table %>%
  left_join(
    expected_severity,
    by = "outcome",
    suffix = c(
      "_calculated",
      "_expected"
    )
  )

print(severity_comparison)

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

for (i in seq_len(nrow(expected_severity))) {

  outcome_i <-
    expected_severity$outcome[i]

  calc_i <-
    severity %>%
    filter(
      outcome == outcome_i
    )

  exp_i <-
    expected_severity[i, ]

  stopifnot(
    abs(calc_i$MAE - exp_i$MAE) < 0.01
  )

  stopifnot(
    abs(calc_i$RMSE - exp_i$RMSE) < 0.01
  )

  stopifnot(
    abs(calc_i$WIS - exp_i$WIS) < 0.01
  )

  stopifnot(
    abs(calc_i$CRPS - exp_i$CRPS) < 0.01
  )

  stopifnot(
    abs(
      calc_i$PeakWeekError -
        exp_i$PeakWeekError
    ) < 0.01
  )

  stopifnot(
    abs(
      calc_i$CumulativeErrorPct -
        exp_i$CumulativeErrorPct
    ) < 0.01
  )
}

cat(
  "\nAll M5 severity performance values verified.\n"
)

write_csv(
  severity_table,
  "M5_severity_performance_table.csv"
)
