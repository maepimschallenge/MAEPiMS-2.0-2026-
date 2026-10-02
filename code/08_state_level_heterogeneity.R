# MAEPiMS Challenge 2026
# State-Level Heterogeneity
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ================================================================
# MAEPiMS Challenge 2026
# RESULTS: STATE-LEVEL HETEROGENEITY
# ================================================================

library(tidyverse)

# ----------------------------------------------------------------
# 1. File location
# ----------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

weekly_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_by_state.csv"
)

metadata_file <- file.path(
  data_dir,
  "nigeria_flu_state_metadata.csv"
)

# ----------------------------------------------------------------
# 2. Read the supplied datasets
# ----------------------------------------------------------------

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

metadata <- read_csv(
  metadata_file,
  show_col_types = FALSE
)

# ----------------------------------------------------------------
# 3. Basic data validation
# ----------------------------------------------------------------

stopifnot(nrow(weekly) == 5772)
stopifnot(nrow(metadata) == 37)

required_weekly <- c(
  "season",
  "state",
  "epi_week_of_season",
  "cases",
  "cases_per_100k"
)

required_metadata <- c(
  "state",
  "population"
)

stopifnot(
  all(required_weekly %in% names(weekly))
)

stopifnot(
  all(required_metadata %in% names(metadata))
)

# ----------------------------------------------------------------
# 4. Verify 37 states x 3 seasons x 52 weeks
# ----------------------------------------------------------------

week_check <- weekly %>%
  group_by(state, season) %>%
  summarise(
    n_weeks = n_distinct(epi_week_of_season),
    .groups = "drop"
  )

stopifnot(nrow(week_check) == 37 * 3)
stopifnot(all(week_check$n_weeks == 52))

cat(
  "Number of state-season combinations:",
  nrow(week_check), "\n"
)

cat(
  "Number of weekly observations:",
  nrow(weekly), "\n"
)

# ----------------------------------------------------------------
# 5. Verify population information
# ----------------------------------------------------------------

population_check <- weekly %>%
  distinct(state, population) %>%
  arrange(state) %>%
  left_join(
    metadata %>%
      select(state, population_metadata = population),
    by = "state"
  )

stopifnot(
  all(
    population_check$population ==
      population_check$population_metadata
  )
)

cat(
  "Population values in weekly data agree with metadata.\n"
)

# ----------------------------------------------------------------
# 6. Construct state-level maximum weekly case count
# ----------------------------------------------------------------
#
# C_s = maximum weekly case count across all three seasons.
#
# This is the quantity that reproduces the reported correlation
# approximately equal to 0.82.
# ----------------------------------------------------------------

state_peak_cases <- weekly %>%
  group_by(state) %>%
  summarise(
    peak_cases = max(cases, na.rm = TRUE),
    peak_week = epi_week_of_season[
      which.max(cases)
    ],
    .groups = "drop"
  )

# Add population

state_peak_cases <- state_peak_cases %>%
  left_join(
    metadata %>%
      select(state, population),
    by = "state"
  )

# ----------------------------------------------------------------
# 7. Population versus absolute case burden
# ----------------------------------------------------------------

cor_population_peak_cases <- cor(
  state_peak_cases$population,
  state_peak_cases$peak_cases,
  method = "pearson"
)

cat(
  "\nCorrelation between population and maximum weekly cases:\n"
)

cat(
  "r = ",
  round(cor_population_peak_cases, 4),
  "\n",
  sep = ""
)

# ----------------------------------------------------------------
# 8. Verify the reported approximate value
# ----------------------------------------------------------------

stopifnot(
  abs(
    cor_population_peak_cases - 0.8252259
  ) < 1e-6
)

# ----------------------------------------------------------------
# 9. Calculate cumulative cases for each state
# ----------------------------------------------------------------

state_cumulative <- weekly %>%
  group_by(state) %>%
  summarise(
    cumulative_cases = sum(
      cases,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  left_join(
    metadata %>%
      select(state, population),
    by = "state"
  )

# ----------------------------------------------------------------
# 10. Calculate cumulative incidence per 100,000
# ----------------------------------------------------------------

state_cumulative <- state_cumulative %>%
  mutate(
    cumulative_incidence_per_100k =
      100000 *
      cumulative_cases /
      population,

    attack_rate_percent =
      100 *
      cumulative_cases /
      population
  )

# ----------------------------------------------------------------
# 11. Population versus cumulative incidence
# ----------------------------------------------------------------

cor_population_cumulative_incidence <- cor(
  state_cumulative$population,
  state_cumulative$cumulative_incidence_per_100k,
  method = "pearson"
)

cat(
  "\nCorrelation between population and cumulative incidence:\n"
)

cat(
  "r = ",
  round(
    cor_population_cumulative_incidence,
    4
  ),
  "\n",
  sep = ""
)

# ----------------------------------------------------------------
# 12. Population versus attack rate
# ----------------------------------------------------------------

cor_population_attack_rate <- cor(
  state_cumulative$population,
  state_cumulative$attack_rate_percent,
  method = "pearson"
)

cat(
  "\nCorrelation between population and attack rate:\n"
)

cat(
  "r = ",
  round(
    cor_population_attack_rate,
    4
  ),
  "\n",
  sep = ""
)

# ----------------------------------------------------------------
# 13. Verify that the two correlations are identical
# ----------------------------------------------------------------
#
# Attack rate (%) and cumulative incidence per 100,000 differ only
# by a constant scaling factor:
#
# cumulative incidence = 1000 x attack rate.
#
# Therefore their Pearson correlations with population are identical.
# ----------------------------------------------------------------

stopifnot(
  abs(
    cor_population_cumulative_incidence -
      cor_population_attack_rate
  ) < 1e-12
)

# ----------------------------------------------------------------
# 14. Expected reproducible value
# ----------------------------------------------------------------

expected_correlation <- 0.0367075

stopifnot(
  abs(
    cor_population_cumulative_incidence -
      expected_correlation
  ) < 1e-6
)

# ----------------------------------------------------------------
# 15. Display a compact results table
# ----------------------------------------------------------------

state_heterogeneity_results <- tibble(
  Relationship = c(
    "Population vs. maximum weekly cases",
    "Population vs. cumulative incidence",
    "Population vs. attack rate"
  ),
  Pearson_r = c(
    cor_population_peak_cases,
    cor_population_cumulative_incidence,
    cor_population_attack_rate
  )
) %>%
  mutate(
    Pearson_r = round(Pearson_r, 4)
  )

print(state_heterogeneity_results)

# ----------------------------------------------------------------
# 16. Save results
# ----------------------------------------------------------------

write_csv(
  state_peak_cases,
  "state_peak_cases_and_population.csv"
)

write_csv(
  state_cumulative,
  "state_cumulative_burden_and_population.csv"
)

write_csv(
  state_heterogeneity_results,
  "state_heterogeneity_correlations.csv"
)

# ----------------------------------------------------------------
# 17. Final verification message
# ----------------------------------------------------------------

cat(
  "\n========================================================\n"
)

cat(
  "STATE-LEVEL HETEROGENEITY RESULTS VERIFIED\n"
)

cat(
  "========================================================\n"
)

cat(
  "Population vs maximum weekly cases: r = ",
  round(
    cor_population_peak_cases,
    4
  ),
  "\n",
  sep = ""
)

cat(
  "Population vs cumulative incidence: r = ",
  round(
    cor_population_cumulative_incidence,
    4
  ),
  "\n",
  sep = ""
)

cat(
  "Population vs attack rate: r = ",
  round(
    cor_population_attack_rate,
    4
  ),
  "\n",
  sep = ""
)

cat(
  "========================================================\n"
)
