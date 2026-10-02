# MAEPiMS Challenge 2026
# Inter-Seasonal Stability and Variability
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# INTER-SEASONAL STABILITY AND VARIABILITY
# MAEPiMS Nigeria Influenza Forecasting Challenge
#
# Reproduces:
#   1. Cumulative incidence correlations
#   2. Peak-week correlations
#   3. Cumulative deaths per 100,000 correlations
#   4. Case fatality ratio correlations
# ============================================================

library(tidyverse)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

weekly_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_by_state.csv"
)

metadata_file <- file.path(
  data_dir,
  "nigeria_flu_state_metadata.csv"
)

# ------------------------------------------------------------
# 2. Read supplied data
# ------------------------------------------------------------

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

metadata <- read_csv(
  metadata_file,
  show_col_types = FALSE
)

# ------------------------------------------------------------
# 3. Basic validation
# ------------------------------------------------------------

stopifnot(nrow(weekly) == 5772)
stopifnot(nrow(metadata) == 37)

required_weekly <- c(
  "season",
  "state",
  "epi_week_of_season",
  "cases",
  "deaths"
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

# ------------------------------------------------------------
# 4. Check 52 weeks for every state-season
# ------------------------------------------------------------

week_check <- weekly %>%
  group_by(
    state,
    season
  ) %>%
  summarise(
    n_weeks =
      n_distinct(epi_week_of_season),
    .groups = "drop"
  )

stopifnot(
  nrow(week_check) == 37 * 3
)

stopifnot(
  all(week_check$n_weeks == 52)
)

# ------------------------------------------------------------
# 5. Verify state populations
# ------------------------------------------------------------

population_check <- weekly %>%
  distinct(
    state,
    population
  ) %>%
  left_join(
    metadata %>%
      select(
        state,
        population_metadata = population
      ),
    by = "state"
  )

stopifnot(
  all(
    population_check$population ==
      population_check$population_metadata
  )
)

# ------------------------------------------------------------
# 6. Calculate state-season cumulative outcomes
# ------------------------------------------------------------

state_season_totals <- weekly %>%
  group_by(
    state,
    season
  ) %>%
  summarise(

    cumulative_cases =
      sum(cases, na.rm = TRUE),

    cumulative_deaths =
      sum(deaths, na.rm = TRUE),

    population =
      first(population),

    .groups = "drop"
  ) %>%
  mutate(

    cumulative_incidence_per_100k =
      100000 *
      cumulative_cases /
      population,

    cumulative_deaths_per_100k =
      100000 *
      cumulative_deaths /
      population,

    case_fatality_ratio =
      cumulative_deaths /
      cumulative_cases
  )

# ------------------------------------------------------------
# 7. Calculate state-season peak week
# ------------------------------------------------------------

state_season_peak <- weekly %>%
  group_by(
    state,
    season
  ) %>%
  slice_max(
    order_by = cases,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup() %>%
  select(
    state,
    season,
    peak_week = epi_week_of_season,
    peak_cases = cases
  )

# Verify one peak per state-season
stopifnot(
  nrow(state_season_peak) == 37 * 3
)

# ------------------------------------------------------------
# 8. Combine all state-season measures
# ------------------------------------------------------------

state_season_results <- state_season_totals %>%
  left_join(
    state_season_peak,
    by = c(
      "state",
      "season"
    )
  )

stopifnot(
  nrow(state_season_results) == 111
)

print(
  state_season_results
)

# ------------------------------------------------------------
# 9. Put seasons into separate columns
# ------------------------------------------------------------

season_order <- c(
  "2023/2024",
  "2024/2025",
  "2025/2026"
)

# ------------------------------------------------------------
# 10. Function to calculate pairwise correlations
# ------------------------------------------------------------

pairwise_correlations <- function(
    data,
    variable
) {

  wide <- data %>%
    select(
      state,
      season,
      value = all_of(variable)
    ) %>%
    pivot_wider(
      names_from = season,
      values_from = value
    )

  s1 <- season_order[1]
  s2 <- season_order[2]
  s3 <- season_order[3]

  tibble(

    measure = variable,

    `2023/2024_vs_2024/2025` =
      cor(
        wide[[s1]],
        wide[[s2]],
        method = "pearson"
      ),

    `2023/2024_vs_2025/2026` =
      cor(
        wide[[s1]],
        wide[[s3]],
        method = "pearson"
      ),

    `2024/2025_vs_2025/2026` =
      cor(
        wide[[s2]],
        wide[[s3]],
        method = "pearson"
      )
  )
}

# ------------------------------------------------------------
# 11. Cumulative incidence correlations
# ------------------------------------------------------------

cumulative_incidence_correlations <-
  pairwise_correlations(
    state_season_results,
    "cumulative_incidence_per_100k"
  )

cat(
  "\nCumulative incidence correlations:\n"
)

print(
  cumulative_incidence_correlations
)

# ------------------------------------------------------------
# 12. Peak timing correlations
# ------------------------------------------------------------

peak_timing_correlations <-
  pairwise_correlations(
    state_season_results,
    "peak_week"
  )

cat(
  "\nPeak timing correlations:\n"
)

print(
  peak_timing_correlations
)

# ------------------------------------------------------------
# 13. Cumulative deaths per 100,000 correlations
# ------------------------------------------------------------

death_burden_correlations <-
  pairwise_correlations(
    state_season_results,
    "cumulative_deaths_per_100k"
  )

cat(
  "\nCumulative deaths per 100,000 correlations:\n"
)

print(
  death_burden_correlations
)

# ------------------------------------------------------------
# 14. CFR correlations
# ------------------------------------------------------------

cfr_correlations <-
  pairwise_correlations(
    state_season_results,
    "case_fatality_ratio"
  )

cat(
  "\nCase fatality ratio correlations:\n"
)

print(
  cfr_correlations
)

# ------------------------------------------------------------
# 15. Combine all correlation results
# ------------------------------------------------------------

interseasonal_correlations <- bind_rows(
  cumulative_incidence_correlations,
  peak_timing_correlations,
  death_burden_correlations,
  cfr_correlations
)

print(
  interseasonal_correlations
)

# ------------------------------------------------------------
# 16. Rounded manuscript table
# ------------------------------------------------------------

interseasonal_correlations_rounded <-
  interseasonal_correlations %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 3)
    )
  )

print(
  interseasonal_correlations_rounded
)

# ------------------------------------------------------------
# 17. Explicit verification of cumulative incidence
# ------------------------------------------------------------

inc <- cumulative_incidence_correlations

stopifnot(
  abs(
    inc$`2023/2024_vs_2024/2025` -
      (-0.063031)
  ) < 1e-5
)

stopifnot(
  abs(
    inc$`2023/2024_vs_2025/2026` -
      0.137471
  ) < 1e-5
)

stopifnot(
  abs(
    inc$`2024/2025_vs_2025/2026` -
      (-0.133541)
  ) < 1e-5
)

# ------------------------------------------------------------
# 18. Explicit verification of peak timing
# ------------------------------------------------------------

peak <- peak_timing_correlations

stopifnot(
  abs(
    peak$`2023/2024_vs_2024/2025` -
      0.456047
  ) < 1e-5
)

stopifnot(
  abs(
    peak$`2023/2024_vs_2025/2026` -
      0.668615
  ) < 1e-5
)

stopifnot(
  abs(
    peak$`2024/2025_vs_2025/2026` -
      0.559463
  ) < 1e-5
)

# ------------------------------------------------------------
# 19. Explicit verification of cumulative mortality
# ------------------------------------------------------------

death_corr <- death_burden_correlations

stopifnot(
  abs(
    death_corr$`2023/2024_vs_2024/2025` -
      0.774560
  ) < 1e-5
)

stopifnot(
  abs(
    death_corr$`2023/2024_vs_2025/2026` -
      0.781988
  ) < 1e-5
)

stopifnot(
  abs(
    death_corr$`2024/2025_vs_2025/2026` -
      0.765217
  ) < 1e-5
)

# ------------------------------------------------------------
# 20. Explicit verification of CFR
# ------------------------------------------------------------

cfr <- cfr_correlations

stopifnot(
  abs(
    cfr$`2023/2024_vs_2024/2025` -
      0.604154
  ) < 1e-5
)

stopifnot(
  abs(
    cfr$`2023/2024_vs_2025/2026` -
      0.730337
  ) < 1e-5
)

stopifnot(
  abs(
    cfr$`2024/2025_vs_2025/2026` -
      0.668559
  ) < 1e-5
)

cat(
  "\n====================================================\n"
)

cat(
  "ALL INTER-SEASONAL RESULTS VERIFIED SUCCESSFULLY.\n"
)

cat(
  "====================================================\n"
)

# ------------------------------------------------------------
# 21. Save detailed state-season data
# ------------------------------------------------------------

write_csv(
  state_season_results,
  "interseasonal_state_season_measures.csv"
)

# ------------------------------------------------------------
# 22. Save correlation results
# ------------------------------------------------------------

write_csv(
  interseasonal_correlations,
  "interseasonal_correlations_exact.csv"
)

write_csv(
  interseasonal_correlations_rounded,
  "interseasonal_correlations_manuscript.csv"
)
