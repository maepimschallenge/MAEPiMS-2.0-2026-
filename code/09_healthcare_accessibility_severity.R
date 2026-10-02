# MAEPiMS Challenge 2026
# Healthcare Accessibility and Severity
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# Healthcare Accessibility and Severity
# Reproducible analysis for Results subsection
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
# 3. Basic data checks
# ------------------------------------------------------------

stopifnot(nrow(weekly) == 5772)
stopifnot(nrow(metadata) == 37)

required_weekly <- c(
  "season",
  "state",
  "population",
  "epi_week_of_season",
  "cases",
  "hospitalizations",
  "deaths"
)

required_metadata <- c(
  "state",
  "population",
  "hc_access"
)

stopifnot(
  all(required_weekly %in% names(weekly))
)

stopifnot(
  all(required_metadata %in% names(metadata))
)

# Check that every state-season contains 52 weeks
week_check <- weekly %>%
  group_by(state, season) %>%
  summarise(
    n_weeks = n_distinct(epi_week_of_season),
    .groups = "drop"
  )

stopifnot(
  nrow(week_check) == 37 * 3
)

stopifnot(
  all(week_check$n_weeks == 52)
)

# ------------------------------------------------------------
# 4. Check population consistency between files
# ------------------------------------------------------------

population_check <- weekly %>%
  distinct(state, population) %>%
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
# 5. Aggregate weekly data to state-season level
# ------------------------------------------------------------

state_season <- weekly %>%
  group_by(
    state,
    season
  ) %>%
  summarise(
    cumulative_cases =
      sum(cases, na.rm = TRUE),

    cumulative_hospitalizations =
      sum(hospitalizations, na.rm = TRUE),

    cumulative_deaths =
      sum(deaths, na.rm = TRUE),

    population =
      first(population),

    .groups = "drop"
  ) %>%
  left_join(
    metadata %>%
      select(
        state,
        hc_access
      ),
    by = "state"
  )

# ------------------------------------------------------------
# 6. Calculate severity and transmission measures
# ------------------------------------------------------------

state_season <- state_season %>%
  mutate(

    # Cumulative deaths per 100,000
    cumulative_deaths_per_100k =
      100000 *
      cumulative_deaths /
      population,

    # Case fatality ratio, expressed as percentage
    cfr_percent =
      100 *
      cumulative_deaths /
      cumulative_cases,

    # Attack rate, expressed as percentage
    attack_rate_percent =
      100 *
      cumulative_cases /
      population,

    # Hospitalisation-to-case ratio, expressed as percentage
    hospitalization_case_ratio =
      100 *
      cumulative_hospitalizations /
      cumulative_cases
  )

# ------------------------------------------------------------
# 7. Check that there are 37 states in each season
# ------------------------------------------------------------

state_counts <- state_season %>%
  count(season)

print(state_counts)

stopifnot(
  all(state_counts$n == 37)
)

# ------------------------------------------------------------
# 8. Calculate Pearson correlations by season
# ------------------------------------------------------------

healthcare_correlations <- state_season %>%
  group_by(season) %>%
  summarise(

    # Healthcare accessibility vs mortality burden
    hc_deaths_per_100k =
      cor(
        hc_access,
        cumulative_deaths_per_100k,
        method = "pearson"
      ),

    # Healthcare accessibility vs case fatality ratio
    hc_cfr =
      cor(
        hc_access,
        cfr_percent,
        method = "pearson"
      ),

    # Healthcare accessibility vs attack rate
    hc_attack_rate =
      cor(
        hc_access,
        attack_rate_percent,
        method = "pearson"
      ),

    .groups = "drop"
  )

print(healthcare_correlations)

# ------------------------------------------------------------
# 9. Rounded results for manuscript
# ------------------------------------------------------------

healthcare_results_table <- healthcare_correlations %>%
  mutate(
    hc_deaths_per_100k =
      round(hc_deaths_per_100k, 3),

    hc_cfr =
      round(hc_cfr, 3),

    hc_attack_rate =
      round(hc_attack_rate, 3)
  )

print(healthcare_results_table)

# ------------------------------------------------------------
# 10. Expected values from direct calculation
# ------------------------------------------------------------

expected_results <- tribble(
  ~season,      ~hc_deaths_per_100k, ~hc_cfr, ~hc_attack_rate,

  "2023/2024",  -0.879,              -0.811,   0.145,
  "2024/2025",  -0.919,              -0.841,   0.089,
  "2025/2026",  -0.834,              -0.812,   0.617
)

comparison <- healthcare_results_table %>%
  left_join(
    expected_results,
    by = "season",
    suffix = c("_calculated", "_expected")
  )

print(comparison)

# ------------------------------------------------------------
# 11. Verify manuscript values
# ------------------------------------------------------------

stopifnot(
  abs(
    healthcare_correlations$hc_deaths_per_100k[
      healthcare_correlations$season == "2023/2024"
    ] + 0.878536
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_deaths_per_100k[
      healthcare_correlations$season == "2024/2025"
    ] + 0.919226
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_deaths_per_100k[
      healthcare_correlations$season == "2025/2026"
    ] + 0.833552
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_cfr[
      healthcare_correlations$season == "2023/2024"
    ] + 0.811366
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_cfr[
      healthcare_correlations$season == "2024/2025"
    ] + 0.841016
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_cfr[
      healthcare_correlations$season == "2025/2026"
    ] + 0.812072
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_attack_rate[
      healthcare_correlations$season == "2023/2024"
    ] - 0.144620
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_attack_rate[
      healthcare_correlations$season == "2024/2025"
    ] - 0.088927
  ) < 1e-5
)

stopifnot(
  abs(
    healthcare_correlations$hc_attack_rate[
      healthcare_correlations$season == "2025/2026"
    ] - 0.616581
  ) < 1e-5
)

cat(
  "\nAll healthcare-accessibility correlations ",
  "were reproduced from the supplied dataset.\n"
)

# ------------------------------------------------------------
# 12. Save results
# ------------------------------------------------------------

write_csv(
  state_season,
  "healthcare_state_season_analysis.csv"
)

write_csv(
  healthcare_correlations,
  "healthcare_access_correlations_exact.csv"
)

write_csv(
  healthcare_results_table,
  "healthcare_access_correlations_table.csv"
)
