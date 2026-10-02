# MAEPiMS Challenge 2026
# Secondary-Wave Heterogeneity by Climate
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ================================================================
# Secondary-Wave Heterogeneity by Climate
# MAEPiMS Challenge 2026
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

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

# ----------------------------------------------------------------
# 2. Basic data checks
# ----------------------------------------------------------------

stopifnot(nrow(weekly) == 5772)

required_vars <- c(
  "season",
  "state",
  "climate",
  "epi_week_of_season",
  "cases"
)

stopifnot(
  all(required_vars %in% names(weekly))
)

# Check that each state-season has 52 weeks
week_check <- weekly %>%
  group_by(state, season) %>%
  summarise(
    n_weeks = n_distinct(epi_week_of_season),
    .groups = "drop"
  )

stopifnot(all(week_check$n_weeks == 52))

# ----------------------------------------------------------------
# 3. Check climate composition
# ----------------------------------------------------------------

climate_counts <- weekly %>%
  distinct(state, climate) %>%
  count(climate)

print(climate_counts)

stopifnot(
  climate_counts$n[
    match("tropical", climate_counts$climate)
  ] == 17
)

stopifnot(
  climate_counts$n[
    match("savanna", climate_counts$climate)
  ] == 10
)

stopifnot(
  climate_counts$n[
    match("sahel", climate_counts$climate)
  ] == 10
)

# ----------------------------------------------------------------
# 4. Identify primary and secondary peaks
# ----------------------------------------------------------------
#
# Definition:
# Primary peak:
#   maximum weekly case count over the 52 weeks.
#
# Secondary peak:
#   largest local maximum occurring after week 20.
#
# A local maximum satisfies:
#   cases[t] > cases[t-1]
#   cases[t] >= cases[t+1]
#
# ----------------------------------------------------------------

state_season_wave <- weekly %>%
  arrange(state, season, epi_week_of_season) %>%
  group_by(state, season) %>%
  group_modify(~ {

    dat <- .x %>%
      arrange(epi_week_of_season) %>%
      mutate(
        previous_cases = lag(cases),
        next_cases = lead(cases)
      )

    # ------------------------------------------------------------
    # Primary peak
    # ------------------------------------------------------------

    primary_row <- dat %>%
      slice_max(
        order_by = cases,
        n = 1,
        with_ties = FALSE
      )

    primary_week <- primary_row$epi_week_of_season
    primary_cases <- primary_row$cases

    # ------------------------------------------------------------
    # Candidate secondary local maxima
    # ------------------------------------------------------------

    secondary_candidates <- dat %>%
      filter(
        epi_week_of_season > 20,
        !is.na(previous_cases),
        !is.na(next_cases),
        cases > previous_cases,
        cases >= next_cases
      )

    # ------------------------------------------------------------
    # Select largest secondary local maximum
    # ------------------------------------------------------------

    if (nrow(secondary_candidates) > 0) {

      secondary_row <- secondary_candidates %>%
        slice_max(
          order_by = cases,
          n = 1,
          with_ties = FALSE
        )

      secondary_week <- secondary_row$epi_week_of_season
      secondary_cases <- secondary_row$cases

      secondary_ratio <-
        secondary_cases / primary_cases

    } else {

      secondary_week <- NA_real_
      secondary_cases <- NA_real_
      secondary_ratio <- NA_real_

    }

    tibble(
      state = dat$state[1],
      season = dat$season[1],
      climate = dat$climate[1],
      primary_week = primary_week,
      primary_cases = primary_cases,
      secondary_week = secondary_week,
      secondary_cases = secondary_cases,
      secondary_ratio = secondary_ratio
    )

  }) %>%
  ungroup()

# ----------------------------------------------------------------
# 5. Verify 37 states x 3 seasons = 111 state-season combinations
# ----------------------------------------------------------------

stopifnot(
  nrow(state_season_wave) == 37 * 3
)

stopifnot(
  n_distinct(
    paste(
      state_season_wave$state,
      state_season_wave$season
    )
  ) == 111
)

# Check that every state-season has a secondary peak
stopifnot(
  all(!is.na(state_season_wave$secondary_ratio))
)

# ----------------------------------------------------------------
# 6. Calculate climate-specific mean ratios
# ----------------------------------------------------------------

secondary_climate_results <- state_season_wave %>%
  group_by(season, climate) %>%
  summarise(
    n_states = n(),
    mean_ratio = mean(
      secondary_ratio,
      na.rm = TRUE
    ),
    sd_ratio = sd(
      secondary_ratio,
      na.rm = TRUE
    ),
    median_ratio = median(
      secondary_ratio,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  arrange(
    season,
    factor(
      climate,
      levels = c(
        "sahel",
        "savanna",
        "tropical"
      )
    )
  )

print(secondary_climate_results)

# ----------------------------------------------------------------
# 7. Create manuscript table
# ----------------------------------------------------------------

secondary_climate_table <- secondary_climate_results %>%
  select(
    season,
    climate,
    mean_ratio
  ) %>%
  mutate(
    mean_ratio = round(mean_ratio, 3)
  ) %>%
  pivot_wider(
    names_from = climate,
    values_from = mean_ratio
  ) %>%
  select(
    season,
    sahel,
    savanna,
    tropical
  )

print(secondary_climate_table)

# ----------------------------------------------------------------
# 8. Expected values for verification
# ----------------------------------------------------------------

expected <- tribble(
  ~season,      ~sahel, ~savanna, ~tropical,

  "2023/2024",   0.342,   0.167,    0.097,
  "2024/2025",   0.581,   0.552,    0.269,
  "2025/2026",   0.318,   0.254,    0.113
)

# ----------------------------------------------------------------
# 9. Compare calculated and expected values
# ----------------------------------------------------------------

comparison <- secondary_climate_table %>%
  left_join(
    expected,
    by = "season",
    suffix = c(
      "_calculated",
      "_expected"
    )
  )

print(comparison)

# ----------------------------------------------------------------
# 10. Exact verification
# ----------------------------------------------------------------

stopifnot(
  all.equal(
    secondary_climate_table$sahel,
    expected$sahel,
    tolerance = 1e-10
  )
)

stopifnot(
  all.equal(
    secondary_climate_table$savanna,
    expected$savanna,
    tolerance = 1e-10
  )
)

stopifnot(
  all.equal(
    secondary_climate_table$tropical,
    expected$tropical,
    tolerance = 1e-10
  )
)

cat(
  "\n====================================================\n"
)

cat(
  "All reported climate-specific secondary-wave ratios\n"
)

cat(
  "have been independently reproduced from the supplied\n"
)

cat(
  "state-week dataset under the stated peak definitions.\n"
)

cat(
  "====================================================\n"
)

# ----------------------------------------------------------------
# 11. Save detailed state-level results
# ----------------------------------------------------------------

write_csv(
  state_season_wave,
  "secondary_wave_state_season_results.csv"
)

# ----------------------------------------------------------------
# 12. Save detailed climate-level results
# ----------------------------------------------------------------

write_csv(
  secondary_climate_results,
  "secondary_wave_climate_results_detailed.csv"
)

# ----------------------------------------------------------------
# 13. Save manuscript table
# ----------------------------------------------------------------

write_csv(
  secondary_climate_table,
  "secondary_wave_climate_table.csv"
)
