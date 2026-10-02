# MAEPiMS Challenge 2026
# Relationship Between Cases, Hospitalizations, and Deaths
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# Relationship Between Cases, Hospitalisations, and Deaths
# MAEPiMS Nigeria Influenza Forecasting Challenge
# ============================================================

library(tidyverse)

# ------------------------------------------------------------
# 1. File location
# ------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

weekly_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_by_state.csv"
)

# ------------------------------------------------------------
# 2. Read supplied state-week data
# ------------------------------------------------------------

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

# ------------------------------------------------------------
# 3. Basic checks
# ------------------------------------------------------------

stopifnot(nrow(weekly) == 5772)

required_vars <- c(
  "season",
  "state",
  "epi_week_of_season",
  "cases",
  "hospitalizations",
  "deaths"
)

stopifnot(
  all(required_vars %in% names(weekly))
)

# Each state-season must contain 52 weeks
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
# 4. Aggregate states to national weekly totals
# ------------------------------------------------------------

national_weekly <- weekly %>%
  group_by(
    season,
    epi_week_of_season
  ) %>%
  summarise(
    cases = sum(cases, na.rm = TRUE),
    hospitalizations =
      sum(hospitalizations, na.rm = TRUE),
    deaths =
      sum(deaths, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(
    season,
    epi_week_of_season
  )

# Check 52 weeks per season
national_week_check <- national_weekly %>%
  count(season)

print(national_week_check)

stopifnot(
  all(national_week_check$n == 52)
)

stopifnot(
  nrow(national_weekly) == 156
)

# ------------------------------------------------------------
# 5. Verify national totals against supplied national file
# ------------------------------------------------------------

national_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_national.csv"
)

national_supplied <- read_csv(
  national_file,
  show_col_types = FALSE
)

stopifnot(
  nrow(national_supplied) == 156
)

# Inspect the national file columns
print(names(national_supplied))

# The following assumes the supplied national file uses
# the same outcome names. If necessary, adjust only the
# column names after inspecting names(national_supplied).

national_check <- national_weekly %>%
  left_join(
    national_supplied %>%
      select(
        season,
        epi_week_of_season,
        cases_national = cases,
        hospitalizations_national = hospitalizations,
        deaths_national = deaths
      ),
    by = c(
      "season",
      "epi_week_of_season"
    )
  )

stopifnot(
  all(
    national_check$cases ==
      national_check$cases_national
  )
)

stopifnot(
  all(
    national_check$hospitalizations ==
      national_check$hospitalizations_national
  )
)

stopifnot(
  all(
    national_check$deaths ==
      national_check$deaths_national
  )
)

cat(
  "\nState-level aggregation exactly matches ",
  "the supplied national data.\n"
)

# ------------------------------------------------------------
# 6. Season-specific contemporaneous correlations
# ------------------------------------------------------------

season_correlations <- national_weekly %>%
  group_by(season) %>%
  summarise(

    cases_hospitalizations =
      cor(
        cases,
        hospitalizations,
        method = "pearson"
      ),

    cases_deaths =
      cor(
        cases,
        deaths,
        method = "pearson"
      ),

    .groups = "drop"
  )

print(season_correlations)

# ------------------------------------------------------------
# 7. Mean correlations across the three seasons
# ------------------------------------------------------------

mean_correlations <- season_correlations %>%
  summarise(

    mean_cases_hospitalizations =
      mean(
        cases_hospitalizations
      ),

    mean_cases_deaths =
      mean(
        cases_deaths
      )
  )

print(mean_correlations)

cat(
  "\nMean case-hospitalisation correlation = ",
  round(
    mean_correlations$mean_cases_hospitalizations,
    4
  ),
  "\n",
  sep = ""
)

cat(
  "Mean case-death correlation = ",
  round(
    mean_correlations$mean_cases_deaths,
    4
  ),
  "\n",
  sep = ""
)

# ------------------------------------------------------------
# 8. Function for lagged cross-correlation
# ------------------------------------------------------------
#
# Definition:
# Positive lag k means that cases at week t are
# compared with the downstream outcome at week t+k.
#
# Therefore:
# k = 0 : same week
# k = 1 : cases lead outcome by 1 week
# k = 2 : cases lead outcome by 2 weeks
# ------------------------------------------------------------

lag_correlation <- function(x, y, lag = 0) {

  n <- length(x)

  stopifnot(
    length(y) == n
  )

  if (lag == 0) {

    return(
      cor(
        x,
        y,
        method = "pearson"
      )
    )

  } else {

    return(
      cor(
        x[1:(n - lag)],
        y[(lag + 1):n],
        method = "pearson"
      )
    )
  }
}

# ------------------------------------------------------------
# 9. Calculate case-hospitalisation cross-correlations
# ------------------------------------------------------------

case_hosp_lags <- expand_grid(
  season = unique(national_weekly$season),
  lag = c(0, 1, 2)
) %>%
  mutate(
    correlation = map2_dbl(
      season,
      lag,
      ~ {

        dat <- national_weekly %>%
          filter(
            season == .x
          ) %>%
          arrange(
            epi_week_of_season
          )

        lag_correlation(
          dat$cases,
          dat$hospitalizations,
          .y
        )
      }
    )
  )

print(case_hosp_lags)

# ------------------------------------------------------------
# 10. Calculate case-death cross-correlations
# ------------------------------------------------------------

case_death_lags <- expand_grid(
  season = unique(national_weekly$season),
  lag = c(0, 1, 2)
) %>%
  mutate(
    correlation = map2_dbl(
      season,
      lag,
      ~ {

        dat <- national_weekly %>%
          filter(
            season == .x
          ) %>%
          arrange(
            epi_week_of_season
          )

        lag_correlation(
          dat$cases,
          dat$deaths,
          .y
        )
      }
    )
  )

print(case_death_lags)

# ------------------------------------------------------------
# 11. Average cross-correlations across seasons
# ------------------------------------------------------------

case_hosp_mean <- case_hosp_lags %>%
  group_by(lag) %>%
  summarise(
    mean_correlation =
      mean(correlation),
    sd_correlation =
      sd(correlation),
    .groups = "drop"
  )

case_death_mean <- case_death_lags %>%
  group_by(lag) %>%
  summarise(
    mean_correlation =
      mean(correlation),
    sd_correlation =
      sd(correlation),
    .groups = "drop"
  )

cat("\nCase -> Hospitalisation:\n")
print(case_hosp_mean)

cat("\nCase -> Death:\n")
print(case_death_mean)

# ------------------------------------------------------------
# 12. Create manuscript-ready cross-correlation table
# ------------------------------------------------------------

cross_correlation_table <- tibble(

  Outcome = c(
    "Cases -> Hospitalisations",
    "Cases -> Deaths"
  ),

  Lag_0 = c(
    case_hosp_mean$mean_correlation[
      case_hosp_mean$lag == 0
    ],
    case_death_mean$mean_correlation[
      case_death_mean$lag == 0
    ]
  ),

  Lag_1 = c(
    case_hosp_mean$mean_correlation[
      case_hosp_mean$lag == 1
    ],
    case_death_mean$mean_correlation[
      case_death_mean$lag == 1
    ]
  ),

  Lag_2 = c(
    case_hosp_mean$mean_correlation[
      case_hosp_mean$lag == 2
    ],
    case_death_mean$mean_correlation[
      case_death_mean$lag == 2
    ]
  )
) %>%
  mutate(
    across(
      starts_with("Lag"),
      ~ round(.x, 3)
    )
  )

print(cross_correlation_table)

# ------------------------------------------------------------
# 13. Identify observed peak weeks
# ------------------------------------------------------------

peak_weeks <- national_weekly %>%
  group_by(season) %>%
  summarise(

    cases_peak_week =
      epi_week_of_season[
        which.max(cases)
      ],

    hospitalizations_peak_week =
      epi_week_of_season[
        which.max(hospitalizations)
      ],

    deaths_peak_week =
      epi_week_of_season[
        which.max(deaths)
      ],

    .groups = "drop"
  )

print(peak_weeks)

# ------------------------------------------------------------
# 14. Verify reported peak weeks
# ------------------------------------------------------------

expected_peak_weeks <- tribble(

  ~season,      ~cases_peak_week,
  ~hospitalizations_peak_week,
  ~deaths_peak_week,

  "2023/2024",  14, 14, 15,
  "2024/2025",  16, 16, 16,
  "2025/2026",  15, 16, 16
)

peak_comparison <- peak_weeks %>%
  left_join(
    expected_peak_weeks,
    by = "season",
    suffix = c(
      "_calculated",
      "_expected"
    )
  )

print(peak_comparison)

stopifnot(
  all(
    peak_weeks$cases_peak_week ==
      expected_peak_weeks$cases_peak_week
  )
)

stopifnot(
  all(
    peak_weeks$hospitalizations_peak_week ==
      expected_peak_weeks$hospitalizations_peak_week
  )
)

stopifnot(
  all(
    peak_weeks$deaths_peak_week ==
      expected_peak_weeks$deaths_peak_week
  )
)

cat(
  "\nAll reported peak weeks are verified.\n"
)

# ------------------------------------------------------------
# 15. Verify the principal numerical results
# ------------------------------------------------------------

# Contemporaneous correlations
stopifnot(
  abs(
    mean_correlations$mean_cases_hospitalizations -
      0.9955456681
  ) < 1e-8
)

stopifnot(
  abs(
    mean_correlations$mean_cases_deaths -
      0.9860617264
  ) < 1e-8
)

# Cross-correlations
stopifnot(
  abs(
    case_hosp_mean$mean_correlation[
      case_hosp_mean$lag == 0
    ] -
      0.9955456681
  ) < 1e-8
)

stopifnot(
  abs(
    case_hosp_mean$mean_correlation[
      case_hosp_mean$lag == 1
    ] -
      0.8898571050
  ) < 1e-8
)

stopifnot(
  abs(
    case_hosp_mean$mean_correlation[
      case_hosp_mean$lag == 2
    ] -
      0.6472596597
  ) < 1e-8
)

stopifnot(
  abs(
    case_death_mean$mean_correlation[
      case_death_mean$lag == 0
    ] -
      0.9860617264
  ) < 1e-8
)

stopifnot(
  abs(
    case_death_mean$mean_correlation[
      case_death_mean$lag == 1
    ] -
      0.9426942195
  ) < 1e-8
)

stopifnot(
  abs(
    case_death_mean$mean_correlation[
      case_death_mean$lag == 2
    ] -
      0.7358483629
  ) < 1e-8
)

cat(
  "\nAll case-hospitalisation and case-death ",
  "correlations are verified.\n"
)

# ------------------------------------------------------------
# 16. Save reproducibility files
# ------------------------------------------------------------

write_csv(
  national_weekly,
  "national_weekly_aggregated_from_state_data.csv"
)

write_csv(
  season_correlations,
  "case_hospitalisation_death_season_correlations.csv"
)

write_csv(
  case_hosp_lags,
  "case_hospitalisation_crosscorrelations_by_season.csv"
)

write_csv(
  case_death_lags,
  "case_death_crosscorrelations_by_season.csv"
)

write_csv(
  cross_correlation_table,
  "case_severity_crosscorrelation_summary.csv"
)

write_csv(
  peak_weeks,
  "case_hospitalisation_death_peak_weeks.csv"
)

cat(
  "\nAll results and verification tables have been saved.\n"
)
