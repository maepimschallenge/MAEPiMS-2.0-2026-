# MAEPiMS Challenge 2026
# Peak and Seasonal Burden Forecasting
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# Observed seasonal epidemic characteristics.
# ============================================================
# RESULTS: OBSERVED NATIONAL SEASONAL TARGETS
# ============================================================

library(tidyverse)

# ------------------------------------------------------------
# 1. File path
# ------------------------------------------------------------

data_path <- "MAEPiMS_Challenge_Data/nigeria_flu_weekly_national.csv"

nat <- read_csv(data_path, show_col_types = FALSE)

# ------------------------------------------------------------
# 2. Basic data checks
# ------------------------------------------------------------

stopifnot(
  nrow(nat) == 156,
  n_distinct(nat$season) == 3,
  all(table(nat$season) == 52)
)

stopifnot(
  all(nat$state == "NATIONAL")
)

# National population
P_N <- unique(nat$population)

stopifnot(length(P_N) == 1)

# ------------------------------------------------------------
# 3. Calculate observed seasonal targets directly
# ------------------------------------------------------------

seasonal_observed <- nat %>%
  group_by(season) %>%
  summarise(
    peak_week = epi_week_of_season[
      which.max(cases)
    ],

    peak_incidence =
      max(cases_per_100k, na.rm = TRUE),

    cumulative_cases =
      sum(cases, na.rm = TRUE),

    cumulative_hospitalisations =
      sum(hospitalizations, na.rm = TRUE),

    cumulative_deaths =
      sum(deaths, na.rm = TRUE),

    attack_rate =
      100 * cumulative_cases / first(population),

    .groups = "drop"
  )

# ------------------------------------------------------------
# 4. Display results
# ------------------------------------------------------------

print(seasonal_observed)

# ------------------------------------------------------------
# 5. Round results for manuscript
# ------------------------------------------------------------

seasonal_observed_table <- seasonal_observed %>%
  mutate(
    peak_incidence = round(peak_incidence, 2),
    attack_rate = round(attack_rate, 2)
  )

print(seasonal_observed_table)

# ------------------------------------------------------------
# 6. Verify the exact manuscript values
# ------------------------------------------------------------

expected_observed <- tibble(
  season = c(
    "2023/2024",
    "2024/2025",
    "2025/2026"
  ),
  peak_week = c(14, 16, 15),
  peak_incidence = c(
    1154.98,
    1920.80,
    1561.23
  ),
  cumulative_cases = c(
    17115135,
    31416391,
    22239254
  ),
  cumulative_hospitalisations = c(
    267587,
    455557,
    327633
  ),
  cumulative_deaths = c(
    24443,
    42412,
    30603
  ),
  attack_rate = c(
    7.46,
    13.69,
    9.69
  )
)

comparison <- seasonal_observed_table %>%
  left_join(expected_observed, by = "season",
            suffix = c("_calculated", "_expected")) %>%
  mutate(
    peak_week_ok =
      peak_week_calculated == peak_week_expected,

    peak_incidence_ok =
      abs(peak_incidence_calculated -
            peak_incidence_expected) < 0.01,

    cumulative_cases_ok =
      cumulative_cases_calculated ==
      cumulative_cases_expected,

    cumulative_hosp_ok =
      cumulative_hospitalisations_calculated ==
      cumulative_hospitalisations_expected,

    cumulative_deaths_ok =
      cumulative_deaths_calculated ==
      cumulative_deaths_expected,

    attack_rate_ok =
      abs(attack_rate_calculated -
            attack_rate_expected) < 0.01
  )

print(comparison)

stopifnot(
  all(comparison$peak_week_ok),
  all(comparison$peak_incidence_ok),
  all(comparison$cumulative_cases_ok),
  all(comparison$cumulative_hosp_ok),
  all(comparison$cumulative_deaths_ok),
  all(comparison$attack_rate_ok)
)

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

write_csv(
  seasonal_observed_table,
  "seasonal_observed_targets.csv"
)
RESULTS: PEAK FORECAST VERIFICATION
# ============================================================
# RESULTS: PEAK FORECAST VERIFICATION
# ============================================================

library(tidyverse)

peak_forecasts <- tribble(
  ~season,      ~target,           ~observed, ~median, ~pi50_low, ~pi50_high,
  ~pi90_low, ~pi90_high,

  "2024/2025", "Peak week",
  16, 14, 14, 15, 13, 15,

  "2024/2025", "Peak incidence",
  1920.80, 526.85, 503.48, 554.98,
  474.69, 601.59,

  "2025/2026", "Peak week",
  15, 16, 15, 16, 15, 17,

  "2025/2026", "Peak incidence",
  1561.23, 661.25, 626.81, 696.81,
  585.68, 763.77
)

# ------------------------------------------------------------
# Calculate point errors
# ------------------------------------------------------------

peak_forecast_errors <- peak_forecasts %>%
  mutate(
    absolute_error = abs(observed - median),
    relative_error_pct =
      100 * absolute_error / observed
  )

print(peak_forecast_errors)

# ------------------------------------------------------------
# Expected errors
# ------------------------------------------------------------

expected_peak_errors <- tribble(
  ~season,      ~target,           ~absolute_error,
  ~relative_error_pct,

  "2024/2025", "Peak week",
  2, 12.50,

  "2024/2025", "Peak incidence",
  1393.95, 72.5713,

  "2025/2026", "Peak week",
  1, 6.6667,

  "2025/2026", "Peak incidence",
  899.98, 57.6456
)

check_peak <- peak_forecast_errors %>%
  left_join(
    expected_peak_errors,
    by = c("season", "target"),
    suffix = c("_calculated", "_expected")
  )

print(check_peak)

stopifnot(
  all(
    abs(
      check_peak$absolute_error_calculated -
      check_peak$absolute_error_expected
    ) < 0.01
  )
)

stopifnot(
  all(
    abs(
      check_peak$relative_error_pct_calculated -
      check_peak$relative_error_pct_expected
    ) < 0.01
  )
)

write_csv(
  peak_forecast_errors,
  "peak_forecast_errors.csv"
)
RESULTS: SEASONAL BURDEN FORECAST VERIFICATION
# ============================================================
# RESULTS: SEASONAL BURDEN FORECAST VERIFICATION
# ============================================================

library(tidyverse)

burden_forecasts <- tribble(
  ~season, ~target, ~observed, ~median,
  ~pi50_low, ~pi50_high, ~pi90_low, ~pi90_high,

  "2024/2025", "Cumulative cases",
  31416391, 11262330,
  11095900, 11413820,
  10877690, 11700740,

  "2024/2025", "Cumulative hospitalisations",
  455557, 176645,
  167138, 186326,
  155201, 200591,

  "2024/2025", "Cumulative deaths",
  42412, 16154,
  15105, 17488,
  13490, 19246,

  "2024/2025", "Attack rate (%)",
  13.69, 4.91,
  4.83, 4.97,
  4.74, 5.10,

  "2025/2026", "Cumulative cases",
  22239254, 13267390,
  13051210, 13507730,
  12750270, 13849540,

  "2025/2026", "Cumulative hospitalisations",
  327633, 199647,
  191954, 207753,
  182112, 219608,

  "2025/2026", "Cumulative deaths",
  30603, 18412,
  17658, 19144,
  16604, 20486,

  "2025/2026", "Attack rate (%)",
  9.69, 5.78,
  5.69, 5.89,
  5.56, 6.03
)

# ------------------------------------------------------------
# Calculate absolute and relative errors
# ------------------------------------------------------------

burden_errors <- burden_forecasts %>%
  mutate(
    absolute_error = abs(observed - median),
    relative_error_pct =
      100 * absolute_error / observed
  )

print(burden_errors)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

write_csv(
  burden_errors,
  "seasonal_burden_forecast_errors.csv"
)
ATTACK RATE INTERNAL CONSISTENCY CHECK
# ============================================================
# ATTACK RATE INTERNAL CONSISTENCY CHECK
# ============================================================

P_N <- 229501935

attack_check <- tibble(
  season = c("2024/2025", "2025/2026"),

  cases_median = c(
    11262330,
    13267390
  ),

  cases_pi50_low = c(
    11095900,
    13051210
  ),

  cases_pi50_high = c(
    11413820,
    13507730
  ),

  cases_pi90_low = c(
    10877690,
    12750270
  ),

  cases_pi90_high = c(
    11700740,
    13849540
  ),

  reported_attack_median = c(
    4.91,
    5.78
  ),

  reported_attack50_low = c(
    4.83,
    5.69
  ),

  reported_attack50_high = c(
    4.97,
    5.89
  ),

  reported_attack90_low = c(
    4.74,
    5.56
  ),

  reported_attack90_high = c(
    5.10,
    6.03
  )
)

attack_check <- attack_check %>%
  mutate(
    calculated_attack_median =
      100 * cases_median / P_N,

    calculated_attack50_low =
      100 * cases_pi50_low / P_N,

    calculated_attack50_high =
      100 * cases_pi50_high / P_N,

    calculated_attack90_low =
      100 * cases_pi90_low / P_N,

    calculated_attack90_high =
      100 * cases_pi90_high / P_N
  )

print(attack_check)

# Check rounded agreement
stopifnot(
  all(
    round(attack_check$calculated_attack_median, 2) ==
    attack_check$reported_attack_median
  )
)

stopifnot(
  all(
    round(attack_check$calculated_attack50_low, 2) ==
    attack_check$reported_attack50_low
  )
)

stopifnot(
  all(
    round(attack_check$calculated_attack50_high, 2) ==
    attack_check$reported_attack50_high
  )
)

stopifnot(
  all(
    round(attack_check$calculated_attack90_low, 2) ==
    attack_check$reported_attack90_low
  )
)

stopifnot(
  all(
    round(attack_check$calculated_attack90_high, 2) ==
    attack_check$reported_attack90_high
  )
)
DERIVE SEASONAL TARGETS FROM AMEM PREDICTIVE TRAJECTORIES
# ============================================================
# DERIVE SEASONAL TARGETS FROM AMEM PREDICTIVE TRAJECTORIES
# ============================================================
# NOTE:
# This section assumes that `forecast_draws` has already been generated
# by the forecasting implementation. This script derives and summarizes
# seasonal targets from those predictive trajectories; it does not
# generate the underlying forecasts.

library(tidyverse)

# ------------------------------------------------------------
# 1. National aggregation of predictive draws
# ------------------------------------------------------------

national_draws <- forecast_draws %>%
  group_by(
    draw,
    season,
    week
  ) %>%
  summarise(
    cases = sum(cases, na.rm = TRUE),
    hospitalizations =
      sum(hospitalizations, na.rm = TRUE),
    deaths =
      sum(deaths, na.rm = TRUE),
    population =
      sum(population, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 2. Derive seasonal targets for every predictive trajectory
# ------------------------------------------------------------

seasonal_draw_targets <- national_draws %>%
  group_by(draw, season) %>%
  summarise(

    # Peak week
    peak_week =
      week[which.max(cases)],

    # Peak incidence
    peak_incidence =
      max(
        100000 * cases / population,
        na.rm = TRUE
      ),

    # Cumulative burden
    cumulative_cases =
      sum(cases, na.rm = TRUE),

    cumulative_hospitalisations =
      sum(
        hospitalizations,
        na.rm = TRUE
      ),

    cumulative_deaths =
      sum(
        deaths,
        na.rm = TRUE
      ),

    .groups = "drop"
  ) %>%
  mutate(
    attack_rate =
      100 * cumulative_cases / population
  )

# ------------------------------------------------------------
# 3. Predictive summaries
# ------------------------------------------------------------

seasonal_predictive_summary <- seasonal_draw_targets %>%
  group_by(season) %>%
  summarise(

    peak_week_median =
      median(peak_week),

    peak_week_pi50_low =
      quantile(peak_week, 0.25),

    peak_week_pi50_high =
      quantile(peak_week, 0.75),

    peak_week_pi90_low =
      quantile(peak_week, 0.05),

    peak_week_pi90_high =
      quantile(peak_week, 0.95),

    peak_incidence_median =
      median(peak_incidence),

    peak_incidence_pi50_low =
      quantile(peak_incidence, 0.25),

    peak_incidence_pi50_high =
      quantile(peak_incidence, 0.75),

    peak_incidence_pi90_low =
      quantile(peak_incidence, 0.05),

    peak_incidence_pi90_high =
      quantile(peak_incidence, 0.95),

    cumulative_cases_median =
      median(cumulative_cases),

    cumulative_cases_pi50_low =
      quantile(cumulative_cases, 0.25),

    cumulative_cases_pi50_high =
      quantile(cumulative_cases, 0.75),

    cumulative_cases_pi90_low =
      quantile(cumulative_cases, 0.05),

    cumulative_cases_pi90_high =
      quantile(cumulative_cases, 0.95),

    cumulative_hospitalisations_median =
      median(cumulative_hospitalisations),

    cumulative_hospitalisations_pi50_low =
      quantile(cumulative_hospitalisations, 0.25),

    cumulative_hospitalisations_pi50_high =
      quantile(cumulative_hospitalisations, 0.75),

    cumulative_hospitalisations_pi90_low =
      quantile(cumulative_hospitalisations, 0.05),

    cumulative_hospitalisations_pi90_high =
      quantile(cumulative_hospitalisations, 0.95),

    cumulative_deaths_median =
      median(cumulative_deaths),

    cumulative_deaths_pi50_low =
      quantile(cumulative_deaths, 0.25),

    cumulative_deaths_pi50_high =
      quantile(cumulative_deaths, 0.75),

    cumulative_deaths_pi90_low =
      quantile(cumulative_deaths, 0.05),

    cumulative_deaths_pi90_high =
      quantile(cumulative_deaths, 0.95),

    attack_rate_median =
      median(attack_rate),

    attack_rate_pi50_low =
      quantile(attack_rate, 0.25),

    attack_rate_pi50_high =
      quantile(attack_rate, 0.75),

    attack_rate_pi90_low =
      quantile(attack_rate, 0.05),

    attack_rate_pi90_high =
      quantile(attack_rate, 0.95),

    .groups = "drop"
  )

print(seasonal_predictive_summary)

write_csv(
  seasonal_predictive_summary,
  "seasonal_predictive_summary.csv"
)
