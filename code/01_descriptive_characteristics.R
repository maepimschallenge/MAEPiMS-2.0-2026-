# MAEPiMS Challenge 2026
# Descriptive Characteristics of the Influenza Epidemic
# Extracted from CODE NEEDED.docx for GitHub submission.
#


# ------------------------------------------------------------
# 0. Install/load required packages
# ------------------------------------------------------------

packages <- c(
  "tidyverse",
  "readr",
  "dplyr",
  "stringr"
)

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p)
  }
}

library(tidyverse)
library(readr)
library(dplyr)
library(stringr)

# ------------------------------------------------------------
# 1. Locate the uploaded ZIP file
# ------------------------------------------------------------

zip_file <- "/mnt/data/MAEPiMS_Challenge_Data(2).zip"

# Directory where files will be extracted
extract_dir <- "/mnt/data/maepims_results_data"

if (!dir.exists(extract_dir)) {
  dir.create(extract_dir, recursive = TRUE)
}

# Extract ZIP file
unzip(
  zip_file,
  exdir = extract_dir
)

# ------------------------------------------------------------
# 2. Locate the MAEPiMS data folder
# ------------------------------------------------------------

data_dir <- file.path(
  extract_dir,
  "MAEPiMS_Challenge_Data"
)

# Check available files
print(list.files(data_dir))

# ------------------------------------------------------------
# 3. Read the supplied MAEPiMS datasets
# ------------------------------------------------------------

weekly_state <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_weekly_by_state.csv"
  ),
  show_col_types = FALSE
)

weekly_national <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_weekly_national.csv"
  ),
  show_col_types = FALSE
)

season_reference <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_season_reference.csv"
  ),
  show_col_types = FALSE
)

state_metadata <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_state_metadata.csv"
  ),
  show_col_types = FALSE
)

season_summary <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_season_summary.csv"
  ),
  show_col_types = FALSE
)

# ------------------------------------------------------------
# 4. Inspect the data
# ------------------------------------------------------------

cat("\n==============================\n")
cat("DATA STRUCTURE\n")
cat("==============================\n")

cat(
  "State weekly observations:",
  nrow(weekly_state),
  "\n"
)

cat(
  "National weekly observations:",
  nrow(weekly_national),
  "\n"
)

cat(
  "Number of states/FCT:",
  n_distinct(weekly_state$state),
  "\n"
)

cat(
  "Number of seasons:",
  n_distinct(weekly_state$season),
  "\n"
)

cat(
  "Weeks per season:",
  n_distinct(weekly_state$epi_week_of_season),
  "\n"
)

# ------------------------------------------------------------
# 5. Verify 37 x 3 x 52 = 5,772
# ------------------------------------------------------------

n_states <- n_distinct(weekly_state$state)

n_seasons <- n_distinct(weekly_state$season)

n_weeks <- n_distinct(
  weekly_state$epi_week_of_season
)

observed_rows <- nrow(weekly_state)

expected_rows <- (
  n_states *
  n_seasons *
  n_weeks
)

cat("\n==============================\n")
cat("STATE-LEVEL DATA DIMENSIONS\n")
cat("==============================\n")

cat(
  n_states,
  "states/FCT x ",
  n_seasons,
  " seasons x ",
  n_weeks,
  " weeks = ",
  expected_rows,
  " observations\n",
  sep = ""
)

cat(
  "Observed observations = ",
  observed_rows,
  "\n",
  sep = ""
)

# Stop if dimensions are not as expected
stopifnot(observed_rows == expected_rows)

# ------------------------------------------------------------
# 6. Calculate national population
# ------------------------------------------------------------

national_population <- sum(
  state_metadata$population,
  na.rm = TRUE
)

cat("\n==============================\n")
cat("NATIONAL POPULATION\n")
cat("==============================\n")

cat(
  "P_N = ",
  format(national_population, big.mark = ","),
  "\n",
  sep = ""
)

stopifnot(
  national_population == 229501935
)

# ------------------------------------------------------------
# 7. Display season classifications
# ------------------------------------------------------------

cat("\n==============================\n")
cat("SEASON REFERENCE\n")
cat("==============================\n")

print(season_reference)

# ------------------------------------------------------------
# 8. Calculate national seasonal totals
# ------------------------------------------------------------

season_totals <- weekly_national %>%
  group_by(season) %>%
  summarise(
    total_cases = sum(cases, na.rm = TRUE),

    total_hospitalizations =
      sum(hospitalizations, na.rm = TRUE),

    total_deaths =
      sum(deaths, na.rm = TRUE),

    .groups = "drop"
  )

cat("\n==============================\n")
cat("NATIONAL SEASONAL TOTALS\n")
cat("==============================\n")

print(season_totals)

# ------------------------------------------------------------
# 9. Calculate national peak week and peak cases
# ------------------------------------------------------------

national_peaks <- weekly_national %>%
  group_by(season) %>%
  slice_max(
    order_by = cases,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup() %>%
  select(
    season,
    peak_week = epi_week_of_season,
    peak_cases = cases,
    peak_incidence_per_100k = cases_per_100k
  )

cat("\n==============================\n")
cat("NATIONAL PEAKS\n")
cat("==============================\n")

print(national_peaks)

# ------------------------------------------------------------
# 10. Calculate seasonal attack rates
# ------------------------------------------------------------

season_totals <- season_totals %>%
  mutate(
    attack_rate_pct =
      total_cases /
      national_population *
      100
  )

cat("\n==============================\n")
cat("SEASONAL ATTACK RATES\n")
cat("==============================\n")

season_totals %>%
  select(
    season,
    attack_rate_pct
  ) %>%
  mutate(
    attack_rate_pct =
      round(attack_rate_pct, 2)
  ) %>%
  print()

# ------------------------------------------------------------
# 11. Combine all national results
# ------------------------------------------------------------

results_table <- season_totals %>%
  left_join(
    national_peaks,
    by = "season"
  ) %>%
  select(
    season,
    total_cases,
    total_hospitalizations,
    total_deaths,
    peak_week,
    peak_cases,
    peak_incidence_per_100k,
    attack_rate_pct
  ) %>%
  arrange(season)

cat("\n==============================\n")
cat("FINAL RESULTS TABLE\n")
cat("==============================\n")

print(results_table)

# ------------------------------------------------------------
# 12. Format the table for easy copying into the paper
# ------------------------------------------------------------

formatted_table <- results_table %>%
  mutate(
    total_cases =
      format(total_cases, big.mark = ",", scientific = FALSE),

    total_hospitalizations =
      format(
        total_hospitalizations,
        big.mark = ",",
        scientific = FALSE
      ),

    total_deaths =
      format(
        total_deaths,
        big.mark = ",",
        scientific = FALSE
      ),

    peak_cases =
      format(
        peak_cases,
        big.mark = ",",
        scientific = FALSE
      ),

    peak_incidence_per_100k =
      round(
        peak_incidence_per_100k,
        2
      ),

    attack_rate_pct =
      paste0(
        round(attack_rate_pct, 2),
        "%"
      )
  )

cat("\n==============================\n")
cat("FORMATTED TABLE\n")
cat("==============================\n")

print(formatted_table)

# ------------------------------------------------------------
# 13. Verify state-level data aggregate to national data
# ------------------------------------------------------------

state_aggregate <- weekly_state %>%
  group_by(
    season,
    epi_week_of_season
  ) %>%
  summarise(
    cases_state =
      sum(cases, na.rm = TRUE),

    hosp_state =
      sum(hospitalizations, na.rm = TRUE),

    deaths_state =
      sum(deaths, na.rm = TRUE),

    .groups = "drop"
  )

national_reference <- weekly_national %>%
  select(
    season,
    epi_week_of_season,
    cases,
    hospitalizations,
    deaths
  ) %>%
  rename(
    cases_national = cases,
    hosp_national = hospitalizations,
    deaths_national = deaths
  )

aggregation_check <- state_aggregate %>%
  inner_join(
    national_reference,
    by = c(
      "season",
      "epi_week_of_season"
    )
  ) %>%
  mutate(
    case_difference =
      cases_state - cases_national,

    hosp_difference =
      hosp_state - hosp_national,

    death_difference =
      deaths_state - deaths_national
  )

# ------------------------------------------------------------
# 14. Report aggregation accuracy
# ------------------------------------------------------------

max_case_difference <- max(
  abs(aggregation_check$case_difference)
)

max_hosp_difference <- max(
  abs(aggregation_check$hosp_difference)
)

max_death_difference <- max(
  abs(aggregation_check$death_difference)
)

cat("\n==============================\n")
cat("STATE -> NATIONAL CONSISTENCY\n")
cat("==============================\n")

cat(
  "Maximum absolute case difference: ",
  max_case_difference,
  "\n",
  sep = ""
)

cat(
  "Maximum absolute hospitalisation difference: ",
  max_hosp_difference,
  "\n",
  sep = ""
)

cat(
  "Maximum absolute death difference: ",
  max_death_difference,
  "\n",
  sep = ""
)

# Exact consistency checks
stopifnot(max_case_difference == 0)
stopifnot(max_hosp_difference == 0)
stopifnot(max_death_difference == 0)

cat(
  "\nPASS: State-level weekly observations ",
  "aggregate exactly to the supplied national ",
  "weekly observations.\n"
)

# ------------------------------------------------------------
# 15. Verify the three seasonal totals independently
# ------------------------------------------------------------

cat("\n==============================\n")
cat("SEASONAL PEAK SUMMARY\n")
cat("==============================\n")

print(
  results_table %>%
    select(
      season,
      total_cases,
      total_hospitalizations,
      total_deaths,
      peak_week,
      peak_cases,
      attack_rate_pct
    )
)

# ------------------------------------------------------------
# 16. Save results for use in the manuscript
# ------------------------------------------------------------

write_csv(
  results_table,
  "/mnt/data/maepims_results_season_overview.csv"
)

write_csv(
  aggregation_check,
  "/mnt/data/maepims_state_national_aggregation_check.csv"
)

cat("\nResults saved to:\n")
cat(
  "/mnt/data/maepims_results_season_overview.csv\n"
)

cat(
  "/mnt/data/maepims_state_national_aggregation_check.csv\n"
)
