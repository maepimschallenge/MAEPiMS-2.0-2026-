# MAEPiMS Challenge 2026
# Evidence of Multiple Epidemic Waves
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# MAEPiMS 2026
# RESULTS: Evidence of Multiple Epidemic Waves
# ============================================================

# ------------------------------------------------------------
# 1. Load packages
# ------------------------------------------------------------

required_packages <- c(
  "tidyverse",
  "readr",
  "dplyr",
  "ggplot2"
)

installed_packages <- rownames(
  installed.packages()
)

for (pkg in required_packages) {

  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }

}

library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)

# ------------------------------------------------------------
# 2. Set data directory
# ------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

# ------------------------------------------------------------
# 3. Read national weekly data
# ------------------------------------------------------------

national <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_weekly_national.csv"
  ),
  show_col_types = FALSE
)

# ------------------------------------------------------------
# 4. Inspect structure
# ------------------------------------------------------------

cat("\n====================================\n")
cat("NATIONAL WEEKLY DATA\n")
cat("====================================\n")

cat(
  "Number of observations:",
  nrow(national),
  "\n"
)

cat(
  "Number of seasons:",
  n_distinct(national$season),
  "\n"
)

cat(
  "Weeks per season:",
  n_distinct(
    national$epi_week_of_season
  ),
  "\n"
)

# Expected:
# 156 observations = 3 seasons x 52 weeks

stopifnot(
  nrow(national) == 156
)

# ------------------------------------------------------------
# 5. Sort the data
# ------------------------------------------------------------

national <- national %>%
  arrange(
    season,
    epi_week_of_season
  )

# ------------------------------------------------------------
# 6. Identify the principal peak
# ------------------------------------------------------------

primary_peaks <- national %>%
  group_by(season) %>%
  slice_max(
    order_by = cases,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup() %>%
  select(
    season,
    primary_week = epi_week_of_season,
    primary_cases = cases
  )

cat("\n====================================\n")
cat("PRIMARY NATIONAL PEAKS\n")
cat("====================================\n")

print(primary_peaks)

# ------------------------------------------------------------
# 7. Identify local maxima
# ------------------------------------------------------------
#
# A local maximum is a week whose case count is greater than
# the immediately preceding and following weeks.
#
# The secondary wave is defined as the largest local maximum
# occurring AFTER the principal peak.
# ------------------------------------------------------------

national_with_local_peaks <- national %>%

  group_by(season) %>%

  arrange(
    epi_week_of_season,
    .by_group = TRUE
  ) %>%

  mutate(

    previous_cases =
      lag(cases),

    next_cases =
      lead(cases),

    local_peak =
      cases > previous_cases &
      cases >= next_cases

  ) %>%

  ungroup()

# ------------------------------------------------------------
# 8. Identify secondary peaks
# ------------------------------------------------------------

secondary_peaks <- national_with_local_peaks %>%

  left_join(
    primary_peaks,
    by = "season"
  ) %>%

  filter(
    local_peak,
    epi_week_of_season > primary_week
  ) %>%

  group_by(season) %>%

  slice_max(
    order_by = cases,
    n = 1,
    with_ties = FALSE
  ) %>%

  ungroup() %>%

  select(
    season,
    secondary_week = epi_week_of_season,
    secondary_cases = cases
  )

cat("\n====================================\n")
cat("SECONDARY NATIONAL PEAKS\n")
cat("====================================\n")

print(secondary_peaks)

# ------------------------------------------------------------
# 9. Calculate secondary-to-primary peak ratios
# ------------------------------------------------------------

wave_results <- primary_peaks %>%

  left_join(
    secondary_peaks,
    by = "season"
  ) %>%

  mutate(

    secondary_primary_ratio =
      secondary_cases /
      primary_cases,

    secondary_primary_percent =
      100 *
      secondary_cases /
      primary_cases

  ) %>%

  arrange(season)

cat("\n====================================\n")
cat("MULTI-WAVE RESULTS\n")
cat("====================================\n")

print(
  wave_results
)

# ------------------------------------------------------------
# 10. Display results rounded as in manuscript
# ------------------------------------------------------------

wave_results_formatted <- wave_results %>%

  mutate(

    primary_cases =
      format(
        primary_cases,
        big.mark = ",",
        scientific = FALSE
      ),

    secondary_cases =
      format(
        secondary_cases,
        big.mark = ",",
        scientific = FALSE
      ),

    secondary_primary_percent =
      sprintf(
        "%.1f%%",
        secondary_primary_percent
      )

  )

cat("\n====================================\n")
cat("VALUES USED IN MANUSCRIPT\n")
cat("====================================\n")

print(
  wave_results_formatted
)

# ------------------------------------------------------------
# 11. Verify the expected numerical values
# ------------------------------------------------------------

expected_primary_weeks <- c(
  "2023/2024" = 14,
  "2024/2025" = 16,
  "2025/2026" = 15
)

expected_secondary_weeks <- c(
  "2023/2024" = 27,
  "2024/2025" = 29,
  "2025/2026" = 27
)

expected_secondary_cases <- c(
  "2023/2024" = 573265,
  "2024/2025" = 2881605,
  "2025/2026" = 831576
)

for (i in seq_len(nrow(wave_results))) {

  season_i <- wave_results$season[i]

  stopifnot(
    wave_results$primary_week[i] ==
      expected_primary_weeks[season_i]
  )

  stopifnot(
    wave_results$secondary_week[i] ==
      expected_secondary_weeks[season_i]
  )

  stopifnot(
    wave_results$secondary_cases[i] ==
      expected_secondary_cases[season_i]
  )
}

cat(
  "\nPASS: All primary and secondary peak results ",
  "match the expected MAEPiMS values.\n"
)

# ------------------------------------------------------------
# 12. Verify ratios
# ------------------------------------------------------------

cat("\n====================================\n")
cat("SECONDARY-TO-PRIMARY RATIOS\n")
cat("====================================\n")

wave_results %>%
  select(
    season,
    primary_week,
    primary_cases,
    secondary_week,
    secondary_cases,
    secondary_primary_percent
  ) %>%
  print()

# ------------------------------------------------------------
# 13. Save numerical results
# ------------------------------------------------------------

write_csv(
  wave_results,
  "/mnt/data/maepims_multiwave_results.csv"
)

cat(
  "\nSaved: /mnt/data/maepims_multiwave_results.csv\n"
)

# ============================================================
# NATIONAL MULTI-WAVE FIGURE= Evidence of Multiple Epidemic Waves
# ============================================================

library(ggplot2)

# ------------------------------------------------------------
# 1. Prepare plotting data
# ------------------------------------------------------------

plot_data <- national %>%
  select(
    season,
    epi_week_of_season,
    cases
  )

# ------------------------------------------------------------
# 2. Prepare primary peak points
# ------------------------------------------------------------

primary_plot <- primary_peaks %>%
  mutate(
    label =
      paste0(
        "Primary: W",
        primary_week
      )
  )

# ------------------------------------------------------------
# 3. Prepare secondary peak points
# ------------------------------------------------------------

secondary_plot <- secondary_peaks %>%
  mutate(
    label =
      paste0(
        "Secondary: W",
        secondary_week
      )
  )

# ------------------------------------------------------------
# 4. Generate figure
# ------------------------------------------------------------

multiwave_plot <- ggplot(
  plot_data,
  aes(
    x = epi_week_of_season,
    y = cases
  )
) +

  geom_line(
    linewidth = 1
  ) +

  geom_point(
    data = primary_plot,
    aes(
      x = primary_week,
      y = primary_cases
    ),
    size = 3
  ) +

  geom_point(
    data = secondary_plot,
    aes(
      x = secondary_week,
      y = secondary_cases
    ),
    size = 3
  ) +

  facet_wrap(
    ~season,
    ncol = 1,
    scales = "free_y"
  ) +

  labs(
    title =
      "National Influenza Epidemic Trajectories",

    subtitle =
      "Primary and secondary epidemic activity across the three simulated seasons",

    x =
      "Epidemiological week of season",

    y =
      "Weekly influenza cases"
  ) +

  theme_minimal(
    base_size = 12
  ) +

  theme(
    strip.text =
      element_text(
        face = "bold",
        size = 12
      ),

    plot.title =
      element_text(
        face = "bold"
      ),

    legend.position =
      "none"
  )

print(multiwave_plot)

# ------------------------------------------------------------
# 5. Save PDF
# ------------------------------------------------------------

ggsave(
  filename =
    "/mnt/data/2-national-multiwave.pdf",

  plot =
    multiwave_plot,

  width =
    8.5,

  height =
    10,

  units =
    "in"
)

# ------------------------------------------------------------
# 6. Save high-resolution PNG
# ------------------------------------------------------------

ggsave(
  filename =
    "/mnt/data/2-national-multiwave.png",

  plot =
    multiwave_plot,

  width =
    8.5,

  height =
    10,

  units =
    "in",

  dpi =
    300
)

cat(
  "\nFigures saved:\n",
  "/mnt/data/2-national-multiwave.pdf\n",
  "/mnt/data/2-national-multiwave.png\n"
)
