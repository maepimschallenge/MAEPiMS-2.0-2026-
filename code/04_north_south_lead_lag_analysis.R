# MAEPiMS Challenge 2026
# North--South Temporal Lead--Lag Structure
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ================================================================
# MAEPiMS Challenge 2026
# Results: North--South Temporal Lead--Lag Structure
# ================================================================

library(tidyverse)

# ----------------------------------------------------------------
# 1. File path
# ----------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

weekly_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_by_state.csv"
)

# ----------------------------------------------------------------
# 2. Read the supplied weekly state-level data
# ----------------------------------------------------------------

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

# Basic checks
stopifnot(nrow(weekly) == 5772)
stopifnot(
  all(c(
    "season",
    "state",
    "zone",
    "population",
    "epi_week_of_season",
    "cases"
  ) %in% names(weekly))
)

# ----------------------------------------------------------------
# 3. Define Northern and Southern Nigeria
# ----------------------------------------------------------------

north_zones <- c("NC", "NE", "NW")
south_zones <- c("SE", "SS", "SW")

weekly <- weekly %>%
  mutate(
    broad_region = case_when(
      zone %in% north_zones ~ "N",
      zone %in% south_zones ~ "S",
      TRUE ~ NA_character_
    )
  )

# Confirm all observations belong to one of the two regions
stopifnot(!any(is.na(weekly$broad_region)))

# ----------------------------------------------------------------
# 4. Aggregate weekly cases and population by broad region
# ----------------------------------------------------------------

regional <- weekly %>%
  group_by(
    season,
    broad_region,
    epi_week_of_season
  ) %>%
  summarise(
    cases = sum(cases),
    population = sum(population),
    .groups = "drop"
  ) %>%
  mutate(
    incidence_per_100k =
      100000 * cases / population,

    # Population-normalised logarithmic incidence
    Y = log1p(incidence_per_100k)
  ) %>%
  arrange(
    season,
    broad_region,
    epi_week_of_season
  )

# Check that there are 52 weeks for each region-season
check_weeks <- regional %>%
  count(season, broad_region)

print(check_weeks)

stopifnot(
  all(check_weeks$n == 52)
)

# ----------------------------------------------------------------
# 5. Identify principal epidemic peak week
# ----------------------------------------------------------------

regional_peaks <- regional %>%
  group_by(season, broad_region) %>%
  slice_max(
    order_by = cases,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup() %>%
  select(
    season,
    broad_region,
    peak_week = epi_week_of_season,
    peak_cases = cases,
    peak_incidence = incidence_per_100k
  )

print(regional_peaks)

# ----------------------------------------------------------------
# 6. Create the North--South peak timing table
# ----------------------------------------------------------------

north_south_peaks <- regional_peaks %>%
  select(
    season,
    broad_region,
    peak_week
  ) %>%
  pivot_wider(
    names_from = broad_region,
    values_from = peak_week,
    names_prefix = "peak_"
  ) %>%
  mutate(
    peak_difference =
      peak_N - peak_S
  ) %>%
  arrange(season)

print(north_south_peaks)

# Expected peak weeks:
# 2023/2024: South = 14, North = 16
# 2024/2025: South = 15, North = 17
# 2025/2026: South = 15, North = 17

expected_peaks <- tibble(
  season = c(
    "2023/2024",
    "2024/2025",
    "2025/2026"
  ),
  peak_S = c(14, 15, 15),
  peak_N = c(16, 17, 17)
)

check_peak_results <- north_south_peaks %>%
  left_join(
    expected_peaks,
    by = "season"
  )

stopifnot(
  all(check_peak_results$peak_S.x ==
        check_peak_results$peak_S.y)
)

stopifnot(
  all(check_peak_results$peak_N.x ==
        check_peak_results$peak_N.y)
)

stopifnot(
  all(check_peak_results$peak_difference == 2)
)

# ----------------------------------------------------------------
# 7. Function for cross-correlation
#
# Positive lag definition:
#
# lag = +1 means:
# Southern[t] is correlated with Northern[t + 1]
#
# Therefore Southern Nigeria leads Northern Nigeria by 1 week.
# ----------------------------------------------------------------

cross_corr_lag <- function(south, north, lag) {

  if (lag > 0) {

    south_aligned <- south[1:(length(south) - lag)]
    north_aligned <- north[(lag + 1):length(north)]

  } else if (lag < 0) {

    k <- abs(lag)

    south_aligned <- south[(k + 1):length(south)]
    north_aligned <- north[1:(length(north) - k)]

  } else {

    south_aligned <- south
    north_aligned <- north
  }

  cor(
    south_aligned,
    north_aligned,
    method = "pearson"
  )
}

# ----------------------------------------------------------------
# 8. Calculate cross-correlations across a broad lag range
# ----------------------------------------------------------------

lag_grid <- -12:12

cross_correlations <- map_dfr(
  unique(regional$season),
  function(season_i) {

    south <- regional %>%
      filter(
        season == season_i,
        broad_region == "S"
      ) %>%
      arrange(epi_week_of_season) %>%
      pull(Y)

    north <- regional %>%
      filter(
        season == season_i,
        broad_region == "N"
      ) %>%
      arrange(epi_week_of_season) %>%
      pull(Y)

    tibble(
      season = season_i,
      lag = lag_grid,
      correlation = map_dbl(
        lag_grid,
        ~ cross_corr_lag(
          south = south,
          north = north,
          lag = .x
        )
      )
    )
  }
)

# Display all correlations
print(cross_correlations)

# ----------------------------------------------------------------
# 9. Identify the maximum cross-correlation for each season
# ----------------------------------------------------------------

maximum_cross_correlation <- cross_correlations %>%
  group_by(season) %>%
  slice_max(
    order_by = correlation,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup()

print(maximum_cross_correlation)

# Expected:
#
# 2023/2024: lag = +1, r approximately 0.9355
# 2024/2025: lag = +1, r approximately 0.9435
# 2025/2026: lag = +1, r approximately 0.9361

# ----------------------------------------------------------------
# 10. Verify the reported numerical values
# ----------------------------------------------------------------

expected_correlations <- tibble(
  season = c(
    "2023/2024",
    "2024/2025",
    "2025/2026"
  ),
  expected_lag = c(1, 1, 1),
  expected_r = c(
    0.9354653359,
    0.9435175962,
    0.9361060033
  )
)

verification <- maximum_cross_correlation %>%
  select(
    season,
    observed_lag = lag,
    observed_r = correlation
  ) %>%
  left_join(
    expected_correlations,
    by = "season"
  ) %>%
  mutate(
    absolute_difference =
      abs(observed_r - expected_r),

    lag_correct =
      observed_lag == expected_lag,

    correlation_correct =
      absolute_difference < 1e-8
  )

print(verification)

stopifnot(
  all(verification$lag_correct)
)

stopifnot(
  all(verification$correlation_correct)
)

# ----------------------------------------------------------------
# 11. Produce the compact results table for the manuscript
# ----------------------------------------------------------------

manuscript_results <- north_south_peaks %>%
  select(
    season,
    southern_peak_week = peak_S,
    northern_peak_week = peak_N,
    peak_difference
  ) %>%
  left_join(
    maximum_cross_correlation %>%
      select(
        season,
        maximum_lag = lag,
        maximum_cross_correlation = correlation
      ),
    by = "season"
  ) %>%
  arrange(season)

print(manuscript_results)

# Save results
write_csv(
  manuscript_results,
  "north_south_results.csv"
)

write_csv(
  cross_correlations,
  "north_south_cross_correlations.csv"
)

# ----------------------------------------------------------------
# 12. Generate Figure: North--South lead--lag structure
# ----------------------------------------------------------------

figure_data <- cross_correlations %>%
  mutate(
    season = factor(
      season,
      levels = c(
        "2023/2024",
        "2024/2025",
        "2025/2026"
      )
    )
  )

p <- ggplot(
  figure_data,
  aes(
    x = lag,
    y = correlation
  )
) +
  geom_hline(
    yintercept = 0,
    linewidth = 0.4
  ) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 1.8
  ) +
  geom_vline(
    xintercept = 1,
    linetype = "dashed"
  ) +
  facet_wrap(
    ~ season,
    ncol = 1
  ) +
  scale_x_continuous(
    breaks = seq(-12, 12, by = 2)
  ) +
  labs(
    x = "Lag (weeks; positive = Southern region leads)",
    y = "Pearson cross-correlation",
    title = "North--South Temporal Lead--Lag Structure"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    strip.text = element_text(
      face = "bold"
    )
  )

print(p)

# Save publication-quality figure
ggsave(
  "2-north-south-leadlag.pdf",
  p,
  width = 8,
  height = 9,
  units = "in"
)

ggsave(
  "2-north-south-leadlag.png",
  p,
  width = 8,
  height = 9,
  units = "in",
  dpi = 300
)

# ----------------------------------------------------------------
# END
# ----------------------------------------------------------------
