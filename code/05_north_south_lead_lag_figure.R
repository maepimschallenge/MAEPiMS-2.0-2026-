# MAEPiMS Challenge 2026
# North--South Temporal Lead--Lag Structure (figure code)
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# Figure: North--South Temporal Lead--Lag Structure
# ================================================================
# MAEPiMS Challenge 2026
# Figure: North--South Temporal Lead--Lag Relationship
# Output:
#   2-north-south-leadlag.pdf
#   2-north-south-leadlag.png
# ================================================================

# ----------------------------------------------------------------
# 1. Load packages
# ----------------------------------------------------------------

library(tidyverse)

# ----------------------------------------------------------------
# 2. Set data location
# ----------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

weekly_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_by_state.csv"
)

# ----------------------------------------------------------------
# 3. Read supplied MAEPiMS weekly state-level data
# ----------------------------------------------------------------

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

# Check dataset dimensions
stopifnot(nrow(weekly) == 5772)

# Check required variables
required_vars <- c(
  "season",
  "zone",
  "population",
  "epi_week_of_season",
  "cases"
)

stopifnot(
  all(required_vars %in% names(weekly))
)

# ----------------------------------------------------------------
# 4. Define Northern and Southern Nigeria
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

# Verify that every observation is assigned
stopifnot(
  !any(is.na(weekly$broad_region))
)

# ----------------------------------------------------------------
# 5. Aggregate states into Northern and Southern regions
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

    # Weekly incidence per 100,000 population
    incidence_per_100k =
      100000 * cases / population,

    # Log population-normalised incidence
    log_incidence =
      log1p(incidence_per_100k)

  ) %>%
  arrange(
    season,
    broad_region,
    epi_week_of_season
  )

# ----------------------------------------------------------------
# 6. Check that every region-season has 52 weeks
# ----------------------------------------------------------------

week_check <- regional %>%
  count(
    season,
    broad_region
  )

print(week_check)

stopifnot(
  all(week_check$n == 52)
)

# ----------------------------------------------------------------
# 7. Cross-correlation function
#
# Positive lag means:
#
# Southern[t] is compared with Northern[t + lag].
#
# Therefore:
#
# lag = +1
# means Southern Nigeria leads Northern Nigeria by one week.
# ----------------------------------------------------------------

cross_corr_lag <- function(south, north, lag) {

  if (lag > 0) {

    south_aligned <-
      south[1:(length(south) - lag)]

    north_aligned <-
      north[(lag + 1):length(north)]

  } else if (lag < 0) {

    k <- abs(lag)

    south_aligned <-
      south[(k + 1):length(south)]

    north_aligned <-
      north[1:(length(north) - k)]

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
# 8. Calculate cross-correlations
# ----------------------------------------------------------------

lags <- -12:12

cross_correlations <- map_dfr(
  unique(regional$season),
  function(season_i) {

    south <- regional %>%
      filter(
        season == season_i,
        broad_region == "S"
      ) %>%
      arrange(epi_week_of_season) %>%
      pull(log_incidence)

    north <- regional %>%
      filter(
        season == season_i,
        broad_region == "N"
      ) %>%
      arrange(epi_week_of_season) %>%
      pull(log_incidence)

    tibble(
      season = season_i,
      lag = lags,
      correlation = map_dbl(
        lags,
        ~ cross_corr_lag(
          south = south,
          north = north,
          lag = .x
        )
      )
    )
  }
)

# ----------------------------------------------------------------
# 9. Identify the maximum correlation in each season
# ----------------------------------------------------------------

maximum_correlation <- cross_correlations %>%
  group_by(season) %>%
  slice_max(
    order_by = correlation,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup()

print(maximum_correlation)

# ----------------------------------------------------------------
# 10. Check expected maximum lag
# ----------------------------------------------------------------

stopifnot(
  all(maximum_correlation$lag == 1)
)

# Display numerical results
print(
  maximum_correlation %>%
    select(
      season,
      lag,
      correlation
    )
)

# ----------------------------------------------------------------
# 11. Calculate principal epidemic peak weeks
# ----------------------------------------------------------------

regional_peak <- regional %>%
  group_by(
    season,
    broad_region
  ) %>%
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
    peak_cases = cases
  )

print(regional_peak)

# ----------------------------------------------------------------
# 12. Verify the two-week Southern lead in peak timing
# ----------------------------------------------------------------

peak_comparison <- regional_peak %>%
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
    difference = peak_N - peak_S
  )

print(peak_comparison)

# Expected:
#
# 2023/2024: South 14, North 16
# 2024/2025: South 15, North 17
# 2025/2026: South 15, North 17

stopifnot(
  all(peak_comparison$difference == 2)
)

# ----------------------------------------------------------------
# 13. Prepare data for plotting
# ----------------------------------------------------------------

plot_data <- cross_correlations %>%
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

# ----------------------------------------------------------------
# 14. Create the North--South lead--lag figure
# ----------------------------------------------------------------

p_north_south <- ggplot(
  plot_data,
  aes(
    x = lag,
    y = correlation
  )
) +

  # Zero-correlation reference line
  geom_hline(
    yintercept = 0,
    linewidth = 0.4
  ) +

  # Cross-correlation curve
  geom_line(
    linewidth = 0.8
  ) +

  # Individual lag correlations
  geom_point(
    size = 1.8
  ) +

  # Highlight the one-week positive lag
  geom_vline(
    xintercept = 1,
    linetype = "dashed",
    linewidth = 0.5
  ) +

  # Separate panel for each season
  facet_wrap(
    ~ season,
    ncol = 1
  ) +

  # Lag axis
  scale_x_continuous(
    breaks = seq(-12, 12, by = 2)
  ) +

  # Labels
  labs(
    x = "Lag (weeks; positive = Southern region leads)",
    y = "Pearson cross-correlation",
    title = "North--South Temporal Lead--Lag Structure"
  ) +

  # Publication-style theme
  theme_bw() +

  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),

    strip.text = element_text(
      face = "bold"
    ),

    axis.title = element_text(
      face = "bold"
    ),

    panel.grid.minor = element_blank()
  )

# Display figure
print(p_north_south)

# ----------------------------------------------------------------
# 15. Save PDF for LaTeX
# ----------------------------------------------------------------

ggsave(
  filename = "2-north-south-leadlag.pdf",
  plot = p_north_south,
  width = 8,
  height = 9,
  units = "in"
)

# ----------------------------------------------------------------
# 16. Also save high-resolution PNG
# ----------------------------------------------------------------

ggsave(
  filename = "2-north-south-leadlag.png",
  plot = p_north_south,
  width = 8,
  height = 9,
  units = "in",
  dpi = 300
)

# ----------------------------------------------------------------
# 17. Save numerical cross-correlation results
# ----------------------------------------------------------------

write_csv(
  cross_correlations,
  "north_south_cross_correlations.csv"
)

# Save maximum values
write_csv(
  maximum_correlation,
  "north_south_maximum_correlations.csv"
)

# Save peak-week comparison
write_csv(
  peak_comparison,
  "north_south_peak_weeks.csv"
)

# ----------------------------------------------------------------
# END OF CODE
# ----------------------------------------------------------------
