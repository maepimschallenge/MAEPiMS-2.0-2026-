# MAEPiMS Challenge 2026
# Climate-Related Differences in Epidemic Timing
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# RESULTS: Climate-Related Differences in Epidemic Timing
# MAEPiMS Challenge 2026
# ============================================================

library(tidyverse)

# ------------------------------------------------------------
# 1. File path
# ------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

weekly_file <- file.path(
  data_dir,
  "nigeria_flu_weekly_by_state.csv"
)

# ------------------------------------------------------------
# 2. Read weekly state-level data
# ------------------------------------------------------------

weekly <- read_csv(
  weekly_file,
  show_col_types = FALSE
)

# ------------------------------------------------------------
# 3. Basic data checks
# ------------------------------------------------------------

stopifnot(nrow(weekly) == 5772)

required_vars <- c(
  "season",
  "state",
  "climate",
  "epi_week_of_season",
  "cases"
)

stopifnot(all(required_vars %in% names(weekly)))

# Check that every state-season has exactly 52 weeks
week_check <- weekly %>%
  group_by(state, season) %>%
  summarise(
    n_weeks = n_distinct(epi_week_of_season),
    .groups = "drop"
  )

stopifnot(all(week_check$n_weeks == 52))

# ------------------------------------------------------------
# 4. Check climate categories
# ------------------------------------------------------------

climate_counts <- weekly %>%
  distinct(state, climate) %>%
  count(climate)

print(climate_counts)

# Expected:
# tropical = 17
# savanna  = 10
# sahel    = 10

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

# ------------------------------------------------------------
# 5. Identify the peak week for every state-season
# ------------------------------------------------------------

state_season_peaks <- weekly %>%
  group_by(state, season) %>%
  slice_max(
    order_by = cases,
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup() %>%
  select(
    season,
    state,
    climate,
    epi_week_of_season,
    cases
  ) %>%
  rename(
    peak_week = epi_week_of_season,
    peak_cases = cases
  )

# ------------------------------------------------------------
# 6. Verify that there is exactly one peak for every
#    state-season combination
# ------------------------------------------------------------

stopifnot(
  nrow(state_season_peaks) == 37 * 3
)

stopifnot(
  n_distinct(
    paste(
      state_season_peaks$state,
      state_season_peaks$season
    )
  ) == 111
)

# ------------------------------------------------------------
# 7. Calculate mean peak week by climate and season
# ------------------------------------------------------------

climate_peak_results <- state_season_peaks %>%
  group_by(season, climate) %>%
  summarise(
    n_states = n(),
    mean_peak_week = mean(peak_week),
    sd_peak_week = sd(peak_week),
    .groups = "drop"
  ) %>%
  arrange(
    season,
    factor(
      climate,
      levels = c("tropical", "savanna", "sahel")
    )
  )

# Display full results
print(climate_peak_results)

# ------------------------------------------------------------
# 8. Create the manuscript table
# ------------------------------------------------------------

climate_peak_table <- climate_peak_results %>%
  select(
    season,
    climate,
    mean_peak_week
  ) %>%
  mutate(
    mean_peak_week = round(mean_peak_week, 2)
  ) %>%
  pivot_wider(
    names_from = climate,
    values_from = mean_peak_week
  ) %>%
  select(
    season,
    tropical,
    savanna,
    sahel
  )

print(climate_peak_table)

# ------------------------------------------------------------
# 9. Verify the exact reported values
# ------------------------------------------------------------

expected <- tribble(
  ~season,      ~tropical, ~savanna, ~sahel,
  "2023/2024",      13.82,     15.00,  16.70,
  "2024/2025",      15.24,     17.60,  20.40,
  "2025/2026",      15.35,     16.70,  17.70
)

comparison <- climate_peak_table %>%
  left_join(
    expected,
    by = "season",
    suffix = c("_calculated", "_expected")
  )

print(comparison)

# Check equality after rounding
stopifnot(
  all.equal(
    climate_peak_table$tropical,
    expected$tropical,
    tolerance = 1e-10
  )
)

stopifnot(
  all.equal(
    climate_peak_table$savanna,
    expected$savanna,
    tolerance = 1e-10
  )
)

stopifnot(
  all.equal(
    climate_peak_table$sahel,
    expected$sahel,
    tolerance = 1e-10
  )
)

cat("\nAll reported climate peak-week values are verified.\n")

# ------------------------------------------------------------
# 10. Save results
# ------------------------------------------------------------

write_csv(
  state_season_peaks,
  "climate_state_season_peak_weeks.csv"
)

write_csv(
  climate_peak_results,
  "climate_peak_week_results_detailed.csv"
)

write_csv(
  climate_peak_table,
  "climate_peak_week_table.csv"
)

FIGURE=Climate-Related Differences in Epidemic Timing
# ============================================================
# FIGURE: State-level peak weeks by climate classification
# ============================================================

library(tidyverse)

climate_plot_data <- state_season_peaks %>%
  mutate(
    climate = factor(
      climate,
      levels = c("tropical", "savanna", "sahel")
    )
  )

p_climate <- ggplot(
  climate_plot_data,
  aes(
    x = climate,
    y = peak_week
  )
) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0.10,
    height = 0,
    size = 2,
    alpha = 0.65
  ) +
  facet_wrap(
    ~season,
    ncol = 1
  ) +
  scale_y_continuous(
    breaks = seq(1, 52, by = 4),
    limits = c(1, 52)
  ) +
  labs(
    title = "State-Level Influenza Peak Timing by Climate Classification",
    x = "Climate classification",
    y = "Peak epidemiological week"
  ) +
  theme_minimal(base_size = 12)

print(p_climate)

ggsave(
  "climate_peak_timing.pdf",
  p_climate,
  width = 8,
  height = 9
)

ggsave(
  "climate_peak_timing.png",
  p_climate,
  width = 8,
  height = 9,
  dpi = 300
)
