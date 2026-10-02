# MAEPiMS Challenge 2026
# Probabilistic Forecast Calibration
# Extracted from CODE NEEDED.docx for GitHub submission.
#

# ============================================================
# MAEPiMS 2026
# Probabilistic Forecast Calibration
# ============================================================

library(dplyr)
library(readr)
library(tidyr)

# ------------------------------------------------------------
# 1. Read forecast-level predictive intervals
# ------------------------------------------------------------

# Change this to the actual location of the forecast file
forecast_file <- "amem_forecasts.csv"

fc <- read_csv(forecast_file, show_col_types = FALSE)

# ------------------------------------------------------------
# 2. Basic checks
# ------------------------------------------------------------

required_cols <- c(
  "season",
  "state",
  "week",
  "observed",
  "lower50",
  "upper50",
  "lower90",
  "upper90"
)

stopifnot(all(required_cols %in% names(fc)))

# There should be 37 states/FCT x 52 weeks x 2 holdout seasons
stopifnot(nrow(fc) == 3848)

stopifnot(
  length(unique(fc$season)) == 2,
  all(table(fc$season) == 1924)
)

# ------------------------------------------------------------
# 3. Calculate interval coverage for every forecast instance
# ------------------------------------------------------------

fc_cal <- fc %>%
  mutate(
    inside_50 =
      observed >= lower50 &
      observed <= upper50,

    inside_90 =
      observed >= lower90 &
      observed <= upper90
  )

# ------------------------------------------------------------
# 4. Overall empirical coverage
# ------------------------------------------------------------

coverage_results <- tibble(
  interval = c("50%", "90%"),
  nominal = c(0.50, 0.90),
  empirical = c(
    mean(fc_cal$inside_50),
    mean(fc_cal$inside_90)
  )
) %>%
  mutate(
    coverage_error = empirical - nominal
  )

print(coverage_results)

# ------------------------------------------------------------
# 5. Expected results
# ------------------------------------------------------------

expected_results <- tibble(
  interval = c("50%", "90%"),
  nominal = c(0.50, 0.90),
  empirical_expected = c(
    0.16943866943866945,
    0.4293139293139293
  ),
  error_expected = c(
    -0.3305613305613305,
    -0.4706860706860707
  )
)

# Verify empirical coverage
stopifnot(
  all.equal(
    coverage_results$empirical,
    expected_results$empirical_expected,
    tolerance = 1e-10
  )
)

# Verify coverage error
stopifnot(
  all.equal(
    coverage_results$coverage_error,
    expected_results$error_expected,
    tolerance = 1e-10
  )
)

# ------------------------------------------------------------
# 6. Display rounded results used in the manuscript
# ------------------------------------------------------------

coverage_table <- coverage_results %>%
  mutate(
    nominal = sprintf("%.4f", nominal),
    empirical = sprintf("%.4f", empirical),
    coverage_error = sprintf("%.4f", coverage_error)
  )

print(coverage_table)

# Save results
write_csv(
  coverage_results,
  "calibration_summary_R.csv"
)
# ============================================================
# Calibration by chronological hold-out season
# ============================================================

calibration_by_season <- fc_cal %>%
  group_by(season) %>%
  summarise(
    n = n(),

    nominal_50 = 0.50,
    coverage_50 = mean(inside_50),
    error_50 = coverage_50 - nominal_50,

    nominal_90 = 0.90,
    coverage_90 = mean(inside_90),
    error_90 = coverage_90 - nominal_90,

    .groups = "drop"
  )

print(calibration_by_season)

coverage_90 <- mean(fc_cal$inside_90)

outside_90 <- 1 - coverage_90

cat(
  "Empirical 90% coverage:",
  round(coverage_90, 4),
  "\n"
)

cat(
  "Proportion outside the 90% PI:",
  round(outside_90, 4),
  "\n"
)

cat(
  "Percentage outside the 90% PI:",
  round(100 * outside_90, 2),
  "%\n"
)
library(ggplot2)

plot_data <- coverage_results %>%
  mutate(
    interval = factor(
      interval,
      levels = c("50%", "90%")
    )
  )

calibration_plot <- ggplot(
  plot_data,
  aes(
    x = nominal,
    y = empirical,
    label = interval
  )
) +

  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +

  geom_point(size = 3) +

  geom_text(
    nudge_y = 0.04,
    size = 4
  ) +

  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +

  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +

  labs(
    x = "Nominal Coverage",
    y = "Empirical Coverage",
    title = "Probabilistic Calibration of Preliminary AMEM Forecasts"
  ) +

  theme_minimal(base_size = 12)

print(calibration_plot)

ggsave(
  "figures-calibration-plot.pdf",
  calibration_plot,
  width = 7,
  height = 5,
  device = cairo_pdf
)

ggsave(
  "figures-calibration-plot.png",
  calibration_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# ============================================================
# Figure 8: National weekly influenza outcomes
# MAEPiMS Challenge 2026
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# 1. Set the path to the challenge data
data_path <- "MAEPiMS_Challenge_Data/nigeria_flu_weekly_national.csv"

# 2. Read national weekly data
national <- read_csv(data_path, show_col_types = FALSE) %>%
  filter(state == "NATIONAL") %>%
  mutate(
    season = recode(
