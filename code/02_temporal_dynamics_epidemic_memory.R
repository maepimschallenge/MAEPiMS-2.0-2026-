# MAEPiMS Challenge 2026
# Temporal Dynamics and Epidemic Memory
# Extracted from CODE NEEDED.docx for GitHub submission.
#


library(tidyverse)
library(readr)

# ------------------------------------------------------------
# 1. Read the supplied state-level weekly dataset
# ------------------------------------------------------------

data_dir <- "/mnt/data/maepims_check/MAEPiMS_Challenge_Data"

flu <- read_csv(
  file.path(
    data_dir,
    "nigeria_flu_weekly_by_state.csv"
  ),
  show_col_types = FALSE
)

# ------------------------------------------------------------
# 2. Check the dimensions
# ------------------------------------------------------------

n_states <- flu %>%
  distinct(state) %>%
  nrow()

n_seasons <- flu %>%
  distinct(season) %>%
  nrow()

n_state_season <- flu %>%
  distinct(state, season) %>%
  nrow()

cat("Number of states/FCT:", n_states, "\n")
cat("Number of seasons:", n_seasons, "\n")
cat("Number of state-season series:", n_state_season, "\n")

stopifnot(n_states == 37)
stopifnot(n_seasons == 3)
stopifnot(n_state_season == 111)

# ------------------------------------------------------------
# 3. Construct log population-normalised incidence
# ------------------------------------------------------------

flu <- flu %>%
  arrange(
    state,
    season,
    epi_week_of_season
  ) %>%
  mutate(
    Y = log(
      1 +
        (100000 * cases / population)
    )
  )

# ------------------------------------------------------------
# 4. Function for lag-k autocorrelation
# ------------------------------------------------------------

lag_correlation <- function(x, k) {

  x_current <- x[(k + 1):length(x)]
  x_lagged  <- x[1:(length(x) - k)]

  cor(
    x_current,
    x_lagged,
    method = "pearson",
    use = "complete.obs"
  )
}

# ------------------------------------------------------------
# 5. Calculate autocorrelation for every state-season
# ------------------------------------------------------------

acf_results <- flu %>%
  group_by(
    state,
    season
  ) %>%
  summarise(

    lag1 = lag_correlation(Y, 1),

    lag2 = lag_correlation(Y, 2),

    lag3 = lag_correlation(Y, 3),

    lag4 = lag_correlation(Y, 4),

    lag5 = lag_correlation(Y, 5),

    lag6 = lag_correlation(Y, 6),

    .groups = "drop"
  )

# ------------------------------------------------------------
# 6. Convert to long format
# ------------------------------------------------------------

acf_long <- acf_results %>%
  pivot_longer(
    cols = starts_with("lag"),
    names_to = "lag",
    values_to = "autocorrelation"
  ) %>%
  mutate(
    lag = as.integer(
      str_remove(lag, "lag")
    )
  )

# ------------------------------------------------------------
# 7. Calculate the mean autocorrelation at each lag
# ------------------------------------------------------------

acf_summary <- acf_long %>%
  group_by(lag) %>%
  summarise(

    n_series = n(),

    mean_acf =
      mean(
        autocorrelation,
        na.rm = TRUE
      ),

    sd_acf =
      sd(
        autocorrelation,
        na.rm = TRUE
      ),

    se_acf =
      sd_acf / sqrt(n_series),

    lower_95 =
      mean_acf - 1.96 * se_acf,

    upper_95 =
      mean_acf + 1.96 * se_acf,

    .groups = "drop"
  )

# ------------------------------------------------------------
# 8. Print exact results
# ------------------------------------------------------------

cat("\n====================================\n")
cat("MEAN AUTOCORRELATION RESULTS\n")
cat("====================================\n")

print(
  acf_summary %>%
    mutate(
      mean_acf = round(mean_acf, 6),
      lower_95 = round(lower_95, 6),
      upper_95 = round(upper_95, 6)
    )
)

# ------------------------------------------------------------
# 9. Print values rounded to three decimals
# ------------------------------------------------------------

cat("\n====================================\n")
cat("VALUES USED IN MANUSCRIPT\n")
cat("====================================\n")

acf_summary %>%
  mutate(
    result =
      sprintf(
        "rho(%d) = %.3f",
        lag,
        mean_acf
      )
  ) %>%
  select(
    lag,
    result
  ) %>%
  print()


# ============================================================
# Plot autocorrelation with approximate 95% CI-UNDER Temporal Dynamics and Epidemic Memory
# ============================================================

library(ggplot2)

acf_plot <- ggplot(
  acf_summary,
  aes(
    x = lag,
    y = mean_acf
  )
) +

  # Approximate 95% confidence interval
  geom_ribbon(
    aes(
      ymin = lower_95,
      ymax = upper_95
    ),
    alpha = 0.20
  ) +

  # Mean autocorrelation
  geom_line(
    linewidth = 1
  ) +

  geom_point(
    size = 3
  ) +

  # Reference line
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +

  scale_x_continuous(
    breaks = 1:6
  ) +

  labs(
    title =
      "Autocorrelation of Weekly Influenza Incidence in Nigeria",

    subtitle =
      "Mean across 111 state--season series based on log population-normalised incidence",

    x =
      "Lag (weeks)",

    y =
      "Mean autocorrelation"
  ) +

  theme_minimal(
    base_size = 12
  )

print(acf_plot)

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

ggsave(
  filename =
    "/mnt/data/2-autocorrelation.pdf",

  plot = acf_plot,

  width = 8,
  height = 5.5,

  units = "in"
)

ggsave(
  filename =
    "/mnt/data/2-autocorrelation.png",

  plot = acf_plot,

  width = 8,
  height = 5.5,

  units = "in",

  dpi = 300
)

# ============================================================
# Save autocorrelation results UNDER Temporal Dynamics and Epidemic Memory
# ============================================================

write_csv(
  acf_results,
  "/mnt/data/state_season_autocorrelations.csv"
)

write_csv(
  acf_summary,
  "/mnt/data/temporal_autocorrelation_summary.csv"
)

cat("\nFiles created:\n")
cat("1. state_season_autocorrelations.csv\n")
cat("2. temporal_autocorrelation_summary.csv\n")

# ============================================================
# Autocorrelation by season= Temporal Dynamics and Epidemic Memory
# ============================================================

acf_by_season <- acf_long %>%
  group_by(
    season,
    lag
  ) %>%
  summarise(
    mean_acf =
      mean(
        autocorrelation,
        na.rm = TRUE
      ),

    sd_acf =
      sd(
        autocorrelation,
        na.rm = TRUE
      ),

    n_series = n(),

    se_acf =
      sd_acf / sqrt(n_series),

    lower_95 =
      mean_acf - 1.96 * se_acf,

    upper_95 =
      mean_acf + 1.96 * se_acf,

    .groups = "drop"
  )

cat("\n====================================\n")
cat("AUTOCORRELATION BY SEASON\n")
cat("====================================\n")

print(
  acf_by_season %>%
    mutate(
      mean_acf = round(mean_acf, 4),
      lower_95 = round(lower_95, 4),
      upper_95 = round(upper_95, 4)
    )
)
