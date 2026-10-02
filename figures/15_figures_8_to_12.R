# MAEPiMS Challenge 2026
# Figures 8--12
# Extracted from CODE NEEDED.docx for GitHub submission.
#

      season,
      "2023/2024" = "2023/24",
      "2024/2025" = "2024/25",
      "2025/2026" = "2025/26"
    ),
    season = factor(season, levels = c("2023/24", "2024/25", "2025/26"))
  )

# 3. Convert the three outcomes to long format
plot_data <- national %>%
  select(season, epi_week_of_season,
         cases, hospitalizations, deaths) %>%
  pivot_longer(
    cols = c(cases, hospitalizations, deaths),
    names_to = "outcome",
    values_to = "value"
  ) %>%
  mutate(
    outcome = recode(
      outcome,
      cases = "Cases",
      hospitalizations = "Hospitalisations",
      deaths = "Deaths"
    ),
    outcome = factor(
      outcome,
      levels = c("Cases", "Hospitalisations", "Deaths")
    )
  )

# 4. Publication-style colours
season_cols <- c(
  "2023/24" = "#1B9E77",
  "2024/25" = "#D95F02",
  "2025/26" = "#7570B3"
)

# 5. Draw Figure 8
fig8 <- ggplot(
  plot_data,
  aes(
    x = epi_week_of_season,
    y = value,
    colour = season
  )
) +
  geom_line(linewidth = 0.9) +
  facet_grid(
    outcome ~ .,
    scales = "free_y"
  ) +
  scale_colour_manual(values = season_cols) +
  scale_x_continuous(
    breaks = seq(1, 52, by = 4),
    limits = c(1, 52)
  ) +
  scale_y_continuous(
    labels = label_number(
      accuracy = 1,
      scale_cut = cut_short_scale()
    )
  ) +
  labs(
    title = "National weekly influenza outcomes across the three seasons",
    x = "Epidemiological week of season",
    y = NULL,
    colour = "Season"
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 14,
      hjust = 0.5
    ),
    strip.text = element_text(
      face = "bold",
      size = 11
    ),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    axis.text = element_text(colour = "black"),
    axis.title = element_text(face = "bold"),
    panel.grid.major.y = element_line(
      colour = "grey88",
      linewidth = 0.3
    ),
    panel.grid.major.x = element_blank(),
    strip.background = element_blank(),
    plot.margin = margin(8, 10, 8, 8)
  )

# 6. Display
print(fig8)

# 7. Save high-resolution PNG and vector PDF
ggsave(
  "Figure_8_National_Weekly_Outcomes.png",
  fig8,
  width = 8.2,
  height = 8.6,
  units = "in",
  dpi = 400,
  bg = "white"
)

ggsave(
  "Figure_8_National_Weekly_Outcomes.pdf",
  fig8,
  width = 8.2,
  height = 8.6,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)
# ============================================================
# Figure 9: Northern versus Southern Nigeria epidemic curves
# MAEPiMS Challenge 2026
# ============================================================

library(readr)
library(dplyr)
library(ggplot2)

# 1. Set the path to the challenge state-level data
data_path <- "MAEPiMS_Challenge_Data/nigeria_flu_weekly_by_state.csv"

# 2. Read the state-level weekly surveillance data
dat <- read_csv(data_path, show_col_types = FALSE)

# 3. Define Northern and Southern Nigeria
# Northern: NC, NE, NW
# Southern: SE, SS, SW
regional <- dat %>%
  mutate(
    region = case_when(
      zone %in% c("NC", "NE", "NW") ~ "Northern Nigeria",
      zone %in% c("SE", "SS", "SW") ~ "Southern Nigeria",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(region)) %>%
  group_by(season, region, epi_week_of_season) %>%
  summarise(
    cases = sum(cases, na.rm = TRUE),
    population = sum(population, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    incidence_per_100k = cases / population * 100000,
    season = factor(
      season,
      levels = c("2023/2024", "2024/2025", "2025/2026")
    ),
    region = factor(
      region,
      levels = c("Northern Nigeria", "Southern Nigeria")
    )
  )

# 4. Figure 9
fig9 <- ggplot(
  regional,
  aes(
    x = epi_week_of_season,
    y = incidence_per_100k,
    linetype = region
  )
) +
  geom_line(linewidth = 0.9) +
  facet_wrap(
    ~ season,
    ncol = 1,
    scales = "free_y"
  ) +
  scale_x_continuous(
    breaks = seq(1, 52, by = 4),
    limits = c(1, 52)
  ) +
  labs(
    title = "Northern and Southern Nigeria influenza epidemic curves",
    x = "Epidemiological week of season",
    y = "Incidence per 100,000",
    linetype = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 14,
      hjust = 0.5
    ),
    strip.text = element_text(
      face = "bold",
      size = 11
    ),
    legend.position = "top",
    axis.title = element_text(face = "bold"),
    axis.text = element_text(colour = "black"),
    panel.grid.major.y = element_line(
      colour = "grey88",
      linewidth = 0.3
    ),
    panel.grid.major.x = element_blank(),
    strip.background = element_blank(),
    plot.margin = margin(8, 10, 8, 8)
  )

print(fig9)

# 5. Save publication-quality files
ggsave(
  "Figure_9_North_South_Regional_Epidemic_Curves.png",
  fig9,
  width = 8.2,
  height = 8.2,
  units = "in",
  dpi = 400,
  bg = "white"
)

ggsave(
  "Figure_9_North_South_Regional_Epidemic_Curves.pdf",
  fig9,
  width = 8.2,
  height = 8.2,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

# Figure 10: State-level weekly influenza incidence heatmaps
# MAEPiMS Challenge 2026
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# 1. Path to supplied challenge data
data_path <- "MAEPiMS_Challenge_Data/nigeria_flu_weekly_by_state.csv"

# 2. Read state-level weekly data
dat <- read_csv(data_path, show_col_types = FALSE)

# 3. Use the supplied incidence variable when available.
#    Otherwise calculate population-normalised incidence.
if ("incidence_per_100k" %in% names(dat)) {
  dat <- dat %>%
    mutate(incidence_plot = incidence_per_100k)
} else {
  dat <- dat %>%
    mutate(incidence_plot = cases / population * 100000)
}

# 4. Order seasons
dat <- dat %>%
  mutate(
    season = factor(
      season,
      levels = c("2023/2024", "2024/2025", "2025/2026")
    )
  )

# 5. Put states in alphabetical order and FCT at the bottom
state_levels <- sort(unique(dat$state))
state_levels <- c(
  setdiff(state_levels, "FCT"),
  "FCT"
)

dat <- dat %>%
  mutate(
    state = factor(state, levels = state_levels)
  )

# 6. Use a common scale across all three seasons.
#    A high quantile prevents a small number of extreme cells
#    from compressing the rest of the heatmap.
vmax <- quantile(dat$incidence_plot, 0.995, na.rm = TRUE)

# 7. Figure 10
fig10 <- ggplot(
  dat,
  aes(
    x = epi_week_of_season,
    y = state,
    fill = incidence_plot
  )
) +
  geom_tile(width = 1, height = 0.95) +
  facet_grid(
    season ~ .,
    scales = "free_y",
    switch = "y"
  ) +
  scale_x_continuous(
    breaks = seq(1, 52, by = 4),
    limits = c(1, 52),
    expand = c(0, 0)
  ) +
  scale_fill_viridis_c(
    option = "C",
    name = "Weekly incidence\nper 100,000",
    limits = c(0, vmax),
    oob = scales::squish
  ) +
  labs(
    title = "State-level weekly influenza incidence across Nigeria",
    x = "Epidemiological week of season",
    y = "State / FCT"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 14,
      hjust = 0.5
    ),
    axis.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 8),
    panel.grid = element_blank(),
    strip.text.y = element_text(
      face = "bold",
      size = 11,
      angle = 0
    ),
    strip.background = element_blank(),
    legend.title = element_text(face = "bold"),
    legend.position = "right",
    plot.margin = margin(8, 8, 8, 8)
  )

print(fig10)

# 8. Save publication-quality files
ggsave(
  "Figure_10_State_Weekly_Incidence_Heatmaps.png",
  fig10,
  width = 9.2,
  height = 10,
  units = "in",
  dpi = 400,
  bg = "white"
)

ggsave(
  "Figure_10_State_Weekly_Incidence_Heatmaps.pdf",
  fig10,
  width = 9.2,
  height = 10,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)
# ============================================================
# FIGURE 11
# State-level attack-rate and mortality maps
# MAEPiMS Challenge 2026
# ============================================================

# ------------------------------------------------------------
# 1. Load packages
# ------------------------------------------------------------

library(sf)
library(dplyr)
library(readr)
library(ggplot2)
library(scales)
library(patchwork)
library(viridis)

# ------------------------------------------------------------
# 2. File paths
# ------------------------------------------------------------

data_path <- "MAEPiMS_Challenge_Data/nigeria_flu_season_summary.csv"

boundary_file <- "nigeria_state_boundaries.geojson"

# ------------------------------------------------------------
# 3. Read Nigeria state boundaries
# ------------------------------------------------------------

nga <- st_read(
  boundary_file,
  quiet = TRUE
)

# Check the boundary variables
print(names(nga))

# SimpleMaps boundary file uses "name" for state name
nga <- nga %>%
  rename(state = name)

# ------------------------------------------------------------
# 4. Read MAEPiMS seasonal state-level data
# ------------------------------------------------------------

season_data <- read_csv(
  data_path,
  show_col_types = FALSE
)

# Check variables
print(names(season_data))

# ------------------------------------------------------------
# 5. Standardise state names
# ------------------------------------------------------------

season_data <- season_data %>%
  mutate(
    state = case_when(
      state %in% c("Nassarawa", "Nasarawa") ~ "Nasarawa",

      state %in% c(
        "Federal Capital Territory",
        "FCT",
        "Abuja Federal Capital Territory"
      ) ~ "Federal Capital Territory",

      TRUE ~ state
    )
  )

nga <- nga %>%
  mutate(
    state = case_when(
      state %in% c("Nassarawa", "Nasarawa") ~ "Nasarawa",

      state %in% c(
        "Federal Capital Territory",
        "FCT",
        "Abuja Federal Capital Territory"
      ) ~ "Federal Capital Territory",

      TRUE ~ state
    )
  )

# ------------------------------------------------------------
# 6. Check that all states match
# ------------------------------------------------------------

missing_in_map <- anti_join(
  season_data %>% distinct(state),

  nga %>%
    st_drop_geometry() %>%
    distinct(state),

  by = "state"
)

missing_in_data <- anti_join(
  nga %>%
    st_drop_geometry() %>%
    distinct(state),

  season_data %>%
    distinct(state),

  by = "state"
)

cat("\nStates in MAEPiMS data but not in map:\n")
print(missing_in_map)

cat("\nStates in map but not in MAEPiMS data:\n")
print(missing_in_data)

# ------------------------------------------------------------
# 7. Join epidemiological data to geographic polygons
# ------------------------------------------------------------

map_data <- nga %>%
  left_join(
    season_data,
    by = "state"
  )

# ------------------------------------------------------------
# 8. Order the seasons
# ------------------------------------------------------------

map_data <- map_data %>%
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

# ------------------------------------------------------------
# 9. FIGURE 11A
# Seasonal attack-rate maps
# ------------------------------------------------------------

attack_map <- ggplot(map_data) +

  geom_sf(
    aes(fill = attack_rate_pct),
    colour = "white",
    linewidth = 0.25
  ) +

  facet_wrap(
    ~season,
    nrow = 1
  ) +

  scale_fill_viridis_c(
    option = "C",
    name = "Attack rate (%)",
    labels = label_number(
      accuracy = 0.1
    )
  ) +

  labs(
    title = "Seasonal attack rate across Nigerian states"
  ) +

  theme_void(base_size = 11) +

  theme(
    plot.title = element_text(
      face = "bold",
      size = 13,
      hjust = 0.5
    ),

    strip.text = element_text(
      face = "bold",
      size = 11
    ),

    legend.title = element_text(
      face = "bold"
    ),

    legend.position = "right"
  )

# Display attack-rate map
print(attack_map)

# ------------------------------------------------------------
# 10. FIGURE 11B
# Seasonal cumulative mortality maps
# ------------------------------------------------------------

mortality_map <- ggplot(map_data) +

  geom_sf(
    aes(fill = cum_deaths_per_100k),
    colour = "white",
    linewidth = 0.25
  ) +

  facet_wrap(
    ~season,
    nrow = 1
  ) +

  scale_fill_viridis_c(
    option = "B",
    name = "Deaths per 100,000",
    labels = label_number(
      accuracy = 0.1
    )
  ) +

  labs(
    title = "Seasonal cumulative mortality across Nigerian states"
  ) +

  theme_void(base_size = 11) +

  theme(
    plot.title = element_text(
      face = "bold",
      size = 13,
      hjust = 0.5
    ),

    strip.text = element_text(
      face = "bold",
      size = 11
    ),

    legend.title = element_text(
      face = "bold"
    ),

    legend.position = "right"
  )

# Display mortality map
print(mortality_map)

# ------------------------------------------------------------
# 11. Combine both maps into Figure 11
# ------------------------------------------------------------

fig11 <- attack_map / mortality_map +

  plot_annotation(
    title =
      "State-level seasonal influenza burden and mortality in Nigeria",

    theme = theme(
      plot.title = element_text(
        face = "bold",
        size = 15,
        hjust = 0.5
      )
    )
  )

# Display final Figure 11
print(fig11)

# ------------------------------------------------------------
# 12. Save Figure 11 as PNG
# ------------------------------------------------------------

ggsave(
  filename =
    "Figure_11_State_Attack_Rate_Mortality_Maps.png",

  plot = fig11,

  width = 11,
  height = 8.5,

  units = "in",

  dpi = 400,

  bg = "white"
)

# ------------------------------------------------------------
# 13. Save Figure 11 as PDF
# ------------------------------------------------------------

ggsave(
  filename =
    "Figure_11_State_Attack_Rate_Mortality_Maps.pdf",

  plot = fig11,

  width = 11,
  height = 8.5,

  units = "in",

  device = cairo_pdf,

  bg = "white"
)

# ------------------------------------------------------------
# 14. Confirmation
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FIGURE 11 SUCCESSFULLY CREATED\n")
cat("============================================\n")

cat("\nPDF:\n")
cat("Figure_11_State_Attack_Rate_Mortality_Maps.pdf\n")

cat("\nPNG:\n")
cat("Figure_11_State_Attack_Rate_Mortality_Maps.png\n")


# ============================================================
# FIGURE 12: AMEM PROBABILISTIC FORECAST UNCERTAINTY
# 2024/2025 chronological holdout period
# ============================================================

library(ggplot2)
library(readr)
library(scales)

df <- read_csv("observed_vs_forecast_values.csv", show_col_types = FALSE)

p12 <- ggplot(df, aes(x = week)) +
  geom_ribbon(
    aes(ymin = PI90_lower, ymax = PI90_upper,
        fill = "90% prediction interval"),
    alpha = 0.18
  ) +
  geom_ribbon(
    aes(ymin = PI50_lower, ymax = PI50_upper,
        fill = "50% prediction interval"),
    alpha = 0.30
  ) +
  geom_line(
    aes(y = observed_cases, colour = "Observed cases"),
    linewidth = 0.8
  ) +
  geom_point(
    aes(y = observed_cases, colour = "Observed cases"),
    size = 1.6
  ) +
  geom_line(
    aes(y = predicted_median, colour = "AMEM predictive median"),
    linewidth = 0.8,
    linetype = "dashed"
  ) +
  scale_x_continuous(
    breaks = seq(21, 52, by = 2),
    limits = c(20.5, 52.5)
  ) +
  scale_y_continuous(labels = comma) +
  scale_fill_manual(
    values = c(
      "90% prediction interval" = "grey70",
      "50% prediction interval" = "grey45"
    )
  ) +
  scale_colour_manual(
    values = c(
      "Observed cases" = "black",
      "AMEM predictive median" = "grey20"
    )
  ) +
  labs(
    title = "AMEM Probabilistic Forecasts for the 2024/2025 Holdout Period",
    x = "Epidemiological week",
    y = "Weekly national influenza cases",
    fill = NULL,
    colour = NULL
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 11),
    legend.position = "top",
    legend.text = element_text(size = 10),
    panel.grid.major.y = element_line(
      linewidth = 0.3, linetype = "dashed"
    ),
    panel.grid.minor = element_blank()
  )

print(p12)

ggsave(
  "Figure_12_AMEM_Forecast_Uncertainty.pdf",
  p12,
  width = 12,
  height = 6.8,
  units = "in"
)

ggsave(
  "Figure_12_AMEM_Forecast_Uncertainty.png",
  p12,
  width = 12,
  height = 6.8,
  units = "in",
  dpi = 600
)
