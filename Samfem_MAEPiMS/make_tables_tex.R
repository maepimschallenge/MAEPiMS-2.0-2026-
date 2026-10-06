## Generates LaTeX tables (booktabs) from output/tables/*.csv so every number in the paper comes from the R pipeline.
rd <- function(f) read.csv(file.path("output/tables", f), stringsAsFactors = FALSE)
fm <- function(x, d = 0) formatC(x, format = "f", digits = d, big.mark = ",")
vlab <- c(B1_persistence = "Persistence (empirical growth quantiles)", B2_climatology = "Climatology (week-specific)", V1_lags = "Q-GBC, lags only",
          V2_ma = "Q-GBC + moving-average features", V3_wavelet = "Q-GBC + MRWD (Tier 1)", V4_uniform = "Q-GBC + MRWD + uniform spatial pooling",
          V5_adjacency = "Q-GBC + MRWD + adjacency attention (no $hc$)", V6_WDA = "\\textbf{WDA-STHNet} (MRWD + HW-SA + Q-GBC)")
tex <- function(lines, f) writeLines(lines, file.path("output/tables", f))
## T1 data summary
st <- read.csv("data/nigeria_flu_weekly_by_state.csv"); nat <- read.csv("data/nigeria_flu_weekly_national.csv"); meta <- read.csv("data/nigeria_flu_state_metadata.csv")
pop <- sum(meta$population); a <- aggregate(cbind(cases, hospitalizations, deaths) ~ season, nat, sum)
pk <- sapply(a$season, function(s) { d <- nat[nat$season == s & nat$epi_week_of_season %in% 2:22, ]; d[which.max(d$cases), c("epi_week_of_season", "cases")] })
L <- c("\\begin{tabular}{lrrrrrr}", "\\toprule", "Season & Cases & Attack rate (\\%) & Hospitalisations & Deaths & CFR (\\%) & Peak week (cases) \\\\", "\\midrule")
for (k in 1:3) L <- c(L, sprintf("%s & %s & %.2f & %s & %s & %.3f & %d (%s) \\\\", a$season[k], fm(a$cases[k]), a$cases[k] / pop * 100, fm(a$hospitalizations[k]), fm(a$deaths[k]), a$deaths[k] / a$cases[k] * 100, pk[[1, k]], fm(pk[[2, k]])))
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_data.tex")
## T3 main results
A <- rd("state_by_variant.csv"); B <- rd("state_active_by_variant.csv"); ord <- names(vlab)
L <- c("\\begin{tabular}{lrrrrrr}", "\\toprule", "Model & WIS & MAE & RMSE & Cov.\\ 50\\% & Cov.\\ 90\\% & WIS (epidemic) \\\\", "\\midrule")
for (v in ord) { r <- A[A$variant == v, ]; w <- B$wis[B$variant == v]; L <- c(L, sprintf("%s & %s & %s & %s & %.2f & %.2f & %s \\\\", vlab[v], fm(r$wis), fm(r$ae), fm(r$rmse), r$in50, r$in90, fm(w))); if (v == "B2_climatology") L <- c(L, "\\midrule") }
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_main.tex")
## T4 by horizon (WIS) for selected models
H <- rd("state_by_variant_horizon.csv"); sel <- c("B1_persistence", "V1_lags", "V3_wavelet", "V5_adjacency", "V6_WDA")
L <- c("\\begin{tabular}{lrrrr}", "\\toprule", "Model & $h=1$ & $h=2$ & $h=3$ & $h=4$ \\\\", "\\midrule")
for (v in sel) L <- c(L, paste0(vlab[v], " & ", paste(fm(H$wis[H$variant == v][order(H$h[H$variant == v])]), collapse = " & "), " \\\\"))
L <- c(L, "\\midrule", paste0("90\\% coverage, WDA-STHNet & ", paste(sprintf("%.2f", H$in90[H$variant == "V6_WDA"][order(H$h[H$variant == "V6_WDA"])]), collapse = " & "), " \\\\"),
       paste0("50\\% coverage, WDA-STHNet & ", paste(sprintf("%.2f", H$in50[H$variant == "V6_WDA"][order(H$h[H$variant == "V6_WDA"])]), collapse = " & "), " \\\\"))
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_horizon.tex")
## T5 by season
Sx <- rd("state_by_variant_season.csv"); seas <- sort(unique(Sx$season)); sel <- c("B1_persistence", "B2_climatology", "V1_lags", "V3_wavelet", "V6_WDA")
L <- c("\\begin{tabular}{lrrrrrr}", "\\toprule", " & \\multicolumn{3}{c}{WIS} & \\multicolumn{3}{c}{Coverage of 90\\% PI} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", paste0("Model & ", paste(rep(seas, 2), collapse = " & "), " \\\\"), "\\midrule")
for (v in sel) { r <- Sx[Sx$variant == v, ]; r <- r[order(r$season), ]; L <- c(L, sprintf("%s & %s & %.2f & %.2f & %.2f \\\\", paste0(vlab[v], " & ", paste(fm(r$wis), collapse = " & ")), 0, 0, 0, 0)); L[length(L)] <- paste0(vlab[v], " & ", paste(fm(r$wis), collapse = " & "), " & ", paste(sprintf("%.2f", r$in90), collapse = " & "), " \\\\") }
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_season.tex")
## T6 paired comparisons
P <- rd("paired_comparisons.csv"); L <- c("\\begin{tabular}{lrrrr}", "\\toprule", "Comparison (A vs B) & Mean $\\Delta$WIS (A$-$B) & 95\\% bootstrap CI & States where A better \\\\", "\\midrule")
for (k in 1:nrow(P)) L <- c(L, sprintf("%s & %s & [%s, %s] & %.0f\\%% \\\\", P$cmp[k], fm(P$mean[k], 0), fm(P$lo[k], 0), fm(P$hi[k], 0), 100 * P$share_better[k]))
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_paired.tex")
## T7 national
N <- rd("national_by_variant.csv"); Nl <- c(NB_persistence = "Persistence", N1_lags = "Q-GBC, lags only", N2_wavelet = "Q-GBC + MRWD", N3_WDA = "\\textbf{WDA-STHNet}")
L <- c("\\begin{tabular}{llrrrrr}", "\\toprule", "Target & Model & WIS & MAE & RMSE & Cov.\\ 50\\% & Cov.\\ 90\\% \\\\", "\\midrule")
for (tg in c("cases", "hospitalizations", "deaths")) { first <- TRUE; for (v in names(Nl)) { r <- N[N$target == tg & N$variant == v, ]; L <- c(L, sprintf("%s & %s & %s & %s & %s & %.2f & %.2f \\\\", if (first) tg else "", Nl[v], fm(r$wis, if (tg == "deaths") 1 else 0), fm(r$ae, if (tg == "deaths") 1 else 0), fm(r$rmse, if (tg == "deaths") 1 else 0), r$in50, r$in90)); first <- FALSE }; if (tg != "deaths") L <- c(L, "\\midrule") }
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_national.tex")
## T8 leakage
Lk <- rd("leakage_replication.csv"); L <- c("\\begin{tabular}{lr}", "\\toprule", "Predictor (test: 2025/26, weeks 22--52, $n=31$) & MAE (cases/week) \\\\", "\\midrule")
for (k in 1:4) L <- c(L, sprintf("%s & %s \\\\", Lk$item[k], fm(Lk$MAE[k])))
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_leak.tex")
## T9 feature importance
F <- rd("feature_importance_wda.csv"); F <- F[order(-F$importance), ]; grp <- ifelse(grepl("^sp_", F$feature), "HW-SA (Tier 2)", ifelse(grepl("^w", F$feature), "MRWD (Tier 1)", "own lags / metadata"))
desc <- c(sp_growth = "weekly growth of attention-pooled neighbour pressure", sp_press = "attention-pooled neighbour pressure", sp_gap = "own incidence minus neighbour pressure", d2 = "2-week log change", ex = "excess over early-season baseline", wd2 = "Haar detail, level 2 (4-wk span)", d1 = "1-week log change", wd1 = "Haar detail, level 1 (2-wk span)", d4 = "4-week log change", y = "current log cases", wsm = "Haar smooth A(3)", wd3 = "Haar detail, level 3 (8-wk span)", epi_week = "epi-week", cum = "cumulative incidence", hc = "healthcare access", sahel = "Sahel flag", north = "north flag", savanna = "savanna flag")
L <- c("\\begin{tabular}{llr}", "\\toprule", "Feature & Meaning (group) & Share of importance \\\\", "\\midrule")
for (k in 1:nrow(F)) L <- c(L, sprintf("\\texttt{%s} & %s (%s) & %.1f\\%% \\\\", gsub("_", "\\\\_", F$feature[k]), desc[F$feature[k]], grp[k], 100 * F$importance[k]))
tex(c(L, "\\bottomrule", "\\end{tabular}"), "t_importance.tex")
## numbers used in text
SB <- rd("state_relative_wis.csv"); m <- c(
  sprintf("\\newcommand{\\relPersistMed}{%.2f}", median(SB$rel_persist)), sprintf("\\newcommand{\\relWavMed}{%.2f}", median(SB$rel_wavelet)),
  sprintf("\\newcommand{\\corrHCrel}{%.2f}", cor(SB$hc, SB$rel_persist)), sprintf("\\newcommand{\\corrHCwav}{%.2f}", cor(SB$hc, SB$rel_wavelet)),
  sprintf("\\newcommand{\\gainVsPersist}{%.0f}", 100 * (1 - A$wis[A$variant == "V6_WDA"] / A$wis[A$variant == "B1_persistence"])),
  sprintf("\\newcommand{\\gainVsLags}{%.1f}", 100 * (1 - A$wis[A$variant == "V6_WDA"] / A$wis[A$variant == "V1_lags"])),
  sprintf("\\newcommand{\\gainVsClim}{%.0f}", 100 * (1 - A$wis[A$variant == "V6_WDA"] / A$wis[A$variant == "B2_climatology"])),
  sprintf("\\newcommand{\\gainWavLags}{%.1f}", 100 * (1 - A$wis[A$variant == "V3_wavelet"] / A$wis[A$variant == "V1_lags"])),
  sprintf("\\newcommand{\\gainMALags}{%.1f}", 100 * (1 - A$wis[A$variant == "V2_ma"] / A$wis[A$variant == "V1_lags"])),
  sprintf("\\newcommand{\\gainSpatial}{%.1f}", 100 * (1 - A$wis[A$variant == "V6_WDA"] / A$wis[A$variant == "V3_wavelet"])),
  sprintf("\\newcommand{\\gainUniform}{%.1f}", 100 * (1 - A$wis[A$variant == "V6_WDA"] / A$wis[A$variant == "V4_uniform"])),
  sprintf("\\newcommand{\\lossAdj}{%.1f}", 100 * (A$wis[A$variant == "V6_WDA"] / A$wis[A$variant == "V5_adjacency"] - 1)),
  sprintf("\\newcommand{\\natGain}{%.0f}", 100 * (1 - N$wis[N$target == "cases" & N$variant == "N3_WDA"] / N$wis[N$target == "cases" & N$variant == "NB_persistence"])),
  sprintf("\\newcommand{\\origRatio}{%.1f}", Lk$MAE[2] / Lk$MAE[1]))
writeLines(m, "output/tables/macros2.tex"); cat("latex tables written\n")
