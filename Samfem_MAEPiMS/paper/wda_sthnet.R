#!/usr/bin/env Rscript
## =============================================================================
##  WDA-STHNet : Wavelet-Decomposed Attention-Enhanced Spatio-Temporal Hybrid Network
##  Reproducible R implementation (base R + 'rpart' only; no CRAN downloads)
##  MAEPiMS Challenge 2026 - Nigeria Influenza Forecasting
##
##  Tier 1  MRWD : causal maximal-overlap Haar wavelet transform (own implementation)
##  Tier 2  HW-SA: healthcare-weighted spatial attention  alpha_ij = softmax_j( sim(h_i,h_j)*hc_j + log A_ij )
##  Tier 3  Q-GBC: quantile gradient boosting (pinball loss, tree leaves re-fitted to residual quantiles)
##
##  Evaluation: leave-one-season-out (LOSO), rolling origins, horizons 1-4 weeks,
##  quantiles {0.05,0.25,0.5,0.75,0.95} -> 50% and 90% prediction intervals, WIS, MAE, RMSE, coverage.
##
##  USAGE:  Rscript wda_sthnet.R     (project folder containing ./data/*.csv)
##  OUTPUT: ./output/tables/*.csv, ./output/figures/*.png, ./output/tables/macros.tex
##  Runtime ~ 10-15 min on one core. All seeds fixed.
## =============================================================================
suppressMessages(library(rpart))
set.seed(2026)
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE); dir.create("output/tables", showWarnings = FALSE)
T0 <- Sys.time(); say <- function(...) cat(sprintf("[%5.1f min] ", as.numeric(difftime(Sys.time(), T0, units = "mins"))), ..., "\n", sep = "")

## ----------------------------------------------------------------------------
## 0. DATA
## ----------------------------------------------------------------------------
nat  <- read.csv("data/nigeria_flu_weekly_national.csv", stringsAsFactors = FALSE)
st   <- read.csv("data/nigeria_flu_weekly_by_state.csv", stringsAsFactors = FALSE)
meta <- read.csv("data/nigeria_flu_state_metadata.csv", stringsAsFactors = FALSE)
meta <- meta[order(meta$state), ]; rownames(meta) <- NULL
seasons <- sort(unique(st$season)); states <- meta$state; S <- 3; I <- length(states); NW <- 52
arr <- function(col) { A <- array(NA_real_, c(S, I, NW), dimnames = list(seasons, states, 1:NW))
  A[cbind(match(st$season, seasons), match(st$state, states), st$epi_week_of_season)] <- st[[col]]; A }
CASES <- arr("cases"); HOSP <- arr("hospitalizations"); DEATH <- arr("deaths")
POP <- meta$population; HC <- meta$hc_access; ZONE <- meta$zone; CLIM <- meta$climate; NORTH <- ZONE %in% c("NC", "NE", "NW")
NATC <- apply(CASES, c(1, 3), sum); NATH <- apply(HOSP, c(1, 3), sum); NATD <- apply(DEATH, c(1, 3), sum)
stopifnot(max(abs(NATC - t(sapply(seasons, function(s) nat$cases[nat$season == s])))) == 0)
say("data loaded: ", I, " states x ", S, " seasons x ", NW, " weeks; national == sum of states verified")

## ----------------------------------------------------------------------------
## 1. TIER 1 - MRWD : Haar MODWT (causal and non-causal) and moving-average control
## ----------------------------------------------------------------------------
## Causal MODWT-Haar: W_j(t) = (V_{j-1}(t) - V_{j-1}(t-2^{j-1}))/2 ;  V_j(t) = (V_{j-1}(t) + V_{j-1}(t-2^{j-1}))/2
## Only values up to t are used -> no look-ahead.  y = A^(J) + sum_j D^(j) holds exactly (additivity check below).
haar_causal <- function(x, J = 3) {
  n <- length(x); V <- x; D <- matrix(0, n, J)
  for (j in 1:J) { lag <- 2^(j - 1); Vp <- c(rep(V[1], lag), V)[1:n]; D[, j] <- (V - Vp) / 2; V <- (V + Vp) / 2 }
  list(D = D, A = V)
}
## additivity: original = A^(J) + sum_j D^(j)  requires the telescoping identity V_{j-1} = V_j + W_j
chk <- haar_causal(log1p(NATC[2, ])); stopifnot(max(abs(log1p(NATC[2, ]) - (chk$A + rowSums(chk$D)))) < 1e-9)
## trailing moving-average control features (RQ1)
ma_feats <- function(x) { n <- length(x); tr <- function(k) sapply(1:n, function(t) mean(x[max(1, t - k + 1):t]))
  m2 <- tr(2); m4 <- tr(4); m8 <- tr(8); cbind(ma1 = x - m2, ma2 = m2 - m4, ma3 = m4 - m8, masm = m8) }
## boundary pulse (week 1 = 3.6-4.2x baseline in every season): replaced by week-2 value inside feature series
clean_b <- function(y) { y[1] <- y[2]; y }

## ----------------------------------------------------------------------------
## 2. TIER 2 - HW-SA : healthcare-weighted spatial attention
## ----------------------------------------------------------------------------
## Prior adjacency A (only supplied data are allowed -> geopolitical-zone contiguity proxy):
##   1.0 same zone, 0.5 same macro-region (north / south), 0.1 otherwise
Aadj <- outer(1:I, 1:I, Vectorize(function(i, j) if (i == j) 0 else if (ZONE[i] == ZONE[j]) 1 else if (NORTH[i] == NORTH[j]) 0.5 else 0.1))
cos_sim <- function(E) { En <- E / pmax(sqrt(rowSums(E^2)), 1e-9); En %*% t(En) }
## attention weights at one time point.  E: I x d embeddings.  mode: "hwsa" | "adj" | "uniform"
attention <- function(E, mode = "hwsa") {
  if (mode == "uniform") { W <- matrix(1 / (I - 1), I, I); diag(W) <- 0; return(W) }
  sim <- cos_sim(E); sc <- if (mode == "hwsa") sim * matrix(HC, I, I, byrow = TRUE) else sim
  sc <- sc + log(pmax(Aadj, 1e-12)); diag(sc) <- -Inf
  W <- exp(sc - apply(sc, 1, max)); W / rowSums(W)
}

## ----------------------------------------------------------------------------
## 3. TIER 3 - Q-GBC : quantile gradient boosting with rpart base learners
## ----------------------------------------------------------------------------
## Pinball-loss boosting: fit a regression tree to the negative gradient (tau - 1{r<0}) and replace each leaf value by the
## tau-quantile of the current residuals in that leaf (Friedman 2001, LAD/quantile TreeBoost).
qgbc_fit <- function(X, y, tau, ntree = 100, lr = 0.1, depth = 3, minbucket = 40, subsample = 0.7, seed = 1) {
  set.seed(seed); n <- nrow(X); df <- as.data.frame(X); f0 <- as.numeric(quantile(y, tau)); F <- rep(f0, n); trees <- vector("list", ntree)
  imp <- setNames(numeric(ncol(X)), colnames(X))
  for (m in 1:ntree) {
    idx <- sample.int(n, floor(subsample * n)); r <- y[idx] - F[idx]; g <- ifelse(r > 0, tau, tau - 1)
    d <- cbind(df[idx, , drop = FALSE], .g = g)
    tr <- rpart(.g ~ ., d, method = "anova", control = rpart.control(maxdepth = depth, minbucket = min(minbucket, floor(length(idx) / 4)), cp = 0, xval = 0))
    leaf <- which(tr$frame$var == "<leaf>")
    for (l in leaf) { w <- tr$where == l; tr$frame$yval[l] <- as.numeric(quantile(r[w], tau)) }
    if (!is.null(tr$variable.importance)) imp[names(tr$variable.importance)] <- imp[names(tr$variable.importance)] + tr$variable.importance
    trees[[m]] <- tr; F <- F + lr * predict(tr, df)
  }
  structure(list(f0 = f0, trees = trees, lr = lr, imp = imp), class = "qgbc")
}
predict.qgbc <- function(object, X, ...) { df <- as.data.frame(X); f <- rep(object$f0, nrow(df)); for (tr in object$trees) f <- f + object$lr * predict(tr, df); f }

## ----------------------------------------------------------------------------
## 4. SCORING
## ----------------------------------------------------------------------------
TAUS <- c(0.05, 0.25, 0.5, 0.75, 0.95)
wis5 <- function(y, Q) {          # Q: n x 5 matrix of quantiles (0.05,0.25,0.5,0.75,0.95); WIS with K=2 intervals (alpha 0.1, 0.5)
  is <- function(lo, hi, a) (hi - lo) + 2 / a * pmax(lo - y, 0) + 2 / a * pmax(y - hi, 0)
  (0.5 * abs(y - Q[, 3]) + 0.05 * is(Q[, 1], Q[, 5], 0.1) + 0.25 * is(Q[, 2], Q[, 4], 0.5)) / 2.5
}
pin <- function(y, q, tau) ifelse(y >= q, tau * (y - q), (1 - tau) * (q - y))
fix_cross <- function(Q) t(apply(Q, 1, sort))     # remove quantile crossing

## ----------------------------------------------------------------------------
## 5. FEATURE CONSTRUCTION (all strictly causal: use information up to origin t only)
## ----------------------------------------------------------------------------
HOR <- 1:4; ORIG <- 6:48
build_state_panel <- function(s, wav = TRUE, spatial = "hwsa") {
  ## per-state cleaned log-incidence per 100k and own cases
  Y <- matrix(NA_real_, I, NW); Wf <- vector("list", I); Mf <- vector("list", I)
  for (i in 1:I) { y <- log1p(clean_b(CASES[s, i, ])); Y[i, ] <- y; h <- haar_causal(y, 3); Wf[[i]] <- cbind(h$D, h$A); Mf[[i]] <- ma_feats(y) }
  X100 <- log1p(sapply(1:I, function(i) clean_b(CASES[s, i, ]) / POP[i] * 1e5)) |> t()      # I x NW  (log incidence per 100k)
  rows <- list(); k <- 0
  for (t in ORIG) {
    if (spatial != "none") {
      E <- t(sapply(1:I, function(i) Wf[[i]][t, ]))                      # embeddings h_i,t = causal wavelet vector of state i
      Al <- attention(E, spatial); Pt <- as.numeric(Al %*% X100[, t]); Pt1 <- as.numeric(attention(t(sapply(1:I, function(i) Wf[[i]][t - 1, ])), spatial) %*% X100[, t - 1])
      sp <- cbind(sp_press = Pt, sp_growth = Pt - Pt1, sp_gap = X100[, t] - Pt)
    } else sp <- NULL
    for (i in 1:I) {
      y <- Y[i, ]
      base <- cbind(epi_week = t, y = y[t], d1 = y[t] - y[t - 1], d2 = y[t] - y[t - 2], d4 = y[t] - y[t - 4], ex = y[t] - mean(y[2:6]),
                    cum = log1p(sum(expm1(y[1:t])) / POP[i] * 1e5), hc = HC[i], north = as.numeric(NORTH[i]), sahel = as.numeric(CLIM[i] == "sahel"), savanna = as.numeric(CLIM[i] == "savanna"))
      sig <- if (wav == "wavelet") setNames(Wf[[i]][t, ], c("wd1", "wd2", "wd3", "wsm")) else if (wav == "ma") setNames(Mf[[i]][t, ], c("ma1", "ma2", "ma3", "masm")) else NULL
      k <- k + 1; tg <- sapply(HOR, function(h) y[t + h] - y[t]); names(tg) <- paste0("g", HOR)
      rows[[k]] <- c(season = s, state = i, t = t, unlist(base[1, ]), sig, if (!is.null(sp)) sp[i, ], tg, y0 = y[t],
                     c1 = CASES[s, i, t + 1], c2 = CASES[s, i, t + 2], c3 = CASES[s, i, t + 3], c4 = CASES[s, i, t + 4])
    }
  }
  as.data.frame(do.call(rbind, rows))
}
VARIANTS <- list(
  V1_lags      = list(wav = "none",    spatial = "none",    label = "Lags only (classic Q-GBC)"),
  V2_ma        = list(wav = "ma",      spatial = "none",    label = "+ moving-average features"),
  V3_wavelet   = list(wav = "wavelet", spatial = "none",    label = "+ MRWD wavelet (Tier 1)"),
  V4_uniform   = list(wav = "wavelet", spatial = "uniform", label = "+ MRWD + uniform spatial pooling"),
  V5_adjacency = list(wav = "wavelet", spatial = "adj",     label = "+ MRWD + adjacency attention (no hc)"),
  V6_WDA       = list(wav = "wavelet", spatial = "hwsa",    label = "WDA-STHNet (MRWD + HW-SA + Q-GBC)"))
feat_names <- function(d) setdiff(names(d), c("season", "state", "t", paste0("g", HOR), "y0", paste0("c", HOR)))

say("building feature panels for all variants/seasons ...")
PAN <- lapply(VARIANTS, function(v) lapply(1:S, function(s) build_state_panel(s, v$wav, v$spatial)))
say("feature panels built")

## ----------------------------------------------------------------------------
## 6. LOSO EVALUATION - state-level pooled models, all variants
## ----------------------------------------------------------------------------
NT <- 100
qgbc_all <- function(Xtr, ytr, ntree = NT, ...) lapply(TAUS, function(tau) qgbc_fit(Xtr, ytr, tau, ntree = ntree, ...))
pred_all <- function(models, Xte) fix_cross(sapply(models, function(m) predict(m, Xte)))
EV <- list(); IMP <- NULL; SAVED <- list()
for (vn in names(VARIANTS)) for (s in 1:S) {
  tr <- setdiff(1:S, s); dtr <- do.call(rbind, PAN[[vn]][tr]); dte <- PAN[[vn]][[s]]; fn <- feat_names(dtr)
  for (h in HOR) {
    mods <- qgbc_all(dtr[, fn], dtr[[paste0("g", h)]], seed = h)
    qg <- pred_all(mods, dte[, fn]); Qc <- expm1(dte$y0 + qg); yv <- dte[[paste0("c", h)]]
    EV[[length(EV) + 1]] <- data.frame(variant = vn, season = seasons[s], h = h, state = dte$state, t = dte$t, y = yv, y0 = expm1(dte$y0),
                                       q05 = Qc[, 1], q25 = Qc[, 2], q50 = Qc[, 3], q75 = Qc[, 4], q95 = Qc[, 5])
    if (vn == "V6_WDA" && h == 1 && s == 3) IMP <- Reduce(`+`, lapply(mods, function(m) m$imp))
    if (vn == "V6_WDA") SAVED[[paste(s, h)]] <- list(mods = NULL)
  }
  say(vn, " / hold-out ", seasons[s], " done")
}
EV <- do.call(rbind, EV)

## ---- baselines: persistence with empirical log-residual quantiles; climatology -------------
base_ev <- list()
for (s in 1:S) { tr <- setdiff(1:S, s); dtr <- do.call(rbind, PAN$V1_lags[tr]); dte <- PAN$V1_lags[[s]]
  for (h in HOR) { qs <- quantile(dtr[[paste0("g", h)]], TAUS, names = FALSE)               # pooled empirical log-growth quantiles
    Qc <- expm1(dte$y0 + matrix(qs, nrow(dte), 5, byrow = TRUE)); yv <- dte[[paste0("c", h)]]
    base_ev[[length(base_ev) + 1]] <- data.frame(variant = "B1_persistence", season = seasons[s], h = h, state = dte$state, t = dte$t, y = yv, y0 = expm1(dte$y0), q05 = Qc[, 1], q25 = Qc[, 2], q50 = Qc[, 3], q75 = Qc[, 4], q95 = Qc[, 5])
    ## climatology: week-specific mean/sd of log1p cases (training seasons) -> normal quantiles
    L <- sapply(tr, function(a) sapply(1:I, function(i) log1p(clean_b(CASES[a, i, ]))[dte$t[dte$state == i] + h])) ; 
    mu <- sapply(1:nrow(dte), function(r) mean(log1p(sapply(tr, function(a) CASES[a, dte$state[r], dte$t[r] + h]))))
    sdv <- sapply(1:nrow(dte), function(r) max(sd(log1p(sapply(tr, function(a) CASES[a, dte$state[r], dte$t[r] + h]))), 0.25))
    Qk <- expm1(cbind(mu + qnorm(.05) * sdv, mu + qnorm(.25) * sdv, mu, mu + qnorm(.75) * sdv, mu + qnorm(.95) * sdv))
    base_ev[[length(base_ev) + 1]] <- data.frame(variant = "B2_climatology", season = seasons[s], h = h, state = dte$state, t = dte$t, y = yv, y0 = expm1(dte$y0), q05 = Qk[, 1], q25 = Qk[, 2], q50 = Qk[, 3], q75 = Qk[, 4], q95 = Qk[, 5]) } }
EV <- rbind(EV, do.call(rbind, base_ev))
EV$wis <- wis5(EV$y, as.matrix(EV[, c("q05", "q25", "q50", "q75", "q95")]))
EV$ae <- abs(EV$y - EV$q50); EV$se <- (EV$y - EV$q50)^2
EV$in50 <- as.numeric(EV$y >= EV$q25 & EV$y <= EV$q75); EV$in90 <- as.numeric(EV$y >= EV$q05 & EV$y <= EV$q95)
EV$crps <- (pin(EV$y, EV$q05, .05) + pin(EV$y, EV$q25, .25) + pin(EV$y, EV$q50, .5) + pin(EV$y, EV$q75, .75) + pin(EV$y, EV$q95, .95)) * 2 / 5
## epidemic-phase flag: national weekly cases at target week > 3x that season's early baseline
natbase <- apply(NATC[, 2:6], 1, median); EV$active <- as.numeric(mapply(function(se, tt, h) NATC[match(se, seasons), tt + h] > 3 * natbase[match(se, seasons)], EV$season, EV$t, EV$h))
saveRDS(EV, "output/state_eval.rds"); say("state-level LOSO evaluation complete: ", nrow(EV), " forecast records")

## ----------------------------------------------------------------------------
## 7. NATIONAL MODELS (cases, hospitalisations, deaths) - Q-GBC on national features (+ wavelet, + attention-pooled pressure)
## ----------------------------------------------------------------------------
nat_panel <- function(s, series, wav = "wavelet", spatial = TRUE) {
  y <- log1p(clean_b(series[s, ])); h <- haar_causal(y, 3); W <- cbind(h$D, h$A); M <- ma_feats(y)
  Xinc <- log1p(sapply(1:I, function(i) clean_b(CASES[s, i, ]) / POP[i] * 1e5)) |> t(); Wst <- lapply(1:I, function(i) { yy <- log1p(clean_b(CASES[s, i, ])); hh <- haar_causal(yy, 3); cbind(hh$D, hh$A) })
  rows <- list()
  for (t in ORIG) {
    sp <- c(0, 0); if (spatial) { E <- t(sapply(1:I, function(i) Wst[[i]][t, ])); Al <- attention(E, "hwsa"); P <- as.numeric(Al %*% Xinc[, t]); w <- POP / sum(POP); sp <- c(sum(w * P), sum(w * (X100n <- Xinc[, t])) - sum(w * P)) }
    nsh <- sum(CASES[s, NORTH, t]) / sum(CASES[s, , t])
    base <- c(epi_week = t, y = y[t], d1 = y[t] - y[t - 1], d2 = y[t] - y[t - 2], d4 = y[t] - y[t - 4], ex = y[t] - mean(y[2:6]), nshare = nsh)
    sig <- if (wav == "wavelet") setNames(W[t, ], c("wd1", "wd2", "wd3", "wsm")) else NULL
    tg <- sapply(HOR, function(hh) y[t + hh] - y[t]); names(tg) <- paste0("g", HOR)
    cc <- sapply(HOR, function(hh) series[s, t + hh]); names(cc) <- paste0("c", HOR)
    rows[[length(rows) + 1]] <- c(season = s, t = t, base, sig, if (spatial) c(sp_press = sp[1], sp_gap = sp[2]), tg, y0 = y[t], cc)
  }
  as.data.frame(do.call(rbind, rows))
}
NEV <- list()
for (tgt in c("cases", "hospitalizations", "deaths")) {
  ser <- switch(tgt, cases = NATC, hospitalizations = NATH, deaths = NATD)
  for (cfg in c("N1_lags", "N2_wavelet", "N3_WDA")) {
    P <- lapply(1:S, function(s) nat_panel(s, ser, wav = if (cfg == "N1_lags") "none" else "wavelet", spatial = cfg == "N3_WDA"))
    for (s in 1:S) { tr <- setdiff(1:S, s); dtr <- do.call(rbind, P[tr]); dte <- P[[s]]; fn <- setdiff(names(dtr), c("season", "t", paste0("g", HOR), "y0", paste0("c", HOR)))
      for (h in HOR) { mods <- lapply(TAUS, function(tau) qgbc_fit(dtr[, fn], dtr[[paste0("g", h)]], tau, ntree = 60, depth = 2, minbucket = 8, seed = h))
        Qc <- expm1(dte$y0 + pred_all(mods, dte[, fn])); NEV[[length(NEV) + 1]] <- data.frame(target = tgt, variant = cfg, season = seasons[s], h = h, t = dte$t, y = dte[[paste0("c", h)]], y0 = expm1(dte$y0), q05 = Qc[, 1], q25 = Qc[, 2], q50 = Qc[, 3], q75 = Qc[, 4], q95 = Qc[, 5]) } }
  }
  ## persistence baseline
  P0 <- lapply(1:S, function(s) nat_panel(s, ser, "none", FALSE))
  for (s in 1:S) { tr <- setdiff(1:S, s); dtr <- do.call(rbind, P0[tr]); dte <- P0[[s]]
    for (h in HOR) { qs <- quantile(dtr[[paste0("g", h)]], TAUS, names = FALSE); Qc <- expm1(dte$y0 + matrix(qs, nrow(dte), 5, byrow = TRUE))
      NEV[[length(NEV) + 1]] <- data.frame(target = tgt, variant = "NB_persistence", season = seasons[s], h = h, t = dte$t, y = dte[[paste0("c", h)]], y0 = expm1(dte$y0), q05 = Qc[, 1], q25 = Qc[, 2], q50 = Qc[, 3], q75 = Qc[, 4], q95 = Qc[, 5]) } }
  say("national ", tgt, " done")
}
NEV <- do.call(rbind, NEV)
NEV$wis <- wis5(NEV$y, as.matrix(NEV[, c("q05", "q25", "q50", "q75", "q95")])); NEV$ae <- abs(NEV$y - NEV$q50); NEV$se <- (NEV$y - NEV$q50)^2
NEV$in50 <- as.numeric(NEV$y >= NEV$q25 & NEV$y <= NEV$q75); NEV$in90 <- as.numeric(NEV$y >= NEV$q05 & NEV$y <= NEV$q95)
NEV$crps <- (pin(NEV$y, NEV$q05, .05) + pin(NEV$y, NEV$q25, .25) + pin(NEV$y, NEV$q50, .5) + pin(NEV$y, NEV$q75, .75) + pin(NEV$y, NEV$q95, .95)) * 2 / 5
NEV$active <- as.numeric(mapply(function(se, tt, h) NATC[match(se, seasons), tt + h] > 3 * natbase[match(se, seasons)], NEV$season, NEV$t, NEV$h))
saveRDS(NEV, "output/national_eval.rds")
## bottom-up national point forecast from state-level WDA-STHNet medians
bu <- aggregate(cbind(q50, y) ~ season + h + t, EV[EV$variant == "V6_WDA", ], sum); bu$ae <- abs(bu$y - bu$q50)

## ----------------------------------------------------------------------------
## 8. REPLICATION OF THE ORIGINAL SCRIPT'S PROTOCOL (national, 80/20 chronological split, same-week features)
##    shows the effect of target leakage (roll_mean_3 and wavelet 'approx' contain the current week) and of an easy test slice
## ----------------------------------------------------------------------------
orig <- data.frame(season = rep(1:S, each = NW), week = rep(1:NW, S), cases = as.numeric(t(NATC)))
hs <- haar_causal(orig$cases, 2); orig$wavelet_trend <- (orig$cases + c(orig$cases[1], orig$cases[-nrow(orig)])) / 2    # s1 of Haar MODWT = (x_t + x_{t-1})/2
orig$wavelet_noise <- (orig$cases - c(orig$cases[1], orig$cases[-nrow(orig)])) / 2
orig$lag_1 <- c(NA, orig$cases[-nrow(orig)]); orig$lag_2 <- c(NA, NA, orig$cases[1:(nrow(orig) - 2)]); orig$lag_4 <- c(rep(NA, 4), orig$cases[1:(nrow(orig) - 4)])
orig$roll_mean_3 <- sapply(1:nrow(orig), function(i) if (i < 3) NA else mean(orig$cases[(i - 2):i]))        # INCLUDES current week (zoo::rollmean align='right')
orig <- orig[complete.cases(orig), ]; ntr <- floor(0.8 * nrow(orig)); tri <- 1:ntr
fo <- c("week", "wavelet_trend", "wavelet_noise", "lag_1", "lag_2", "lag_4", "roll_mean_3")
m50 <- qgbc_fit(orig[tri, fo], orig$cases[tri], 0.5, ntree = 150, lr = 0.05, depth = 4, minbucket = 5)
p50 <- predict(m50, orig[-tri, fo]); te <- orig[-tri, ]
orig_mae <- mean(abs(te$cases - p50)); orig_rmse <- sqrt(mean((te$cases - p50)^2))
## the same pipeline without the same-week features (honest nowcast-free setting: lags only, no current week)
fo2 <- c("week", "lag_1", "lag_2", "lag_4"); m50b <- qgbc_fit(orig[tri, fo2], orig$cases[tri], 0.5, ntree = 150, lr = 0.05, depth = 4, minbucket = 5)
honest_mae <- mean(abs(te$cases - predict(m50b, orig[-tri, fo2]))); persist_mae <- mean(abs(te$cases - te$lag_1)); mean_mae <- mean(abs(te$cases - mean(orig$cases[tri])))
LEAK <- data.frame(item = c("Original protocol (same-week features)", "Same split, same-week features removed", "Persistence (lag-1)", "Training-mean predictor"),
                   MAE = c(orig_mae, honest_mae, persist_mae, mean_mae), n_test = nrow(te), test_weeks = paste0("season ", seasons[orig$season[-tri][1]], " wk ", orig$week[-tri][1], "-", tail(orig$week[-tri], 1)))
write.csv(LEAK, "output/tables/leakage_replication.csv", row.names = FALSE)
say("original-protocol replication: MAE ", round(orig_mae), " vs honest ", round(honest_mae))

## ----------------------------------------------------------------------------
## 9. SUMMARY TABLES
## ----------------------------------------------------------------------------
smry <- function(d, by) { a <- aggregate(cbind(wis, ae, se, crps, in50, in90) ~ ., d[, c(by, "wis", "ae", "se", "crps", "in50", "in90")], mean); a$rmse <- sqrt(a$se); a$se <- NULL; a }
S1 <- smry(EV, c("variant", "h")); write.csv(S1, "output/tables/state_by_variant_horizon.csv", row.names = FALSE)
S1a <- smry(EV[EV$active == 1, ], c("variant", "h")); write.csv(S1a, "output/tables/state_active_by_variant_horizon.csv", row.names = FALSE)
S2 <- smry(EV, c("variant", "season")); write.csv(S2, "output/tables/state_by_variant_season.csv", row.names = FALSE)
S3 <- smry(EV, "variant"); write.csv(S3, "output/tables/state_by_variant.csv", row.names = FALSE)
S3a <- smry(EV[EV$active == 1, ], "variant"); write.csv(S3a, "output/tables/state_active_by_variant.csv", row.names = FALSE)
N1 <- smry(NEV, c("target", "variant", "h")); write.csv(N1, "output/tables/national_by_variant_horizon.csv", row.names = FALSE)
N2 <- smry(NEV, c("target", "variant")); write.csv(N2, "output/tables/national_by_variant.csv", row.names = FALSE)
N2a <- smry(NEV[NEV$active == 1, ], c("target", "variant")); write.csv(N2a, "output/tables/national_active_by_variant.csv", row.names = FALSE)
N3 <- smry(NEV[NEV$target == "cases", ], c("variant", "season")); write.csv(N3, "output/tables/national_cases_by_season.csv", row.names = FALSE)
## relative WIS vs persistence by state & correlation with hc access
rw <- aggregate(wis ~ state + variant, EV, mean); w6 <- rw[rw$variant == "V6_WDA", ]; wb <- rw[rw$variant == "B1_persistence", ]; w3 <- rw[rw$variant == "V3_wavelet", ]
SB <- data.frame(state = states[w6$state], zone = ZONE[w6$state], hc = HC[w6$state], wis_wda = w6$wis, wis_persist = wb$wis, wis_wavelet = w3$wis); SB$rel_persist <- SB$wis_wda / SB$wis_persist; SB$rel_wavelet <- SB$wis_wda / SB$wis_wavelet
write.csv(SB, "output/tables/state_relative_wis.csv", row.names = FALSE)
## paired comparison (state-level mean WIS differences, 37 states) with bootstrap CI
pb <- function(a, b) { d <- (a - b); r <- replicate(4000, mean(sample(d, replace = TRUE))); c(mean = mean(d), lo = unname(quantile(r, .025)), hi = unname(quantile(r, .975)), share_better = mean(d < 0)) }
set.seed(7)
PAIR <- rbind(data.frame(cmp = "Wavelet vs lags-only", t(pb(w3$wis, rw$wis[rw$variant == "V1_lags"]))),
              data.frame(cmp = "Wavelet vs moving-average", t(pb(w3$wis, rw$wis[rw$variant == "V2_ma"]))),
              data.frame(cmp = "HW-SA vs no spatial (wavelet)", t(pb(w6$wis, w3$wis))),
              data.frame(cmp = "HW-SA vs uniform pooling", t(pb(w6$wis, rw$wis[rw$variant == "V4_uniform"]))),
              data.frame(cmp = "HW-SA vs adjacency attention", t(pb(w6$wis, rw$wis[rw$variant == "V5_adjacency"]))),
              data.frame(cmp = "WDA-STHNet vs persistence", t(pb(w6$wis, wb$wis))))
write.csv(PAIR, "output/tables/paired_comparisons.csv", row.names = FALSE)
write.csv(data.frame(feature = names(IMP), importance = as.numeric(IMP) / sum(IMP)), "output/tables/feature_importance_wda.csv", row.names = FALSE)
saveRDS(list(SB = SB, PAIR = PAIR, bu = bu, LEAK = LEAK), "output/summaries.rds")
say("tables written")

## ----------------------------------------------------------------------------
## 10. FIGURES
## ----------------------------------------------------------------------------
cols <- c("#1b6ca8", "#d1495b", "#2a9d8f", "#e9a23b", "#6a4c93", "#444444", "#8c564b", "#17becf")
png2 <- function(n, w = 1900, h = 1200) png(file.path("output/figures", n), w, h, res = 200, type = "cairo")
## F1 national series
png2("fig1_national.png", 2000, 1400); par(mfrow = c(3, 1), mar = c(3, 5, 2, 1))
for (k in c("cases", "hospitalizations", "deaths")) { A <- switch(k, cases = NATC, hospitalizations = NATH, deaths = NATD)
  plot(NA, xlim = c(1, 52), ylim = c(0, max(A) / 1e3), xlab = "", ylab = paste(k, "(thousands)"), main = paste("National weekly", k)); for (s in 1:S) lines(1:NW, A[s, ] / 1e3, col = cols[s], lwd = 2); if (k == "cases") legend("topright", seasons, col = cols[1:3], lwd = 2, bty = "n") }
dev.off()
## F2 causal Haar decomposition
y <- log1p(clean_b(NATC[2, ])); h <- haar_causal(y, 3); png2("fig2_mrwd.png", 2000, 1600); par(mfrow = c(5, 1), mar = c(2.5, 5, 1.5, 1))
plot(y, type = "l", lwd = 2, col = cols[1], ylab = "log1p cases", main = "Causal MODWT-Haar decomposition, national 2024/25 (week-1 pulse cleaned)"); for (j in 1:3) plot(h$D[, j], type = "h", lwd = 2, col = cols[1 + j], ylab = paste0("D", j, " (", 2^j, "-wk)")); abline(h = 0)
plot(h$A, type = "l", lwd = 2, col = cols[6], ylab = "A(3)", xlab = "epi-week"); dev.off()
## F3 attention heatmaps
sx <- 2; t0 <- 14; Wm <- lapply(1:I, function(i) { yy <- log1p(clean_b(CASES[sx, i, ])); hh <- haar_causal(yy, 3); cbind(hh$D, hh$A) }); E <- t(sapply(1:I, function(i) Wm[[i]][t0, ]))
ordz <- order(ZONE, HC); png2("fig3_attention.png", 2200, 1000); par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
for (md in c("hwsa", "adj", "uniform")) { Al <- attention(E, md)[ordz, ordz]; image(1:I, 1:I, t(Al)[, I:1], col = hcl.colors(50, "YlGnBu", rev = TRUE), axes = FALSE, xlab = "source state j (ordered by zone, hc)", ylab = "target state i", main = switch(md, hwsa = "HW-SA", adj = "Adjacency attention", uniform = "Uniform")); box() }
dev.off()
## F4 ablation: WIS relative to the lags-only Q-GBC (persistence is ~2-4x worse and is reported in the tables)
vn <- c("V2_ma", "V3_wavelet", "V4_uniform", "V5_adjacency", "V6_WDA"); lab <- c("+ moving averages", "+ MRWD wavelets", "+ MRWD + uniform pooling", "+ MRWD + adjacency attention", "WDA-STHNet (MRWD + HW-SA)")
vcol <- c("#8c564b", "#2a9d8f", "#e9a23b", "#6a4c93", "#d1495b")
png2("fig4_ablation.png", 2200, 1100); par(mfrow = c(1, 2), mar = c(4, 4.5, 3, 1))
for (ph in 1:2) { D <- if (ph == 1) S1 else S1a; ref <- D$wis[D$variant == "V1_lags"][order(D$h[D$variant == "V1_lags"])]
  M <- sapply(vn, function(v) D$wis[D$variant == v][order(D$h[D$variant == v])] / ref)
  matplot(1:4, M, type = "b", pch = 16, lty = 1, lwd = 2, col = vcol, ylim = c(0.65, 1.05), xlab = "forecast horizon (weeks)", ylab = "WIS / WIS(lags-only Q-GBC)", main = if (ph == 1) "All origins" else "Epidemic-phase origins", xaxt = "n"); axis(1, 1:4); abline(h = 1, lty = 2)
  if (ph == 1) legend("bottomleft", lab, col = vcol, pch = 16, lwd = 2, bty = "n", cex = 0.7) }
dev.off()
## F5 calibration
png2("fig5_calibration.png", 2200, 1000); par(mfrow = c(1, 2), mar = c(4, 4.5, 3, 1))
D <- aggregate(cbind(in50, in90) ~ variant + season, EV[EV$variant %in% c("V1_lags", "V3_wavelet", "V6_WDA"), ], mean)
plot(NA, xlim = c(0.3, 1), ylim = c(0, 1), xlab = "nominal", ylab = "empirical coverage (state level)", main = "Coverage of 50% and 90% PIs (by held-out season)"); abline(0, 1, lty = 2)
for (k in 1:nrow(D)) { cc <- c(V1_lags = cols[3], V3_wavelet = cols[1], V6_WDA = cols[2])[D$variant[k]]; pc <- c(16, 17, 15)[match(D$season[k], seasons)]; points(c(.5, .9), c(D$in50[k], D$in90[k]), col = cc, pch = pc, cex = 1.3) }
legend("topleft", c("Lags only", "+MRWD", "WDA-STHNet", seasons), col = c(cols[3], cols[1], cols[2], rep("black", 3)), pch = c(16, 16, 16, 16, 17, 15), bty = "n", cex = 0.75)
Dh <- aggregate(cbind(in50, in90) ~ h, EV[EV$variant == "V6_WDA", ], mean); plot(Dh$h, Dh$in90, type = "b", pch = 16, lwd = 2, col = cols[2], ylim = c(0, 1), xlab = "horizon (weeks)", ylab = "coverage", main = "WDA-STHNet coverage by horizon"); lines(Dh$h, Dh$in50, type = "b", pch = 17, lwd = 2, col = cols[1]); abline(h = c(.5, .9), lty = 3); legend("bottomleft", c("90% PI", "50% PI"), col = cols[2:1], pch = 16:17, lwd = 2, bty = "n"); dev.off()
## F6 national fan charts (cases) for each held-out season, h=1 and h=4
png2("fig6_national_fan.png", 2200, 1500); par(mfrow = c(3, 2), mar = c(3.5, 4.5, 2.2, 1))
for (s in 1:S) for (h in c(1, 4)) { d <- NEV[NEV$target == "cases" & NEV$variant == "N3_WDA" & NEV$season == seasons[s] & NEV$h == h, ]; d <- d[order(d$t), ]
  plot(NA, xlim = c(1, 52), ylim = c(0, max(d$q95, d$y) / 1e6), xlab = "target epi-week", ylab = "cases (millions)", main = paste0(seasons[s], ", h = ", h, " wk")); polygon(c(d$t + h, rev(d$t + h)), c(d$q05, rev(d$q95)) / 1e6, col = adjustcolor(cols[1], .2), border = NA)
  polygon(c(d$t + h, rev(d$t + h)), c(d$q25, rev(d$q75)) / 1e6, col = adjustcolor(cols[1], .4), border = NA); lines(d$t + h, d$q50 / 1e6, col = cols[1], lwd = 2); lines(1:NW, NATC[s, ] / 1e6, col = "black", lwd = 1.5, lty = 2) }
dev.off()
## F7 state examples
ex <- c("Lagos", "Kano", "Borno", "Rivers"); png2("fig7_state_examples.png", 2200, 1400); par(mfrow = c(2, 2), mar = c(3.5, 4.5, 2.2, 1))
for (nm in ex) { i <- match(nm, states); d <- EV[EV$variant == "V6_WDA" & EV$season == "2024/2025" & EV$h == 2 & EV$state == i, ]; d <- d[order(d$t), ]
  plot(NA, xlim = c(1, 52), ylim = c(0, max(d$q95, d$y) / 1e3), xlab = "target epi-week", ylab = "cases (thousands)", main = paste(nm, "(hc =", HC[i], ") 2024/25, h = 2")); polygon(c(d$t + 2, rev(d$t + 2)), c(d$q05, rev(d$q95)) / 1e3, col = adjustcolor(cols[2], .2), border = NA); polygon(c(d$t + 2, rev(d$t + 2)), c(d$q25, rev(d$q75)) / 1e3, col = adjustcolor(cols[2], .4), border = NA)
  lines(d$t + 2, d$q50 / 1e3, col = cols[2], lwd = 2); lines(1:NW, CASES[2, i, ] / 1e3, lty = 2) }
dev.off()
## F8 relative WIS by state vs hc
png2("fig8_state_gain.png", 2200, 1000); par(mfrow = c(1, 2), mar = c(4, 4.5, 3, 1)); zc <- setNames(cols[c(1, 2, 3, 4, 5, 6)], sort(unique(ZONE)))
plot(SB$hc, SB$rel_persist, pch = 16, col = zc[SB$zone], xlab = "healthcare-access index", ylab = "WIS(WDA-STHNet) / WIS(persistence)", main = "Skill by state"); abline(h = 1, lty = 2); legend("topright", names(zc), col = zc, pch = 16, bty = "n", cex = 0.7, ncol = 2)
plot(SB$hc, SB$rel_wavelet, pch = 16, col = zc[SB$zone], xlab = "healthcare-access index", ylab = "WIS(WDA-STHNet) / WIS(+MRWD, no spatial)", main = "Value added by HW-SA"); abline(h = 1, lty = 2); abline(lm(rel_wavelet ~ hc, SB), col = "grey30"); dev.off()
## F9 leakage
png2("fig9_leakage.png", 1700, 1100); par(mar = c(4, 12, 3, 1)); barplot(rev(LEAK$MAE) / 1e3, names.arg = rev(c("Original protocol\n(same-week feat.)", "Same split,\nno same-week feat.", "Persistence", "Training mean")), horiz = TRUE, las = 1, col = rev(c(cols[2], cols[1], cols[3], "grey60")), xlab = "MAE on test slice (thousand cases/week)", main = "Effect of same-week features in the original script"); dev.off()
## F10 feature importance
fi <- read.csv("output/tables/feature_importance_wda.csv"); fi <- fi[order(fi$importance), ]; png2("fig10_importance.png", 1700, 1300); par(mar = c(4, 7, 3, 1)); barplot(fi$importance, names.arg = fi$feature, horiz = TRUE, las = 1, col = ifelse(grepl("^sp_", fi$feature), cols[2], ifelse(grepl("^w", fi$feature), cols[1], "grey65")), xlab = "share of total split importance", main = "WDA-STHNet (h=1, quantile ensemble) feature importance"); legend("bottomright", c("spatial (HW-SA)", "wavelet (MRWD)", "other"), fill = c(cols[2], cols[1], "grey65"), bty = "n"); dev.off()
## F11 state heatmap of observed incidence (2024/25)
png2("fig11_heatmap.png", 2000, 1500); par(mar = c(4, 6.5, 3, 5)); ordz <- order(NORTH, ZONE, HC); A <- log10(1 + CASES[2, ordz, ] / POP[ordz] * 1e5)
image(1:NW, 1:I, t(A), col = hcl.colors(60, "YlOrRd", rev = TRUE), axes = FALSE, xlab = "epi-week of season", ylab = "", main = "Weekly incidence per 100,000 (log10), 2024/25"); axis(1); axis(2, at = 1:I, labels = states[ordz], las = 1, cex.axis = 0.55); dev.off()
say("figures written")

## ----------------------------------------------------------------------------
## 11. MACROS FOR LATEX (numbers in the paper are generated here)
## ----------------------------------------------------------------------------
f0 <- function(x) formatC(x, format = "f", digits = 0, big.mark = ","); f2 <- function(x) formatC(x, format = "f", digits = 2); f3 <- function(x) formatC(x, format = "f", digits = 3)
g <- function(D, v, col, h = NULL) { r <- D[D$variant == v & (is.null(h) | D$h == h), ]; if (!is.null(h)) r[[col]] else mean(r[[col]]) }
mac <- c(
  sprintf("\\newcommand{\\wisWDA}{%s}", f0(S3$wis[S3$variant == "V6_WDA"])), sprintf("\\newcommand{\\wisPersist}{%s}", f0(S3$wis[S3$variant == "B1_persistence"])),
  sprintf("\\newcommand{\\wisLags}{%s}", f0(S3$wis[S3$variant == "V1_lags"])), sprintf("\\newcommand{\\wisWav}{%s}", f0(S3$wis[S3$variant == "V3_wavelet"])), sprintf("\\newcommand{\\wisMA}{%s}", f0(S3$wis[S3$variant == "V2_ma"])),
  sprintf("\\newcommand{\\cvNinety}{%s}", f2(S3$in90[S3$variant == "V6_WDA"])), sprintf("\\newcommand{\\cvFifty}{%s}", f2(S3$in50[S3$variant == "V6_WDA"])),
  sprintf("\\newcommand{\\origMAE}{%s}", f0(orig_mae)), sprintf("\\newcommand{\\honestMAE}{%s}", f0(honest_mae)), sprintf("\\newcommand{\\persistMAE}{%s}", f0(persist_mae)),
  sprintf("\\newcommand{\\nrecords}{%s}", f0(nrow(EV))), sprintf("\\newcommand{\\nstatemodels}{%s}", f0(length(VARIANTS) * S * length(HOR) * length(TAUS))))
writeLines(mac, "output/tables/macros.tex")
say("DONE. Elapsed ", round(difftime(Sys.time(), T0, units = "mins"), 1), " min")
