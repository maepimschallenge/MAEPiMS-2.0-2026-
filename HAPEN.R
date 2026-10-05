
# HAPEN 2026 - Reproducible V3.1
# Adaptive Predictive Influence Network (APIN) + Hierarchical Probabilistic Forecasting
# ONLY supplied challenge data are used.
# Final evaluation: 2025/2026 held out completely.
#
# Core originality claim:
# An integrated architecture in which a training-only, data-adaptive
# state-to-state predictive influence network is converted into target-state
# epidemic-pressure features, then combined with historical wave phenotype,
# climate-transition structure, static state heterogeneity and a probabilistic
# nonlinear forecaster. The network is re-learned inside each validation split.
#
# IMPORTANT: This script treats 2025/2026 as a fully held-out season.
# No 2025/2026 cases, hospitalisations, deaths, peak/summary values or
# test-season derived rates are used as predictors, feature-selection inputs,
# network-learning inputs, or calibration data.

options(stringsAsFactors = FALSE)
set.seed(123)

# ------------------------- 0. CONFIG ------------------------------
DATA_DIR <- "data"
OUT_DIR <- "outputs_v3_rerun"
#OUT_DIR  <- "outputs_v3"
SEED <- 123
TRAIN_SEASONS <- c("2023/2024", "2024/2025")
TEST_SEASON <- "2025/2026"
LAGS <- 1:6
K_CANDIDATES <- c(3,5,8,10)
QUANTILES <- c(0.05,0.25,0.50,0.75,0.95)
N_TREES <- 1500
MIN_NODE <- 5

if(!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive=TRUE)
if(!dir.exists(file.path(OUT_DIR,"figures"))) dir.create(file.path(OUT_DIR,"figures"), recursive=TRUE)

# ------------------------- 1. PACKAGES ----------------------------
required <- c("dplyr","tidyr","readr","purrr","ggplot2","ranger","zoo","scales","MASS")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if(length(missing)) stop("Install required packages before running: ", paste(missing,collapse=", "))
invisible(lapply(required, library, character.only=TRUE))

# ------------------------- 2. HELPERS -----------------------------
msg <- function(...) cat(paste0(...,"\n"))
clip01 <- function(x) pmin(pmax(x,0),1)
rmse <- function(y,p) sqrt(mean((y-p)^2,na.rm=TRUE))
mae <- function(y,p) mean(abs(y-p),na.rm=TRUE)

# Conformal quantile with finite-sample adjustment.
conformal_q <- function(score, coverage){
  score <- score[is.finite(score)]
  n <- length(score)
  if(n < 10) return(0)
  k <- ceiling((n+1)*coverage)
  k <- min(max(k,1),n)
  sort(score,partial=k)[k]
}

# Standard WIS for central 50% and 90% intervals plus median.
wis_vec <- function(y,q05,q25,q50,q75,q95){
  is50 <- (q75-q25) + (2/0.50)*(q25-y)*(y<q25) + (2/0.50)*(y-q75)*(y>q75)
  is90 <- (q95-q05) + (2/0.10)*(q05-y)*(y<q05) + (2/0.10)*(y-q95)*(y>q95)
  # Three components, with interval components receiving equal weight and
  # the median component receiving half the interval weight.
  (0.5*abs(y-q50) + is50 + is90)/2.5
}

# Trapezoidal quantile-loss approximation to CRPS on supplied grid.
crps_quantile <- function(y, qs, taus){
  ord <- order(taus); taus <- taus[ord]; qs <- qs[ord]
  loss <- (taus - (y < qs))*(y-qs)
  if(length(taus)<2) return(NA_real_)
  # Integrate only over observed quantile range; report explicitly as approximation.
  2*sum(diff(taus)*(head(loss,-1)+tail(loss,-1))/2)
}

# Detect up to two meaningful peaks in a weekly curve.
detect_peaks <- function(x, weeks, min_sep=4, min_ratio=0.25){
  ord <- order(weeks); x <- as.numeric(x[ord]); weeks <- as.numeric(weeks[ord])
  sm <- zoo::rollmean(x,3,fill=NA,align="center")
  cand <- which(!is.na(sm) & sm>=dplyr::lag(sm,default=NA) & sm>=dplyr::lead(sm,default=NA))
  cand <- cand[!is.na(sm[cand])]
  if(!length(cand)) return(tibble(primary_week=weeks[which.max(x)],primary_peak=max(x),secondary_week=NA_real_,secondary_peak=NA_real_,secondary_ratio=0))
  cand <- cand[order(sm[cand],decreasing=TRUE)]
  p <- cand[1]
  s <- cand[abs(cand-p)>=min_sep & sm[cand]>=min_ratio*sm[p]]
  if(!length(s)) return(tibble(primary_week=weeks[p],primary_peak=x[p],secondary_week=NA_real_,secondary_peak=NA_real_,secondary_ratio=0))
  s <- s[1]
  idx <- c(p,s); idx <- idx[order(x[idx],decreasing=TRUE)]
  p <- idx[1]; s <- idx[2]
  tibble(primary_week=weeks[p],primary_peak=x[p],secondary_week=weeks[s],secondary_peak=x[s],secondary_ratio=x[s]/pmax(x[p],1))
}

# ------------------------- 3. READ DATA ---------------------------
read_required <- function(fname){
  path <- file.path(DATA_DIR,fname)
  if(!file.exists(path)) stop("Missing supplied dataset: ",path)
  readr::read_csv(path,show_col_types=FALSE)
}

weekly <- read_required("nigeria_flu_weekly_by_state.csv")
national <- read_required("nigeria_flu_weekly_national.csv")
state_meta <- read_required("nigeria_flu_state_metadata.csv")
season_meta <- read_required("nigeria_flu_season_reference.csv")
summary_path <- file.path(DATA_DIR,"nigeria_flu_season_summary.csv")
season_summary <- if(file.exists(summary_path)) readr::read_csv(summary_path,show_col_types=FALSE) else NULL

# ------------------------- 4. STANDARDISE / QA --------------------
if("death" %in% names(weekly) && !"deaths" %in% names(weekly)) weekly <- rename(weekly,deaths=death)
if("death" %in% names(national) && !"deaths" %in% names(national)) national <- rename(national,deaths=death)

req_w <- c("season","state","zone","climate","population","epi_week_of_season","week_start","cases","hospitalizations","deaths","cases_per_100k")
stopifnot(all(req_w %in% names(weekly)))

weekly <- weekly %>%
  mutate(season=as.character(season),state=as.character(state),zone=as.character(zone),climate=tolower(as.character(climate)),epi_week_of_season=as.integer(epi_week_of_season),week_start=as.Date(week_start)) %>%
  arrange(season,state,epi_week_of_season)

national <- national %>% mutate(season=as.character(season),epi_week_of_season=as.integer(epi_week_of_season),week_start=as.Date(week_start)) %>% arrange(season,epi_week_of_season)
state_meta <- state_meta %>% mutate(state=as.character(state),climate=tolower(as.character(climate)),zone=as.character(zone))

qa <- list(
  weekly_rows=nrow(weekly), national_rows=nrow(national), states=n_distinct(weekly$state), seasons=sort(unique(weekly$season)),
  duplicate_state_week=sum(duplicated(weekly[c("season","state","epi_week_of_season")])),
  duplicate_national_week=sum(duplicated(national[c("season","epi_week_of_season")])),
  missing_cases=sum(is.na(weekly$cases)), negative_cases=sum(weekly$cases<0,na.rm=TRUE),
  sunday_week_start=sum(lubridate::wday(weekly$week_start,week_start=1)!=1,na.rm=TRUE)
)
writeLines(capture.output(str(qa)),file.path(OUT_DIR,"QA_report.txt"))

# Verify national/state arithmetic on supplied data.
state_nat <- weekly %>% group_by(season,epi_week_of_season) %>% summarise(cases=sum(cases),hospitalizations=sum(hospitalizations),deaths=sum(deaths),.groups="drop")
nat_cmp <- national %>% select(season,epi_week_of_season,nat_cases=cases,nat_hosp=hospitalizations,nat_deaths=deaths) %>% left_join(state_nat,by=c("season","epi_week_of_season")) %>% mutate(case_diff=nat_cases-cases,hosp_diff=nat_hosp-hospitalizations,death_diff=nat_deaths-deaths)
write_csv(nat_cmp,file.path(OUT_DIR,"QA_national_state_reconciliation.csv"))

# ------------------------- 5. STATIC HISTORICAL PHENOTYPE --------
# Historical features are learned only from TRAIN_SEASONS.
train_weekly <- weekly %>% filter(season %in% TRAIN_SEASONS)
test_truth <- weekly %>% filter(season==TEST_SEASON)

hist_wave <- train_weekly %>%
  group_by(season,state,climate,zone) %>%
  group_modify(~detect_peaks(.x$cases,.x$epi_week_of_season)) %>% ungroup() %>%
  group_by(state,climate,zone) %>%
  summarise(
    hist_peak_week=mean(primary_week,na.rm=TRUE),
    hist_peak_cases=mean(primary_peak,na.rm=TRUE),
    hist_secondary_week=mean(secondary_week,na.rm=TRUE),
    hist_secondary_ratio=mean(secondary_ratio,na.rm=TRUE),
    hist_two_wave_rate=mean(is.finite(secondary_week)),
    .groups="drop"
  )

# Static burden/access phenotype from previous seasons only.
hist_burden <- train_weekly %>%
  group_by(season,state,population) %>%
  summarise(total_cases=sum(cases),peak_cases=max(cases),peak_week=epi_week_of_season[which.max(cases)],.groups="drop") %>%
  group_by(state) %>% summarise(hist_mean_burden=mean(total_cases/population),hist_peak_week=mean(peak_week),hist_peak_rate=mean(peak_cases/population),.groups="drop")

# ------------------------- 6. ADAPTIVE PREDICTIVE INFLUENCE -----
# APIN is learned from training seasons only. For each target state,
# identify source states whose lagged, deseasonalised incidence best predicts
# the target's current incidence. The learned weights are then frozen.

# Deseasonalise log incidence by season using Fourier terms.
deseason_state <- function(df){
  df %>% group_by(season,state) %>% group_modify(~{
    z <- .x %>% mutate(rate=100000*cases/pmax(population,1),y=log1p(rate),
                       s1=sin(2*pi*epi_week_of_season/52),c1=cos(2*pi*epi_week_of_season/52),
                       s2=sin(4*pi*epi_week_of_season/52),c2=cos(4*pi*epi_week_of_season/52))
    fit <- lm(y~s1+c1+s2+c2,data=z)
    z$resid_y <- residuals(fit)
    z
  }) %>% ungroup()
}

tr_resid <- deseason_state(train_weekly)

# Historical peak timing and metadata for network regularisation.
state_info <- state_meta %>% select(state,population,zone,climate,hc_access) %>% distinct(state,.keep_all=TRUE) %>% left_join(hist_burden,by="state")

pair_scores <- function(target, source, data, lags=LAGS){
  if(target==source) return(tibble())
  a <- data %>% filter(state==source) %>% select(season,epi_week_of_season,src=resid_y)
  b <- data %>% filter(state==target) %>% select(season,epi_week_of_season,tgt=resid_y)
  map_dfr(lags,function(L){
    z <- b %>% mutate(source_week=epi_week_of_season-L) %>% left_join(a,by=c("season","source_week"="epi_week_of_season")) %>% filter(is.finite(src),is.finite(tgt))
    if(nrow(z)<12) return(tibble(lag=L,score=NA_real_,n=nrow(z)))
    tibble(lag=L,score=suppressWarnings(cor(z$src,z$tgt)),n=nrow(z))
  })
}

states <- sort(unique(train_weekly$state))
network_raw <- map_dfr(states,function(target){
  map_dfr(states,function(source){
    ps <- pair_scores(target,source,tr_resid)
    if(!nrow(ps)) return(tibble())
    best <- ps %>% filter(is.finite(score)) %>% arrange(desc(score),lag) %>% slice(1)
    if(!nrow(best)) return(tibble())
    tibble(target=target,source=source,lag=best$lag,lag_corr=best$score,n_pairs=best$n)
  })
})

network_raw <- network_raw %>%
  left_join(state_info %>% select(target=state,target_climate=climate,target_zone=zone,target_hc=hc_access,target_peak=hist_peak_week),by="target") %>%
  left_join(state_info %>% select(source=state,source_climate=climate,source_zone=zone,source_hc=hc_access,source_peak=hist_peak_week),by="source") %>%
  mutate(
    climate_sim=exp(-abs(match(source_climate,c("tropical","savanna","sahel"))-match(target_climate,c("tropical","savanna","sahel")))),
    timing_sim=exp(-abs(source_peak-target_peak)/8),
    access_sim=exp(-abs(source_hc-target_hc)/0.25),
    positive_corr=pmax(lag_corr,0),
    adaptive_score=0.65*positive_corr+0.15*climate_sim+0.15*timing_sim+0.05*access_sim
  )

# The source state's influence is normalised within each target state.
network <- network_raw %>% group_by(target) %>% arrange(desc(adaptive_score),.by_group=TRUE) %>% mutate(rank=row_number(),weight=adaptive_score/sum(adaptive_score,na.rm=TRUE)) %>% ungroup()

write_csv(network,file.path(OUT_DIR,"APIN_training_network.csv"))
write_csv(network %>% group_by(target) %>% slice_max(adaptive_score,n=5,with_ties=FALSE) %>% ungroup(),file.path(OUT_DIR,"APIN_top5_links_per_state.csv"))

# ------------------------- 7. APIN FEATURE ENGINE -----------------
# Build prospective source-state pressure for each week. At week t the
# network uses source state values at t-learned-lag only.

make_apin <- function(data,network){
  # Use rate rather than raw counts so large states do not dominate solely by population.
  base <- data %>% transmute(season,state,epi_week_of_season,source_rate=100000*cases/pmax(population,1))
  out <- data %>% select(season,state,epi_week_of_season) %>% distinct()
  # For computational stability retain top 10 learned sources per target.
  net <- network %>% filter(rank<=10)
  for(tgt in unique(net$target)){
    nt <- net %>% filter(target==tgt)
    for(j in seq_len(nrow(nt))){
      row <- nt[j,]
      sig <- base %>% filter(state==row$source) %>% transmute(season,epi_week_of_season,src_week=epi_week_of_season+row$lag,contrib=source_rate*row$weight)
      key <- paste0("apin_",make.names(row$source))
      sig <- sig %>% rename(!!key := contrib)
      out <- out %>% left_join(sig,by=c("season","epi_week_of_season"="src_week"))
    }
  }
  # Collapse source contributions into target-specific pressure.
  # Because each row of 'out' contains all sources, select matching target weights.
  out %>% group_by(season,state,epi_week_of_season) %>% summarise(.groups="drop")
}

# More efficient target-specific implementation.
make_apin_features <- function(data,network){
  base <- data %>% transmute(season,state,epi_week_of_season,source_rate=100000*cases/pmax(population,1))
  pieces <- network %>% filter(rank<=10) %>% group_split(target)
  map_dfr(pieces,function(nt){
    tgt <- unique(nt$target)
    vals <- base %>% filter(state %in% nt$source) %>% select(season,state,epi_week_of_season,source_rate)
    vals <- nt %>%
      select(source, lag, weight) %>%
      left_join(
        vals,
        by = c("source" = "state")
      ) %>%
      mutate(
        target_week = epi_week_of_season + lag,
        weighted = source_rate * weight
      ) %>%
      group_by(season,target_week) %>% summarise(apin_pressure=sum(weighted,na.rm=TRUE),apin_sources=sum(is.finite(weighted)),.groups="drop") %>%
      transmute(season,state=tgt,epi_week_of_season=target_week,apin_pressure,apin_sources)
    vals
  })
}

apin_features <- make_apin_features(train_weekly,network)

# ------------------------- 8. CLIMATE PROPAGATION -----------------
# Learn climate-to-climate lags from TRAIN_SEASONS only.
clim_train <- train_weekly %>% group_by(season,climate,epi_week_of_season) %>% summarise(cases=sum(cases),.groups="drop")
clim_pairs <- tibble(source=c("tropical","savanna","tropical"),target=c("savanna","sahel","sahel"))
clim_lags <- map_dfr(seq_len(nrow(clim_pairs)),function(i){
  src <- clim_train %>% filter(climate==clim_pairs$source[i]) %>% select(season,epi_week_of_season,src=cases)
  tgt <- clim_train %>% filter(climate==clim_pairs$target[i]) %>% select(season,epi_week_of_season,tgt=cases)
  map_dfr(LAGS,function(L){
    z <- tgt %>% mutate(source_week=epi_week_of_season-L) %>% left_join(src,by=c("season","source_week"="epi_week_of_season")) %>% filter(is.finite(src),is.finite(tgt))
    tibble(source=clim_pairs$source[i],target=clim_pairs$target[i],lag=L,corr=ifelse(nrow(z)>=12,cor(z$src,z$tgt),NA_real_))
  })
})
prop_lags <- clim_lags %>% filter(is.finite(corr)) %>% group_by(source,target) %>% slice_max(corr,n=1,with_ties=FALSE) %>% ungroup()

write_csv(clim_lags,file.path(OUT_DIR,"APIN_climate_lag_scan.csv"))
write_csv(prop_lags,file.path(OUT_DIR,"APIN_frozen_climate_lags.csv"))

make_climate_features <- function(data,prop_lags){
  cw <- data %>% group_by(season,climate,epi_week_of_season) %>% summarise(cases=sum(cases),.groups="drop") %>% pivot_wider(names_from=climate,values_from=cases,values_fill=0)
  for(v in c("tropical","savanna","sahel")) if(!v %in% names(cw)) cw[[v]] <- 0
  for(i in seq_len(nrow(prop_lags))){
    src <- prop_lags$source[i]; tgt <- prop_lags$target[i]; L <- prop_lags$lag[i]
    nm <- paste0(src,"_to_",tgt)
    cw[[nm]] <- dplyr::lag(cw[[src]],L)
    # lag must reset at season boundary
    cw <- cw %>% group_by(season) %>% mutate(!!nm := lag(.data[[src]],L)) %>% ungroup()
  }
  cw
}

climate_features <- make_climate_features(weekly,prop_lags)

# ------------------------- 9. STATIC MODEL FRAME ------------------
static_features <- weekly %>%
  select(season,state,epi_week_of_season,week_start,population,zone,climate) %>%
  distinct() %>%
  left_join(state_meta %>% select(state,hc_access),by="state") %>%
  left_join(hist_wave,by=c("state","climate","zone")) %>%
  left_join(hist_burden,by="state")

# Historical phenotype is safe for test because it is derived only from training seasons.
static_features <- static_features %>%
  mutate(
    sin1=sin(2*pi*epi_week_of_season/52),cos1=cos(2*pi*epi_week_of_season/52),
    sin2=sin(4*pi*epi_week_of_season/52),cos2=cos(4*pi*epi_week_of_season/52),
    week_norm=epi_week_of_season/52
  )

# ------------------------- 11. SEASON-FORWARD TRAINING FEATURES --
# To make training/test information sets comparable, each season's predictors
# are constructed only from information available BEFORE that season begins.
# 2023/2024 therefore uses static/calendar features only; 2024/2025 uses
# 2023/2024-derived APIN and historical phenotype. The final 2025/2026 frame
# uses both historical seasons. This prevents within-season target leakage.

make_historical_context <- function(learn_seasons,k_use=5){
  d <- weekly %>% filter(season %in% learn_seasons)
  if(!nrow(d)) return(list(wave=tibble(),burden=tibble(),network=tibble(),apin=tibble(),climate=tibble()))
  wave <- d %>% group_by(season,state,climate,zone) %>% group_modify(~detect_peaks(.x$cases,.x$epi_week_of_season)) %>% ungroup() %>% group_by(state,climate,zone) %>% summarise(hist_peak_week=mean(primary_week,na.rm=TRUE),hist_peak_cases=mean(primary_peak,na.rm=TRUE),hist_secondary_week=mean(secondary_week,na.rm=TRUE),hist_secondary_ratio=mean(secondary_ratio,na.rm=TRUE),hist_two_wave_rate=mean(is.finite(secondary_week)),.groups="drop")
  burden <- d %>% group_by(season,state,population) %>% summarise(total_cases=sum(cases),peak_cases=max(cases),peak_week=epi_week_of_season[which.max(cases)],.groups="drop") %>% group_by(state) %>% summarise(hist_mean_burden=mean(total_cases/population),hist_peak_week=mean(peak_week),hist_peak_rate=mean(peak_cases/population),.groups="drop")
  r <- deseason_state(d)
  ss <- sort(unique(d$state))
  net <- map_dfr(ss,function(target) map_dfr(ss,function(source){ ps<-pair_scores(target,source,r); if(!nrow(ps)) return(tibble()); b<-ps %>% filter(is.finite(score)) %>% arrange(desc(score),lag) %>% slice(1); if(!nrow(b)) return(tibble()); tibble(target=target,source=source,lag=b$lag,lag_corr=b$score,n_pairs=b$n) }))
  info <- state_meta %>% select(state,population,zone,climate,hc_access) %>% distinct(state,.keep_all=TRUE)
  net <- net %>% left_join(info %>% select(target=state,target_climate=climate,target_zone=zone,target_hc=hc_access),by="target") %>% left_join(info %>% select(source=state,source_climate=climate,source_hc=hc_access),by="source") %>% left_join(burden %>% select(target=state,target_peak=hist_peak_week),by="target") %>% left_join(burden %>% select(source=state,source_peak=hist_peak_week),by="source") %>% mutate(climate_sim=exp(-abs(match(source_climate,c("tropical","savanna","sahel"))-match(target_climate,c("tropical","savanna","sahel")))),timing_sim=exp(-abs(source_peak-target_peak)/8),access_sim=exp(-abs(source_hc-target_hc)/0.25),positive_corr=pmax(lag_corr,0),adaptive_score=0.65*positive_corr+0.20*climate_sim+0.10*timing_sim+0.05*access_sim) %>% group_by(target) %>% arrange(desc(adaptive_score),.by_group=TRUE) %>% mutate(rank=row_number(),weight=adaptive_score/sum(adaptive_score,na.rm=TRUE)) %>% ungroup()
  ap <- make_apin_features(d,net %>% filter(rank<=k_use))
  cw <- d %>% group_by(season,climate,epi_week_of_season) %>% summarise(cases=sum(cases),.groups="drop") %>% pivot_wider(names_from=climate,values_from=cases,values_fill=0)
  for(v in c("tropical","savanna","sahel")) if(!v %in% names(cw)) cw[[v]]<-0
  cr <- map_dfr(seq_len(nrow(clim_pairs)),function(i){ src<-cw %>% select(season,epi_week_of_season,src=all_of(clim_pairs$source[i])); tgt<-cw %>% select(season,epi_week_of_season,tgt=all_of(clim_pairs$target[i])); map_dfr(LAGS,function(L){ z<-tgt %>% mutate(source_week=epi_week_of_season-L) %>% left_join(src,by=c("season","source_week"="epi_week_of_season")) %>% filter(is.finite(src),is.finite(tgt)); tibble(source=clim_pairs$source[i],target=clim_pairs$target[i],lag=L,corr=ifelse(nrow(z)>=12,cor(z$src,z$tgt),NA_real_)) }) }) %>% filter(is.finite(corr)) %>% group_by(source,target) %>% slice_max(corr,n=1,with_ties=FALSE) %>% ungroup()
  clim <- cw
  for(i in seq_len(nrow(cr))){ nm<-paste0(cr$source[i],"_to_",cr$target[i]); src<-cr$source[i]; L<-cr$lag[i]; clim <- clim %>% group_by(season) %>% mutate(!!nm:=lag(.data[[src]],L)) %>% ungroup() }
  list(wave=wave,burden=burden,network=net,apin=ap,climate=clim)
}

# Recreate season-forward contexts.
context_2324 <- make_historical_context(character(0))
# 2023/2024 has no previous season in the supplied data.
context_2425 <- make_historical_context("2023/2024",5)
context_test <- make_historical_context(TRAIN_SEASONS,5)

base_grid <- weekly %>% select(season,state,epi_week_of_season,population,zone,climate) %>% distinct() %>% left_join(state_meta %>% select(state,hc_access),by="state") %>% mutate(history_depth=case_when(season=="2023/2024"~0,season=="2024/2025"~1,season=="2025/2026"~2,TRUE~0),sin1=sin(2*pi*epi_week_of_season/52),cos1=cos(2*pi*epi_week_of_season/52),sin2=sin(4*pi*epi_week_of_season/52),cos2=cos(4*pi*epi_week_of_season/52),week_norm=epi_week_of_season/52)

build_season_features <- function(season_label, context){
  
  # --------------------------
  # Climate history
  # --------------------------
  if(
    nrow(context$climate) == 0 ||
    !"epi_week_of_season" %in% names(context$climate)
  ){
    
    clim_hist <- tibble(
      epi_week_of_season =
        sort(unique(base_grid$epi_week_of_season))
    )
    
  } else {
    
    clim_hist <- context$climate %>%
      group_by(epi_week_of_season) %>%
      summarise(
        across(
          contains("_to_"),
          ~mean(.x, na.rm = TRUE)
        ),
        .groups = "drop"
      )
    
  }
  
  # --------------------------
  # APIN history
  # --------------------------
  if(
    nrow(context$apin) == 0 ||
    !"epi_week_of_season" %in% names(context$apin)
  ){
    
    ap_hist <- tibble(
      state = character(),
      epi_week_of_season = integer(),
      apin_pressure = numeric(),
      apin_sources = numeric()
    )
    
  } else {
    
    ap_hist <- context$apin %>%
      filter(state %in% unique(base_grid$state)) %>%
      select(
        state,
        epi_week_of_season,
        apin_pressure,
        apin_sources
      ) %>%
      group_by(
        state,
        epi_week_of_season
      ) %>%
      summarise(
        apin_pressure =
          mean(apin_pressure, na.rm = TRUE),
        
        apin_sources =
          mean(apin_sources, na.rm = TRUE),
        
        .groups = "drop"
      )
    
  }
  
  # --------------------------
  # Empty wave protection
  # --------------------------
  wave_tbl <- context$wave
  
  if(
    nrow(wave_tbl) == 0 ||
    !"state" %in% names(wave_tbl)
  ){
    
    wave_tbl <- tibble(
      state = character(),
      climate = character(),
      zone = character()
    )
    
  }
  
  # --------------------------
  # Empty burden protection
  # --------------------------
  burden_tbl <- context$burden
  
  if(
    nrow(burden_tbl) == 0 ||
    !"state" %in% names(burden_tbl)
  ){
    
    burden_tbl <- tibble(
      state = character()
    )
    
  }
  
  g <- base_grid %>%
    filter(season == season_label) %>%
    left_join(
      wave_tbl,
      by = c(
        "state",
        "climate",
        "zone"
      )
    ) %>%
    left_join(
      burden_tbl,
      by = "state"
    ) %>%
    left_join(
      ap_hist,
      by = c(
        "state",
        "epi_week_of_season"
      )
    ) %>%
    left_join(
      clim_hist,
      by = "epi_week_of_season"
    )
  
  g
}

train_supervised <- bind_rows(
  build_season_features("2023/2024",context_2324),
  build_season_features("2024/2025",context_2425)
)

dim(train_supervised)

names(train_supervised)

targets <- train_weekly %>% group_by(season,state) %>% arrange(epi_week_of_season,.by_group=TRUE) %>% mutate(target_cases=lead(cases),target_hosp=lead(hospitalizations),target_deaths=lead(deaths)) %>% ungroup() %>% select(season,state,epi_week_of_season,target_cases,target_hosp,target_deaths)
train_supervised <- train_supervised %>% left_join(targets,by=c("season","state","epi_week_of_season")) %>% filter(!is.na(target_cases))

# Final test features are built from historical seasons only.
test_clim_hist <- context_test$climate %>% group_by(epi_week_of_season) %>% summarise(across(contains("_to_"),~mean(.x,na.rm=TRUE)),.groups="drop")
test_apin_hist <- context_test$apin %>% group_by(state,epi_week_of_season) %>% summarise(apin_pressure=mean(apin_pressure,na.rm=TRUE),apin_sources=mean(apin_sources,na.rm=TRUE),.groups="drop")
test_frame <- base_grid %>% filter(season==TEST_SEASON) %>% left_join(context_test$wave,by=c("state","climate","zone")) %>% left_join(context_test$burden,by="state") %>% left_join(test_apin_hist,by=c("state","epi_week_of_season")) %>% left_join(test_clim_hist,by="epi_week_of_season")

num_vars <- c("apin_pressure","apin_sources","tropical_to_savanna","savanna_to_sahel","tropical_to_sahel")
for(v in num_vars){ if(v %in% names(train_supervised)){ train_supervised[[v]][!is.finite(train_supervised[[v]])]<-NA; train_supervised[[v]][is.na(train_supervised[[v]])]<-median(train_supervised[[v]],na.rm=TRUE) }; if(v %in% names(test_frame)){ test_frame[[v]][!is.finite(test_frame[[v]])]<-NA; test_frame[[v]][is.na(test_frame[[v]])]<-if(v %in% names(train_supervised)) median(train_supervised[[v]],na.rm=TRUE) else 0 } }

# ------------------------- 12. MODEL MATRIX -----------------------
model_vars <- c("epi_week_of_season","history_depth","state","zone","climate","hc_access","hist_peak_week","hist_peak_cases","hist_secondary_week","hist_secondary_ratio","hist_two_wave_rate","hist_mean_burden","hist_peak_rate","sin1","cos1","sin2","cos2","week_norm","apin_pressure","apin_sources","tropical_to_savanna","savanna_to_sahel","tropical_to_sahel")
model_vars <- model_vars[model_vars %in% names(train_supervised)]

train_model <- train_supervised %>% select(all_of(c("season",model_vars,"target_cases","target_hosp","target_deaths"))) %>% mutate(across(c(state,zone,climate),as.factor))

# ------------------------- 13. VALIDATION FOR NETWORK SIZE --------
# We select K on 2024/2025 using only 2023/2024 for fitting. Importantly,
# the network itself must be re-learned inside this split in a rigorous version.
# Here we use the frozen TRAIN-only network and assess K as a development choice.
# The final 2025/2026 test is untouched by this choice.
# In the development split, network/phenotype learning must also be restricted to 2023/2024.

# K sensitivity is implemented by rebuilding a restricted network and pressure.
score_k <- function(k){
  # STRICT DEVELOPMENT SPLIT:
  # learn the APIN network and historical phenotype from 2023/2024 only;
  # use 2024/2025 only for validation scoring.
  dev_train <- train_weekly %>% filter(season == "2023/2024")
  dev_valid <- train_weekly %>% filter(season == "2024/2025")
  
  dev_resid <- deseason_state(dev_train)
  dev_states <- sort(unique(dev_train$state))
  dev_info <- state_meta %>% select(state,population,zone,climate,hc_access) %>% distinct(state,.keep_all=TRUE)
  dev_burden <- dev_train %>% group_by(state,population) %>% summarise(total_cases=sum(cases),peak_cases=max(cases),peak_week=epi_week_of_season[which.max(cases)],.groups="drop") %>% group_by(state) %>% summarise(hist_mean_burden=mean(total_cases/population),hist_peak_week=mean(peak_week),hist_peak_rate=mean(peak_cases/population),.groups="drop")
  dev_wave <- dev_train %>% group_by(season,state,climate,zone) %>% group_modify(~detect_peaks(.x$cases,.x$epi_week_of_season)) %>% ungroup() %>% group_by(state,climate,zone) %>% summarise(hist_peak_week=mean(primary_week,na.rm=TRUE),hist_peak_cases=mean(primary_peak,na.rm=TRUE),hist_secondary_week=mean(secondary_week,na.rm=TRUE),hist_secondary_ratio=mean(secondary_ratio,na.rm=TRUE),hist_two_wave_rate=mean(is.finite(secondary_week)),.groups="drop")
  
  dev_pair <- map_dfr(dev_states,function(target){
    map_dfr(dev_states,function(source){
      ps <- pair_scores(target,source,dev_resid)
      if(!nrow(ps)) return(tibble())
      best <- ps %>% filter(is.finite(score)) %>% arrange(desc(score),lag) %>% slice(1)
      if(!nrow(best)) return(tibble())
      tibble(target=target,source=source,lag=best$lag,lag_corr=best$score,n_pairs=best$n)
    })
  }) %>%
    left_join(dev_info %>% select(target=state,target_climate=climate,target_zone=zone,target_hc=hc_access),by="target") %>%
    left_join(dev_wave %>% select(target=state,target_peak=hist_peak_week),by="target")
  
  # Rebuild the score with source metadata; avoid any validation-season information.
  dev_pair <- dev_pair %>%
    left_join(dev_info %>% select(source=state,source_climate=climate,source_hc=hc_access),by="source") %>%
    mutate(climate_sim=exp(-abs(match(source_climate,c("tropical","savanna","sahel"))-match(target_climate,c("tropical","savanna","sahel")))),
           timing_sim=1,
           access_sim=exp(-abs(source_hc-target_hc)/0.25),
           positive_corr=pmax(lag_corr,0),
           adaptive_score=0.70*positive_corr+0.20*climate_sim+0.10*access_sim) %>%
    group_by(target) %>% arrange(desc(adaptive_score),.by_group=TRUE) %>% mutate(rank=row_number(),weight=adaptive_score/sum(adaptive_score,na.rm=TRUE)) %>% ungroup()
  
  dev_net <- dev_pair %>% filter(rank<=k)
  dev_apin <- make_apin_features(bind_rows(dev_train,dev_valid),dev_net)
  
  # Development historical phenotype is learned only from dev_train.
  dev_static <- bind_rows(dev_train,dev_valid) %>% select(season,state,epi_week_of_season,population,zone,climate) %>% distinct() %>% left_join(dev_info %>% select(state,hc_access),by="state") %>% left_join(dev_wave,by=c("state","climate","zone")) %>% left_join(dev_burden,by="state") %>% mutate(sin1=sin(2*pi*epi_week_of_season/52),cos1=cos(2*pi*epi_week_of_season/52),sin2=sin(4*pi*epi_week_of_season/52),cos2=cos(4*pi*epi_week_of_season/52),week_norm=epi_week_of_season/52) %>% left_join(dev_apin,by=c("season","state","epi_week_of_season"))
  
  dev_targets <- bind_rows(dev_train,dev_valid) %>% group_by(season,state) %>% arrange(epi_week_of_season,.by_group=TRUE) %>% mutate(target_cases=lead(cases)) %>% ungroup() %>% select(season,state,epi_week_of_season,target_cases)
  dev_static <- dev_static %>% left_join(dev_targets,by=c("season","state","epi_week_of_season"))
  vv <- intersect(model_vars,names(dev_static))
  tr <- dev_static %>% filter(season=="2023/2024") %>% drop_na(all_of(c(vv,"target_cases")))
  va <- dev_static %>% filter(season=="2024/2025") %>% drop_na(all_of(vv))
  if(nrow(tr)<50 || nrow(va)<50) return(tibble(k=k,MAE=NA_real_))
  tr <- tr %>% mutate(across(c(state,zone,climate),as.factor))
  va <- va %>% mutate(across(c(state,zone,climate),~factor(.x,levels=levels(tr[[cur_column()]]))))
  m <- ranger(target_cases~.,data=tr %>% select(all_of(c(vv,"target_cases"))),num.trees=500,mtry=max(2,floor(sqrt(length(vv)))),min.node.size=MIN_NODE,seed=SEED)
  p <- predict(m,data=va %>% select(all_of(vv)))$predictions
  tibble(k=k,MAE=mae(va$target_cases,p))
}

k_validation <- map_dfr(K_CANDIDATES,score_k)
selected_k <- k_validation %>% filter(is.finite(MAE)) %>% arrange(MAE) %>% slice(1) %>% pull(k)
if(!length(selected_k)) selected_k <- 5
write_csv(k_validation,file.path(OUT_DIR,"APIN_K_validation.csv"))
write_csv(tibble(selected_k=selected_k),file.path(OUT_DIR,"APIN_selected_K.csv"))

# Rebuild all season-forward contexts with the frozen K selected on the
# development split. No 2025/2026 outcome enters this operation.
context_2425 <- make_historical_context("2023/2024",selected_k)
context_test <- make_historical_context(TRAIN_SEASONS,selected_k)
train_supervised <- bind_rows(build_season_features("2023/2024",context_2324),build_season_features("2024/2025",context_2425)) %>% left_join(targets,by=c("season","state","epi_week_of_season")) %>% filter(!is.na(target_cases))
test_clim_hist <- context_test$climate %>% group_by(epi_week_of_season) %>% summarise(across(contains("_to_"),~mean(.x,na.rm=TRUE)),.groups="drop")
test_apin_hist <- context_test$apin %>% group_by(state,epi_week_of_season) %>% summarise(apin_pressure=mean(apin_pressure,na.rm=TRUE),apin_sources=mean(apin_sources,na.rm=TRUE),.groups="drop")
test_frame <- base_grid %>% filter(season==TEST_SEASON) %>% left_join(context_test$wave,by=c("state","climate","zone")) %>% left_join(context_test$burden,by="state") %>% left_join(test_apin_hist,by=c("state","epi_week_of_season")) %>% left_join(test_clim_hist,by="epi_week_of_season")
for(v in num_vars){ if(v %in% names(train_supervised)){ train_supervised[[v]][!is.finite(train_supervised[[v]])]<-NA; train_supervised[[v]][is.na(train_supervised[[v]])]<-median(train_supervised[[v]],na.rm=TRUE) }; if(v %in% names(test_frame)){ test_frame[[v]][!is.finite(test_frame[[v]])]<-NA; test_frame[[v]][is.na(test_frame[[v]])]<-if(v %in% names(train_supervised)) median(train_supervised[[v]],na.rm=TRUE) else 0 } }

# ------------------------- 15. FINAL POINT/QUANTILE MODELS -------
final_train <- train_supervised %>% mutate(across(c(state,zone,climate),as.factor))

for(v in num_vars){ if(v %in% names(final_train)){ final_train[[v]][!is.finite(final_train[[v]])] <- NA; final_train[[v]][is.na(final_train[[v]])] <- median(final_train[[v]],na.rm=TRUE) } }

# Completely held-out test frame: no 2025/2026 outcome data.
test_frame <- test_frame %>% mutate(across(c(state,zone,climate),as.factor))
for(v in c("state","zone","climate")) test_frame[[v]] <- factor(test_frame[[v]],levels=levels(final_train[[v]]))

for(v in num_vars){
  if(v %in% names(test_frame)){
    fill <- median(final_train[[v]],na.rm=TRUE)
    test_frame[[v]][!is.finite(test_frame[[v]])] <- NA
    test_frame[[v]][is.na(test_frame[[v]])] <- fill
  }
}

fit_target <- function(target,train_df,vars){
  d <- train_df %>% select(all_of(c(vars,target))) %>% drop_na()
  ranger(as.formula(paste(target,"~",paste(vars,collapse="+"))),data=d,num.trees=N_TREES,mtry=max(2,floor(sqrt(length(vars)))),min.node.size=MIN_NODE,importance="permutation",seed=SEED,quantreg=TRUE)
}

vars <- model_vars[model_vars %in% names(final_train)]
vars <- setdiff(vars,c("target_cases","target_hosp","target_deaths","season"))

# Interpretable Negative Binomial benchmark. This is a benchmark, not the originality claim.
nb_vars <- setdiff(vars,"state")

# -------------------------------------------------
# Negative Binomial benchmark (safe NA handling)
# -------------------------------------------------

final_train_nb <- final_train

hist_vars <- c(
  "hist_peak_week.x",
  "hist_peak_cases",
  "hist_secondary_week",
  "hist_secondary_ratio",
  "hist_two_wave_rate",
  "hist_mean_burden",
  "hist_peak_week.y",
  "hist_peak_rate"
)

for(v in hist_vars){
  
  if(v %in% names(final_train_nb)){
    
    if(v == "hist_secondary_week"){
      
      final_train_nb[[v]][
        is.na(final_train_nb[[v]])
      ] <- 0
      
    } else {
      
      final_train_nb[[v]][
        is.na(final_train_nb[[v]])
      ] <- median(
        final_train_nb[[v]],
        na.rm = TRUE
      )
      
    }
    
  }
  
}

nb_fit <- MASS::glm.nb(
  target_cases ~
    epi_week_of_season +
    hc_access +
    sin1 +
    cos1,
  data = final_train_nb
)

# Apply same fills to test data
for(v in hist_vars){
  
  if(v %in% names(test_frame)){
    
    if(v == "hist_secondary_week"){
      
      test_frame[[v]][
        is.na(test_frame[[v]])
      ] <- 0
      
    } else {
      
      test_frame[[v]][
        is.na(test_frame[[v]])
      ] <- median(
        final_train_nb[[v]],
        na.rm = TRUE
      )
      
    }
    
  }
  
}

nb_fit$xlevels

case_model <- fit_target("target_cases",final_train,vars)
hosp_model <- fit_target("target_hosp",final_train,vars)
death_model <- fit_target("target_deaths",final_train,vars)

predict_q <- function(model,newdata){
  as.data.frame(predict(model,data=newdata,type="quantiles",quantiles=QUANTILES)$predictions)
}

q_cases_raw <- predict_q(case_model,test_frame); names(q_cases_raw)<-paste0("q",c("05","25","50","75","95"))
q_hosp_raw <- predict_q(hosp_model,test_frame); names(q_hosp_raw)<-paste0("q",c("05","25","50","75","95"))
q_death_raw <- predict_q(death_model,test_frame); names(q_death_raw)<-paste0("q",c("05","25","50","75","95"))
nb_test_pred <- as.numeric(predict(nb_fit,newdata=test_frame,type="response"))
summary(nb_test_pred)
#sanity check
length(nb_test_pred)

nrow(test_frame)
any(is.na(nb_test_pred))

# ------------------------- 15. LEAKAGE-SAFE CONFORMAL CALIBRATION -
# Calibration is learned only from 2023/2024 -> 2024/2025.
# 2025/2026 is never used here.
calibrate_target <- function(target,vars){
  tr <- final_train_nb %>%
    filter(season=="2023/2024")
  
  ca <- final_train_nb %>%
    filter(season=="2024/2025")
  
  m <- ranger(as.formula(paste(target,"~",paste(vars,collapse="+"))),data=tr %>% select(all_of(c(vars,target))),num.trees=1000,mtry=max(2,floor(sqrt(length(vars)))),min.node.size=MIN_NODE,quantreg=TRUE,seed=SEED)
  qq <- predict(m,data=ca %>% select(all_of(vars)),type="quantiles",quantiles=QUANTILES)$predictions
  qq <- as.data.frame(qq); names(qq)<-paste0("q",c("05","25","50","75","95"))
  y <- ca[[target]]
  s90 <- pmax(qq$q05-y,y-qq$q95,0); s50 <- pmax(qq$q25-y,y-qq$q75,0)
  list(adj90=conformal_q(s90,0.90),adj50=conformal_q(s50,0.50),n=nrow(ca))
}

cal_cases <- calibrate_target("target_cases",vars)
cal_hosp <- calibrate_target("target_hosp",vars)
cal_death <- calibrate_target("target_deaths",vars)

apply_cal <- function(q,cal){
  tibble(
    q05=pmax(q$q05-cal$adj90,0),q25=pmax(q$q25-cal$adj50,0),q50=pmax(q$q50,0),
    q75=pmax(q$q75+cal$adj50,0),q95=pmax(q$q95+cal$adj90,0)
  )
}

q_cases <- apply_cal(q_cases_raw,cal_cases)
q_hosp <- apply_cal(q_hosp_raw,cal_hosp)
q_death <- apply_cal(q_death_raw,cal_death)

save.image(
  file = file.path(
    OUT_DIR,
    "APIN_checkpoint_after_calibration.RData"
  )
)


# ------------------------- 16. FORECAST OUTPUT --------------------
state_forecast <- test_frame %>% select(state,zone,climate,population,epi_week_of_season) %>%
  bind_cols(q_cases %>% rename_with(~paste0("cases_",.x))) %>%
  bind_cols(q_hosp %>% rename_with(~paste0("hosp_",.x))) %>%
  bind_cols(q_death %>% rename_with(~paste0("death_",.x)))

write_csv(state_forecast,file.path(OUT_DIR,"HAPEN_STATE_FORECAST.csv"))

# Coherent probabilistic aggregation by Monte Carlo.
# Quantiles are sampled state-by-state from the calibrated predictive quantile
# functions and then summed, so national intervals are NOT obtained by summing
# marginal state quantiles.
sample_piecewise <- function(q, n=1000){
  tau <- QUANTILES
  u <- runif(n)
  # Flat-tail extrapolation keeps simulated counts non-negative.
  out <- numeric(n)
  for(i in seq_along(u)){
    ui <- u[i]
    if(ui <= tau[1]) out[i] <- q[1]
    else if(ui >= tau[length(tau)]) out[i] <- q[length(q)]
    else {
      j <- findInterval(ui,tau)
      j <- min(max(j,1),length(tau)-1)
      out[i] <- q[j] + (ui-tau[j])*(q[j+1]-q[j])/(tau[j+1]-tau[j])
    }
  }
  pmax(out,0)
}

make_aggregate_samples <- function(sf, group_vars=c(), n=1000){
  keys <- if(length(group_vars)) sf %>% distinct(across(all_of(c(group_vars,"epi_week_of_season")))) else sf %>% distinct(epi_week_of_season)
  out <- vector("list",nrow(keys))
  for(i in seq_len(nrow(keys))){
    key <- keys[i,]
    z <- sf
    if(length(group_vars)) for(g in group_vars) z <- z %>% filter(.data[[g]]==key[[g]])
    z <- z %>% filter(epi_week_of_season==key$epi_week_of_season)
    cs <- matrix(0,nrow=n,ncol=3)
    for(j in seq_len(nrow(z))){
      cs[,1] <- cs[,1] + sample_piecewise(as.numeric(z[j,c("cases_q05","cases_q25","cases_q50","cases_q75","cases_q95")]),n)
      cs[,2] <- cs[,2] + sample_piecewise(as.numeric(z[j,c("hosp_q05","hosp_q25","hosp_q50","hosp_q75","hosp_q95")]),n)
      cs[,3] <- cs[,3] + sample_piecewise(as.numeric(z[j,c("death_q05","death_q25","death_q50","death_q75","death_q95")]),n)
    }
    out[[i]] <- tibble(epi_week_of_season=key$epi_week_of_season,
                       cases_q05=quantile(cs[,1],.05),cases_q25=quantile(cs[,1],.25),cases_q50=quantile(cs[,1],.50),cases_q75=quantile(cs[,1],.75),cases_q95=quantile(cs[,1],.95),
                       hosp_q05=quantile(cs[,2],.05),hosp_q25=quantile(cs[,2],.25),hosp_q50=quantile(cs[,2],.50),hosp_q75=quantile(cs[,2],.75),hosp_q95=quantile(cs[,2],.95),
                       death_q05=quantile(cs[,3],.05),death_q25=quantile(cs[,3],.25),death_q50=quantile(cs[,3],.50),death_q75=quantile(cs[,3],.75),death_q95=quantile(cs[,3],.95))
    if(length(group_vars)) for(g in group_vars) out[[i]][[g]] <- key[[g]]
  }
  bind_rows(out)
}

national_prob <- make_aggregate_samples(state_forecast,n=1000)
write_csv(national_prob,file.path(OUT_DIR,"HAPEN_NATIONAL_PROBABILISTIC_FORECAST.csv"))

# Coherent point forecasts: aggregate medians from the same state hierarchy.
national_point <- national_prob %>% select(epi_week_of_season,cases_q50,hosp_q50,death_q50)
zone_prob <- make_aggregate_samples(state_forecast,group_vars="zone",n=1000); zone_point <- zone_prob %>% select(zone,epi_week_of_season,cases_q50,hosp_q50,death_q50)
region_map <- state_meta %>% distinct(state,zone) %>% mutate(region=if_else(zone %in% c("NC","NE","NW"),"North","South"))
region_sf <- state_forecast %>% left_join(region_map,by=c("state","zone")); region_prob <- make_aggregate_samples(region_sf,group_vars="region",n=1000); region_point <- region_prob %>% select(region,epi_week_of_season,cases_q50,hosp_q50,death_q50)
write_csv(national_point,file.path(OUT_DIR,"HAPEN_NATIONAL_POINT_FORECAST.csv"))
write_csv(zone_point,file.path(OUT_DIR,"HAPEN_ZONE_POINT_FORECAST.csv"))
write_csv(region_point,file.path(OUT_DIR,"HAPEN_REGION_POINT_FORECAST.csv"))

# ------------------------- 17. FINAL TEST EVALUATION -------------
# Actual 2025/2026 is joined only here, AFTER model selection/calibration.
truth <- test_truth %>% select(state,zone,climate,epi_week_of_season,cases,hospitalizations,deaths) %>% rename(observed_cases=cases,observed_hosp=hospitalizations,observed_death=deaths)
eval_state <- state_forecast %>% left_join(truth,by=c("state","zone","climate","epi_week_of_season"))

metric_block <- function(y,q){
  tibble(
    MAE=mae(y,q$q50),RMSE=rmse(y,q$q50),Coverage50=mean(y>=q$q25 & y<=q$q75,na.rm=TRUE),Coverage90=mean(y>=q$q05 & y<=q$q95,na.rm=TRUE),WIS=mean(wis_vec(y,q$q05,q$q25,q$q50,q$q75,q$q95),na.rm=TRUE),CRPS_approx=mean(map_dbl(seq_along(y),~crps_quantile(y[.x],as.numeric(q[.x, c("q05","q25","q50","q75","q95")]),QUANTILES)),na.rm=TRUE)
  )
}

case_metrics <- metric_block(eval_state$observed_cases,eval_state %>% select(cases_q05,cases_q25,cases_q50,cases_q75,cases_q95) %>% setNames(c("q05","q25","q50","q75","q95")))
hosp_metrics <- metric_block(eval_state$observed_hosp,eval_state %>% select(hosp_q05,hosp_q25,hosp_q50,hosp_q75,hosp_q95) %>% setNames(c("q05","q25","q50","q75","q95")))
death_metrics <- metric_block(eval_state$observed_death,eval_state %>% select(death_q05,death_q25,death_q50,death_q75,death_q95) %>% setNames(c("q05","q25","q50","q75","q95")))

metrics <- bind_rows(cases=case_metrics,hospitalizations=hosp_metrics,deaths=death_metrics,.id="target")
write_csv(metrics,file.path(OUT_DIR,"HAPEN_FINAL_METRICS.csv"))

# National observed-vs-predicted evaluation.
nat_truth <- truth %>% group_by(epi_week_of_season) %>% summarise(observed_cases=sum(observed_cases),observed_hosp=sum(observed_hosp),observed_death=sum(observed_death),.groups="drop")
nat_eval <- national_prob %>% left_join(nat_truth,by="epi_week_of_season")
nat_mae <- tibble(target=c("cases","hospitalizations","deaths"),MAE=c(mae(nat_eval$observed_cases,nat_eval$cases_q50),mae(nat_eval$observed_hosp,nat_eval$hosp_q50),mae(nat_eval$observed_death,nat_eval$death_q50)),RMSE=c(rmse(nat_eval$observed_cases,nat_eval$cases_q50),rmse(nat_eval$observed_hosp,nat_eval$hosp_q50),rmse(nat_eval$observed_death,nat_eval$death_q50)))
write_csv(nat_eval,file.path(OUT_DIR,"HAPEN_NATIONAL_EVALUATION.csv"))
write_csv(nat_mae,file.path(OUT_DIR,"HAPEN_NATIONAL_METRICS.csv"))
nb_state_eval <- tibble(state=test_frame$state,epi_week_of_season=test_frame$epi_week_of_season,nb_pred=nb_test_pred) %>% left_join(truth,by=c("state","epi_week_of_season"))
nb_metrics <- tibble(Model="Negative Binomial benchmark",MAE=mae(nb_state_eval$observed_cases,nb_state_eval$nb_pred),RMSE=rmse(nb_state_eval$observed_cases,nb_state_eval$nb_pred))
write_csv(nb_metrics,file.path(OUT_DIR,"HAPEN_NB_BENCHMARK.csv"))

# Seasonal targets from national point forecast.
peak_cases <- national_point %>% slice_max(cases_q50,n=1,with_ties=FALSE)
obs_peak_cases <- nat_truth %>% slice_max(observed_cases,n=1,with_ties=FALSE)
seasonal_summary_out <- tibble(
  metric=c("Observed peak week","Predicted peak week","Peak week absolute error","Observed peak cases","Predicted peak cases","Peak magnitude error percent","Observed cumulative cases","Predicted cumulative cases","Cumulative burden error percent"),
  value=c(obs_peak_cases$epi_week_of_season,peak_cases$epi_week_of_season,abs(obs_peak_cases$epi_week_of_season-peak_cases$epi_week_of_season),obs_peak_cases$observed_cases,peak_cases$cases_q50,100*abs(obs_peak_cases$observed_cases-peak_cases$cases_q50)/obs_peak_cases$observed_cases,sum(nat_truth$observed_cases),sum(national_point$cases_q50),100*abs(sum(nat_truth$observed_cases)-sum(national_point$cases_q50))/sum(nat_truth$observed_cases))
)
seasonal_target_table <- tibble(
  target=c("cases","hospitalizations","deaths"),
  observed_cumulative=c(sum(nat_truth$observed_cases),sum(nat_truth$observed_hosp),sum(nat_truth$observed_death)),
  predicted_cumulative=c(sum(national_point$cases_q50),sum(national_point$hosp_q50),sum(national_point$death_q50))
) %>% mutate(absolute_error=abs(observed_cumulative-predicted_cumulative),percent_error=100*absolute_error/pmax(observed_cumulative,1))
seasonal_target_table <- bind_rows(seasonal_target_table,tibble(target="attack_rate_pct",observed_cumulative=100*sum(nat_truth$observed_cases)/sum(state_meta$population),predicted_cumulative=100*sum(national_point$cases_q50)/sum(state_meta$population),absolute_error=NA_real_,percent_error=NA_real_))
write_csv(seasonal_summary_out,file.path(OUT_DIR,"HAPEN_SEASONAL_TARGETS.csv"))
write_csv(seasonal_target_table,file.path(OUT_DIR,"HAPEN_SEASONAL_ALL_TARGETS.csv"))

# ------------------------- 18. ABLATION ---------------------------
# Compare the final APIN architecture with a calendar/static model on the
# development split. The APIN component is only useful if it improves out-of-sample prediction.
base_vars <- setdiff(vars,c("apin_pressure","apin_sources","tropical_to_savanna","savanna_to_sahel","tropical_to_sahel"))
apin_vars <- vars
ablation <- tibble()
for(nm in c("Static-historical","+Climate-transition","+APIN-full")){
  vv <- if(nm=="Static-historical") base_vars else if(nm=="+Climate-transition") unique(c(base_vars,"tropical_to_savanna","savanna_to_sahel","tropical_to_sahel")) else apin_vars
  tr <- final_train_nb %>%
    filter(season == "2023/2024")
  
  va <- final_train_nb %>%
    filter(season == "2024/2025")
  m <- ranger(target_cases~.,data=tr %>% select(all_of(c(vv,"target_cases"))),num.trees=700,mtry=max(2,floor(sqrt(length(vv)))),min.node.size=MIN_NODE,seed=SEED)
  p <- predict(m,data=va %>% select(all_of(vv)))$predictions
  ablation <- bind_rows(ablation,tibble(Model=nm,MAE=mae(va$target_cases,p),RMSE=rmse(va$target_cases,p)))
}
write_csv(ablation,file.path(OUT_DIR,"HAPEN_ABLATION.csv"))
# ------------------------- 19. VISUALS -----------------------------
# National historical curves + final held-out prediction.
hist_nat <- national %>% filter(season %in% c(TRAIN_SEASONS,TEST_SEASON)) %>% ggplot(aes(epi_week_of_season,cases,group=season)) + geom_line(aes(linetype=season),linewidth=0.9) + theme_minimal() + labs(title="National Influenza Activity Across Supplied Seasons",x="Epidemic week",y="Weekly cases",linetype="Season")
ggsave(file.path(OUT_DIR,"figures","01_national_historical_curves.png"),hist_nat,width=10,height=6,dpi=300)

pred_plot <- nat_eval %>% ggplot(aes(epi_week_of_season)) + geom_line(aes(y=observed_cases),linewidth=1) + geom_line(aes(y=cases_q50),linewidth=1,linetype=2) + theme_minimal() + labs(title="Fully Held-Out 2025/2026 National Forecast",subtitle="Median forecast trained without any 2025/2026 outcomes",x="Epidemic week",y="Cases")
ggsave(file.path(OUT_DIR,"figures","02_test_national_cases.png"),pred_plot,width=10,height=6,dpi=300)

ab_plot <- ablation %>% ggplot(aes(reorder(Model,MAE),MAE)) + geom_col() + coord_flip() + theme_minimal() + labs(title="Development Ablation: Contribution of Climate and APIN Layers",x="Model",y="Validation MAE")
ggsave(file.path(OUT_DIR,"figures","03_ablation.png"),ab_plot,width=9,height=6,dpi=300)

net_plot <- network %>% filter(rank<=5) %>% ggplot(aes(lag_corr)) + geom_histogram(bins=30) + theme_minimal() + labs(title="Training-Learned APIN Lagged Association Distribution",x="Lagged correlation",y="Number of state links")
ggsave(file.path(OUT_DIR,"figures","04_apin_network.png"),net_plot,width=9,height=6,dpi=300)

# ------------------------- 20. REPORT-READY MODEL SPECIFICATION --
model_spec <- c(
  "Model name: Adaptive Predictive Influence Network (APIN) + hierarchical probabilistic quantile forecasting.",
  "1. Historical phenotype is computed from 2023/2024 and 2024/2025 only for the final model.",
  "2. APIN learns state-to-state lagged predictive links from deseasonalised log incidence.",
  "3. Link scores combine positive lagged association, climate similarity, epidemic timing similarity and healthcare-access similarity.",
  "4. Only the top K links per target state are retained; K is selected on the 2023/2024 -> 2024/2025 development split.",
  "5. The final 2025/2026 season is fully held out: no test-season outcome is used as a predictor, feature-selection input, network-learning input or calibration datum.",
  "6. State forecasts are produced at quantiles 0.05, 0.25, 0.50, 0.75 and 0.95; state medians are aggregated to zone, region and national levels.",
  "7. Conformal interval expansion is calibrated only on the 2024/2025 development season after fitting on 2023/2024.",
  "8. A Negative Binomial model is retained as an interpretable count-model benchmark, not as the novelty claim."
)
writeLines(model_spec,file.path(OUT_DIR,"MODEL_SPECIFICATION.txt"))

# ------------------------- 20. REPRODUCIBILITY --------------------
writeLines(capture.output(sessionInfo()),file.path(OUT_DIR,"sessionInfo.txt"))
writeLines(c(
  paste0("Seed=",SEED),
  paste0("Training seasons=",paste(TRAIN_SEASONS,collapse=";")),
  paste0("Final test season=",TEST_SEASON),
  paste0("Selected APIN K=",selected_k),
  "Test-season outcomes used in feature learning: NO",
  "Test-season outcomes used in APIN network learning: NO",
  "Test-season outcomes used in hyperparameter selection: NO",
  "Test-season outcomes used in conformal calibration: NO",
  "External influenza surveillance data used: NO",
  "Only supplied challenge datasets used: YES"
),file.path(OUT_DIR,"LEAKAGE_AUDIT.txt"))

# ------------------------- 21. CONSOLE SCORECARD -------------------
msg("HAPEN completed.")
msg("Selected APIN K: ",selected_k)
msg("Final test season: ",TEST_SEASON)
msg("Case MAE: ",round(case_metrics$MAE,3)," | RMSE: ",round(case_metrics$RMSE,3)," | WIS: ",round(case_metrics$WIS,3)," | CRPS approximation: ",round(case_metrics$CRPS_approx,3))
msg("Case 50% coverage: ",round(case_metrics$Coverage50,3)," | 90% coverage: ",round(case_metrics$Coverage90,3))
msg("All outputs written to: ",normalizePath(OUT_DIR))


##########################################################
#### adding forecast uncertainty bands:

p_eval <- ggplot(
  nat_eval,
  aes(x = epi_week_of_season)
) +
  
  geom_ribbon(
    aes(
      ymin = cases_q05,
      ymax = cases_q95
    ),
    fill = "steelblue",
    alpha = 0.20
  ) +
  
  geom_ribbon(
    aes(
      ymin = cases_q25,
      ymax = cases_q75
    ),
    fill = "steelblue",
    alpha = 0.40
  ) +
  
  geom_line(
    aes(
      y = cases_q50,
      colour = "Forecast"
    ),
    linewidth = 1
  ) +
  
  geom_line(
    aes(
      y = observed_cases,
      colour = "Observed"
    ),
    linewidth = 1
  ) +
  
  scale_colour_manual(
    values = c(
      Forecast = "steelblue",
      Observed = "black"
    )
  ) +
  
  theme_minimal() +
  
  labs(
    title = "Observed vs Predicted National Influenza Cases (2025/2026)",
    x = "Epidemic Week",
    y = "Cases",
    colour = ""
  )

ggsave(
  file.path(
    OUT_DIR,
    "figures",
    "24_observed_vs_predicted_uncertainty.png"
  ),
  p_eval,
  width = 10,
  height = 6,
  dpi = 300
)

#adding legend
p_eval <- ggplot(
  nat_eval,
  aes(x = epi_week_of_season)
) +
  
  geom_ribbon(
    aes(
      ymin = cases_q05,
      ymax = cases_q95,
      fill = "90% Prediction Interval"
    ),
    alpha = 0.20
  ) +
  
  geom_ribbon(
    aes(
      ymin = cases_q25,
      ymax = cases_q75,
      fill = "50% Prediction Interval"
    ),
    alpha = 0.40
  ) +
  
  geom_line(
    aes(
      y = cases_q50,
      colour = "Forecast Median"
    ),
    linewidth = 1
  ) +
  
  geom_line(
    aes(
      y = observed_cases,
      colour = "Observed Cases"
    ),
    linewidth = 1
  ) +
  
  scale_fill_manual(
    values = c(
      "50% Prediction Interval" = "steelblue",
      "90% Prediction Interval" = "lightblue"
    )
  ) +
  
  scale_colour_manual(
    values = c(
      "Forecast Median" = "steelblue",
      "Observed Cases" = "black"
    )
  ) +
  
  guides(
    fill = guide_legend(order = 1),
    colour = guide_legend(order = 2)
  ) +
  
  labs(
    title = "Observed vs Predicted National Influenza Cases (2025/2026)",
    subtitle = "Median forecast with 50% and 90% prediction intervals",
    x = "Epidemic Week",
    y = "Cases",
    fill = "",
    colour = ""
  ) +
  
  theme_minimal()

ggsave(
  file.path(
    OUT_DIR,
    "figures",
    "24_observed_vs_predicted_uncertainty.png"
  ),
  p_eval,
  width = 10,
  height = 6,
  dpi = 300
)

# national probabilistic influenza forecast

library(ggplot2)

p_fan <- ggplot(
  national_prob,
  aes(x = epi_week_of_season)
) +
  
  geom_ribbon(
    aes(
      ymin = cases_q05,
      ymax = cases_q95
    ),
    fill = "steelblue",
    alpha = 0.25
  ) +
  
  geom_ribbon(
    aes(
      ymin = cases_q25,
      ymax = cases_q75
    ),
    fill = "steelblue",
    alpha = 0.45
  ) +
  
  geom_line(
    aes(y = cases_q50),
    linewidth = 1,
    colour = "black"
  ) +
  
  theme_minimal() +
  
  labs(
    title = "National Probabilistic Influenza Forecast",
    subtitle = "Median forecast with 50% and 90% prediction intervals",
    x = "Epidemic Week",
    y = "Forecast Weekly Cases"
  )

ggsave(
  file.path(
    OUT_DIR,
    "figures",
    "23_national_fan_chart.png"
  ),
  p_fan,
  width = 10,
  height = 6,
  dpi = 300
)

#with legend

p_fan <- ggplot(
  national_prob,
  aes(x = epi_week_of_season)
) +
  
  geom_ribbon(
    aes(
      ymin = cases_q05,
      ymax = cases_q95,
      fill = "90% Prediction Interval"
    ),
    alpha = 0.25
  ) +
  
  geom_ribbon(
    aes(
      ymin = cases_q25,
      ymax = cases_q75,
      fill = "50% Prediction Interval"
    ),
    alpha = 0.45
  ) +
  
  geom_line(
    aes(
      y = cases_q50,
      colour = "Median Forecast"
    ),
    linewidth = 1
  ) +
  
  scale_fill_manual(
    name = "",
    values = c(
      "90% Prediction Interval" = "lightblue",
      "50% Prediction Interval" = "steelblue"
    )
  ) +
  
  scale_colour_manual(
    name = "",
    values = c(
      "Median Forecast" = "black"
    )
  ) +
  
  labs(
    title = "National Probabilistic Influenza Forecast",
    subtitle = "Median forecast with 50% and 90% prediction intervals",
    x = "Epidemic Week",
    y = "Forecast Weekly Cases"
  ) +
  
  guides(
    fill = guide_legend(order = 1),
    colour = guide_legend(order = 2)
  ) +
  
  theme_minimal()

p_fan


#forecast validation figure
library(ggplot2)

p_validation <- ggplot(
  nat_eval,
  aes(x = epi_week_of_season)
) +
  
  geom_ribbon(
    aes(
      ymin = cases_q05,
      ymax = cases_q95,
      fill = "90% Prediction Interval"
    ),
    alpha = 0.25
  ) +
  
  geom_ribbon(
    aes(
      ymin = cases_q25,
      ymax = cases_q75,
      fill = "50% Prediction Interval"
    ),
    alpha = 0.45
  ) +
  
  geom_line(
    aes(
      y = cases_q50,
      colour = "Forecast Median"
    ),
    linewidth = 1
  ) +
  
  geom_line(
    aes(
      y = observed_cases,
      colour = "Observed Cases"
    ),
    linewidth = 1
  ) +
  
  scale_fill_manual(
    values = c(
      "90% Prediction Interval" = "lightblue",
      "50% Prediction Interval" = "steelblue"
    )
  ) +
  
  scale_colour_manual(
    values = c(
      "Forecast Median" = "blue",
      "Observed Cases" = "black"
    )
  ) +
  
  labs(
    title = "Observed vs Predicted National Influenza Cases (2025/2026)",
    x = "Epidemic Week",
    y = "Weekly Cases",
    fill = "",
    colour = ""
  ) +
  
  theme_minimal()

p_validation

ggsave(
  filename = file.path(
    OUT_DIR,
    "figures",
    "Figure_09_Observed_vs_Predicted_2025_2026.png"
  ),
  plot = p_validation,
  width = 10,
  height = 6,
  dpi = 300
)



#APIN_training_network.csv
install.packages(c(
  "igraph",
  "ggraph",
  "tidygraph",
  "ggplot2",
  "dplyr",
  "readr"
))

library(readr)
library(dplyr)
library(igraph)
library(tidygraph)
library(ggraph)
library(ggplot2)

#--------------------------------------------------
# Read APIN network
#--------------------------------------------------

apin_network <- read_csv(
  file.path(
    OUT_DIR,
    "APIN_training_network.csv"
  ),
  show_col_types = FALSE
)

#--------------------------------------------------
# Build graph object
#--------------------------------------------------

g <- graph_from_data_frame(
  d = apin_network %>%
    select(
      source,
      target,
      adaptive_score
    ),
  directed = TRUE
)

#--------------------------------------------------
# Network plot
#--------------------------------------------------

p_apin_full <- ggraph(
  g,
  layout = "fr"
) +
  
  geom_edge_link(
    aes(
      width = adaptive_score
    ),
    alpha = 0.40,
    colour = "steelblue"
  ) +
  
  geom_node_point(
    size = 4,
    colour = "darkred"
  ) +
  
  geom_node_text(
    aes(label = name),
    repel = TRUE,
    size = 3
  ) +
  
  scale_edge_width(
    range = c(0.3, 2.5),
    name = "Influence Score"
  ) +
  
  theme_void() +
  
  labs(
    title = "Adaptive Predictive Influence Network (APIN)",
    subtitle = "All learned state-to-state predictive relationships"
  )

# Display
p_apin_full

# Save
ggsave(
  file.path(
    OUT_DIR,
    "figures",
    "10_APIN_network_full.png"
  ),
  p_apin_full,
  width = 14,
  height = 12,
  dpi = 300
)


# forecasting architecture

library(ggplot2)

architecture <- data.frame(
  x = 1,
  y = c(10,9,8,7,6,5,4,3,2,1),
  label = c(
    "Historical Epidemic\nPhenotype",
    "Climate Transition\nLayer",
    "Adaptive Predictive\nInfluence Network (APIN)",
    "Quantile\nForecasting",
    "Conformal\nCalibration",
    "State\nForecasts",
    "Zone\nForecasts",
    "Regional\nForecasts (North/South)",
    "National\nForecasts",
    "Policy & Decision\nSupport"
  )
)

p_arch <- ggplot(architecture, aes(x, y)) +
  
  annotate(
    "segment",
    x = 1,
    xend = 1,
    y = 9.6,
    yend = 1.4,
    linewidth = 1,
    arrow = arrow(length = unit(0.25, "cm"))
  )+
  
  geom_label(
    aes(label = label),
    fill = "lightblue",
    colour = "black",
    size = 4,
    label.size = 0.3,
    label.padding = unit(0.25, "lines")
  ) +
  
  xlim(0.5, 1.5) +
  ylim(0.5, 10.5) +
  
  theme_void() +
  
  labs(
    title = "Hierarchical Adaptive Probabilistic Epidemic Network (HAPEN) Framework"
  )

p_arch

ggsave(
  file.path(
    OUT_DIR,
    "figures",
    "FIGURE_01_HAPEN_ARCHITECTURE.png"
  ),
  p_arch,
  width = 8,
  height = 10,
  dpi = 300
)

##########################################################
#For the reccomended Visualization
#0. Setup — packages and paths
# 00_setup.R  — corrected
rm(list=ls())
required <- c("dplyr", "tidyr", "readr", "ggplot2", "purrr", "MASS", "scales")
new <- required[!required %in% installed.packages()[, "Package"]]
if (length(new)) install.packages(new)
invisible(lapply(required, library, character.only = TRUE))

# Force these to always resolve to dplyr, no matter what else gets loaded later
select    <- dplyr::select
filter    <- dplyr::filter
mutate    <- dplyr::mutate
summarise <- dplyr::summarise
summarize <- dplyr::summarise
arrange   <- dplyr::arrange
rename    <- dplyr::rename

DATA_DIR <- "."             # your 5 CSVs are in the working directory itself
OUT_DIR  <- "outputs"
dir.create(file.path(OUT_DIR, "figures"),   showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "tables"),    showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "forecasts"), showWarnings = FALSE, recursive = TRUE)

state_meta    <- read_csv(file.path(DATA_DIR, "nigeria_flu_state_metadata.csv"),   show_col_types = FALSE)
season_meta   <- read_csv(file.path(DATA_DIR, "nigeria_flu_season_reference.csv"), show_col_types = FALSE)
state_weekly  <- read_csv(file.path(DATA_DIR, "nigeria_flu_weekly_by_state.csv"),  show_col_types = FALSE)
state_summary <- read_csv(file.path(DATA_DIR, "nigeria_flu_season_summary.csv"),   show_col_types = FALSE)
national      <- read_csv(file.path(DATA_DIR, "nigeria_flu_weekly_national.csv"), show_col_types = FALSE)

state_weekly$week_start        <- as.Date(state_weekly$week_start)
national$week_start            <- as.Date(national$week_start)
state_summary$peak_week_start  <- as.Date(state_summary$peak_week_start)
season_meta$season_start       <- as.Date(season_meta$season_start)

# CORRECTED: your zone column uses NC/NE/NW/SE/SS/SW abbreviations, not full words
to_region <- function(zone) {
  z <- toupper(trimws(zone))
  ifelse(z %in% c("NC", "NE", "NW"), "North",
         ifelse(z %in% c("SE", "SS", "SW"), "South", NA))
}
state_meta$region    <- to_region(state_meta$zone)
state_weekly$region  <- to_region(state_weekly$zone)
state_summary$region <- to_region(state_summary$zone)

cat("States:", length(unique(state_meta$state)),
    " Seasons:", length(unique(season_meta$season)),
    " Weeks/season:", max(state_weekly$epi_week_of_season), "\n")
print(table(state_meta$region, useNA = "always"))

#1. QA — validation, reconciliation, missing values

# 01_qa.R  (source 00_setup.R first)
sink(file.path(OUT_DIR, "tables", "qa_report.txt"))

cat("=== SHAPES ===\n")
cat("state_meta:", dim(state_meta), "\n")
cat("season_meta:", dim(season_meta), "\n")
cat("state_weekly:", dim(state_weekly), "\n")
cat("state_summary:", dim(state_summary), "\n")
cat("national:", dim(national), "\n")

cat("\n=== MISSING VALUES ===\n")
for (nm in c("state_meta","season_meta","state_weekly","state_summary","national")) {
  df <- get(nm)
  na_counts <- sapply(df, function(x) sum(is.na(x)))
  na_counts <- na_counts[na_counts > 0]
  cat(nm, ": ", if (length(na_counts)) paste(names(na_counts), na_counts, sep="=", collapse=", ") else "none", "\n")
}

cat("\n=== DUPLICATES ===\n")
cat("dup (season,state,week) in state_weekly:",
    sum(duplicated(state_weekly[, c("season","state","epi_week_of_season")])), "\n")
cat("dup (season,week) in national:",
    sum(duplicated(national[, c("season","epi_week_of_season")])), "\n")

cat("\n=== NEGATIVES ===\n")
for (v in c("cases","hospitalizations","deaths"))
  cat("negative", v, ":", sum(state_weekly[[v]] < 0, na.rm = TRUE), "\n")

cat("\n=== WEEK ALIGNMENT (should start Sunday) ===\n")
cat("week_start not Sunday:", sum(format(state_weekly$week_start, "%u") != "7"), "of", nrow(state_weekly), "\n")

cat("\n=== STATE SUM vs NATIONAL RECONCILIATION ===\n")
agg <- state_weekly %>%
  group_by(season, epi_week_of_season) %>%
  summarise(cases = sum(cases), hospitalizations = sum(hospitalizations),
            deaths = sum(deaths), .groups = "drop")
chk <- agg %>%
  inner_join(national, by = c("season","epi_week_of_season"), suffix = c("_st","_nat"))
for (v in c("cases","hospitalizations","deaths")) {
  d <- chk[[paste0(v,"_st")]] - chk[[paste0(v,"_nat")]]
  cat(v, ": max|diff| =", max(abs(d)), "  total_states =", sum(chk[[paste0(v,"_st")]]),
      " total_national =", sum(chk[[paste0(v,"_nat")]]), "\n")
}

cat("\n=== SEASONAL SUMMARY RE-COMPUTATION CHECK ===\n")
rc <- state_weekly %>%
  group_by(season, state) %>%
  summarise(total_cases_rc = sum(cases), total_hosp_rc = sum(hospitalizations),
            total_deaths_rc = sum(deaths), peak_weekly_cases_rc = max(cases),
            peak_epi_week_rc = epi_week_of_season[which.max(cases)],
            pop = first(population), .groups = "drop") %>%
  mutate(attack_rate_pct_rc = total_cases_rc / pop * 100,
         cfr_pct_rc = total_deaths_rc / pmax(total_cases_rc, 1) * 100)
mm <- state_summary %>% inner_join(rc, by = c("season","state"))
cat("total_cases mismatches:", sum(abs(mm$total_cases - mm$total_cases_rc) > 0.5), "\n")
cat("total_hospitalizations mismatches:", sum(abs(mm$total_hospitalizations - mm$total_hosp_rc) > 0.5), "\n")
cat("total_deaths mismatches:", sum(abs(mm$total_deaths - mm$total_deaths_rc) > 0.5), "\n")
cat("peak_weekly_cases mismatches:", sum(abs(mm$peak_weekly_cases - mm$peak_weekly_cases_rc) > 0.5), "\n")
cat("attack_rate_pct max abs diff:", max(abs(mm$attack_rate_pct - mm$attack_rate_pct_rc)), "\n")
cat("cfr_pct max abs diff:", max(abs(mm$cfr_pct - mm$cfr_pct_rc)), "\n")

cat("\n=== SEASON DESCRIPTIVES ===\n")
print(national %>% group_by(season) %>%
        summarise(total_cases = sum(cases), peak_cases = max(cases),
                  total_hosp = sum(hospitalizations), total_deaths = sum(deaths)))
sink()

cat(readLines(file.path(OUT_DIR, "tables", "qa_report.txt")), sep = "\n")
# 2. EDA — the recommended visualisations
# 02_eda.R  (source 00_setup.R first — must have the corrected to_region() applied)

# --- National weekly cases/hosp/deaths ---
p1 <- national %>%
  tidyr::pivot_longer(c(cases, hospitalizations, deaths), names_to = "metric", values_to = "value") %>%
  ggplot(aes(epi_week_of_season, value, color = season)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~metric, ncol = 1, scales = "free_y") +
  labs(title = "National weekly cases, hospitalisations and deaths",
       x = "Epi week of season", y = NULL) + theme_minimal()
ggsave(file.path(OUT_DIR,"figures","01_national_weekly.png"), p1, width = 9, height = 9)

# --- Regional (zone) epidemic curves, incidence per 100k ---
zone_pop <- state_meta %>% dplyr::group_by(zone) %>% dplyr::summarise(pop = sum(population))
zone_curve <- state_weekly %>%
  dplyr::group_by(season, zone, epi_week_of_season) %>%
  dplyr::summarise(cases = sum(cases), .groups = "drop") %>%
  dplyr::left_join(zone_pop, by = "zone") %>%
  dplyr::mutate(inc = cases / pop * 1e5)
p2 <- ggplot(zone_curve, aes(epi_week_of_season, inc, color = zone)) +
  geom_line() + facet_wrap(~season) +
  labs(title = "Regional (zone) epidemic curves", x = "Epi week", y = "Cases per 100k") +
  theme_minimal()
ggsave(file.path(OUT_DIR,"figures","02_zone_curves.png"), p2, width = 12, height = 4.5)

# --- North vs South (uses the CORRECTED region column) ---
region_pop <- state_meta %>% dplyr::group_by(region) %>% dplyr::summarise(pop = sum(population))
region_curve <- state_weekly %>%
  dplyr::group_by(season, region, epi_week_of_season) %>%
  dplyr::summarise(cases = sum(cases), .groups = "drop") %>%
  dplyr::left_join(region_pop, by = "region") %>%
  dplyr::mutate(inc = cases / pop * 1e5)
p3 <- ggplot(region_curve, aes(epi_week_of_season, inc, color = region)) +
  geom_line(linewidth = 1) + facet_wrap(~season) +
  labs(title = "North vs South wave comparison", x = "Epi week", y = "Cases per 100k") +
  theme_minimal()
ggsave(file.path(OUT_DIR,"figures","03_north_vs_south.png"), p3, width = 12, height = 4.5)

# sanity check printed to console: should show TWO regions, not one
cat("Regions present in region_curve:", paste(unique(region_curve$region), collapse = ", "), "\n")

# --- State-level weekly heatmap ---
state_order <- state_meta %>% dplyr::arrange(zone, state) %>% dplyr::pull(state)
hm <- state_weekly %>%
  dplyr::mutate(state = factor(state, levels = state_order))
p4 <- ggplot(hm, aes(epi_week_of_season, state, fill = log1p(cases_per_100k))) +
  geom_tile() + facet_wrap(~season, nrow = 1) +
  scale_fill_viridis_c(option = "magma", direction = -1) +
  theme_minimal(base_size = 7) +
  labs(title = "State-level weekly incidence heatmap", x = "Epi week", y = NULL, fill = "log(1+inc)")
ggsave(file.path(OUT_DIR,"figures","04_state_heatmap.png"), p4, width = 14, height = 10)


# --- Peak-week summary by zone ---
p7 <- ggplot(state_summary, aes(zone, peak_epi_week, fill = zone)) +
  geom_boxplot(show.legend = FALSE) + facet_wrap(~season) +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Peak epi week by zone", x = NULL, y = "Peak epi week")
ggsave(file.path(OUT_DIR,"figures","07_peak_week_by_zone.png"), p7, width = 12, height = 4.5)

cat("EDA figures written to", file.path(OUT_DIR, "figures"), "\n")
list.files(file.path(OUT_DIR, "figures"))

#to view the exact figure
file.path(OUT_DIR, "figures")
list.files(file.path(OUT_DIR, "figures"))

#####drawing Nigeria maps
install.packages("sp")
library(sp)
install.packages("spdep")
library(spdep)
install.packages("XML")
library(XML)
install.packages("tmap")
library(tmap)
install.packages("tidyverse")
library(tidyverse)
library(readxl)
install.packages("RColorBrewer")
library(RColorBrewer)
library(haven)
install.packages("leaflet")
library(leaflet)
install.packages("mapview")
library(mapview)
library(terra)
install.packages("lwgeom")
library(tmap)
install.packages("SpatialEpi")
library(SpatialEpi)

install.packages("sf")
library(sf)

install.packages("ggplot2")
library(ggplot2)
# Read the shapefile


nig0<- st_read("C:/Users/user/Downloads/gadm41_NGA_0.shp")

nig1 <- st_read("C:/Users/user/Downloads/gadm41_NGA_1.shp")

names(nig1)
head(nig1)

sort(nig1$NAME_1)   # adjust NAME_1 if the actual column name differs

# If state_summary isn't loaded, re-run this (adjust path if needed):
state_summary <- read_csv("nigeria_flu_season_summary.csv", show_col_types = FALSE)

# Then confirm it's there:
sort(unique(state_summary$state))

library(dplyr)
library(sf)
library(ggplot2)

# --- Fix the ONE name mismatch we found ---
nig1$state_matched <- ifelse(nig1$NAME_1 == "Federal Capital Territory", "FCT", nig1$NAME_1)

# --- Merge attack-rate data onto the shapefile ---
map_data <- nig1 %>%
  dplyr::left_join(state_summary, by = c("state_matched" = "state"))

# sanity check: should be 0 unmatched
cat("Unmatched rows (should be 0):", sum(is.na(map_data$attack_rate_pct)), "\n")
cat("States with no match at all:\n")
print(unique(map_data$state_matched[is.na(map_data$attack_rate_pct)]))

# --- Plot: state attack rate, one map per season ---
p_map <- ggplot(map_data) +
  geom_sf(aes(fill = attack_rate_pct), color = "white", linewidth = 0.1) +
  facet_wrap(~season, ncol = 3) +
  scale_fill_viridis_c(option = "magma", direction = -1, name = "Attack rate (%)") +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Nigeria: State-level influenza attack rate by season")
ggsave(file.path(OUT_DIR, "figures", "05b_state_attack_rate_MAP.png"), p_map, width = 14, height = 6)
print(p_map)

# --- Plot: state cumulative mortality, one map per season ---
p_map_death <- ggplot(map_data) +
  geom_sf(aes(fill = cum_deaths_per_100k), color = "white", linewidth = 0.1) +
  facet_wrap(~season, ncol = 3) +
  scale_fill_viridis_c(option = "rocket", direction = -1, name = "Deaths per 100k") +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Nigeria: State-level cumulative mortality by season")
ggsave(file.path(OUT_DIR, "figures", "06b_state_mortality_MAP.png"), p_map_death, width = 14, height = 6)
print(p_map_death)



