%% =====================================================================
% MAEPiMS CHALLENGE 2026 — NIGERIA SEASONAL INFLUENZA FORECASTING
% CPH-DAHU-ZSD MODEL
% =====================================================================
% Load data, fit national and state models, generate probabilistic
% forecasts, evaluate predictions, save outputs, and generate figures.

1;  % enable local functions in the script
clear; clc; close all; rng(20260101,'twister');   % reproducible

%% -------------------- 0. CONFIGURATION ---------------------------
DATA_DIR = pwd;                                   % data folder
OUT_DIR  = fullfile(pwd,'maepims_output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

TRAIN_SEASONS = {'2023/2024','2024/2025'};
TEST_SEASON   = '2025/2026';
NWK           = 52;             % weeks per season
NSIM          = 6000;           % national Monte-Carlo trajectories
NSIM_STATE    = 1500;           % state Monte-Carlo trajectories
NHARM         = 5;              % seasonal harmonics
NINTERACT     = 3;              % severity interactions with low harmonics
SEV_JITTER_SD = 0.1;            % severity jitter
BLOCK_LEN     = 4;              % state/zone bootstrap block length
JITTER_FRAC   = 0.25;           % bootstrap jitter fraction
K_SHRINK      = 3.0e6;          % state-to-zone population shrinkage
QLEVELS       = [0.025 0.05 0.10 0.25 0.50 0.75 0.90 0.95 0.975];
NORTH_ZONES   = {'NC','NE','NW'};
SOUTH_ZONES   = {'SE','SS','SW'};

fprintf('=== MAEPiMS 2026 Flu Forecast (CPH-DAHU-ZSD model, v2) ===\n');

%% -------------------- 1. LOAD DATA ---------------------------------
% Load national weekly data, state weekly data, state metadata,
% season summaries and reference data.
natT   = readtable(findFile(DATA_DIR,'*weekly_national*.csv'));
stT    = readtable(findFile(DATA_DIR,'*weekly_by_state*.csv'));
sumT   = readtable(findFile(DATA_DIR,'*season_summary*.csv'));  %#ok<NASGU>
metaT  = readtable(findFile(DATA_DIR,'*state_metadata*.csv'));
refT   = readtable(findFile(DATA_DIR,'*season_reference*.csv')); %#ok<NASGU>

% Standardise season, state and zone fields.
natT.season = string(natT.season);
stT.season  = string(stT.season);
metaT.state = string(metaT.state);
stT.state   = string(stT.state);
stT.zone    = string(stT.zone);
metaT.zone  = string(metaT.zone);

% Extract state populations, zones and healthcare-access values.
allStates = sort(unique(stT.state));
nStates   = numel(allStates);
statePop  = zeros(nStates,1); stateZone = strings(nStates,1); stateHC = zeros(nStates,1);
for i=1:nStates
    r = metaT(metaT.state==allStates(i),:);
    statePop(i)  = r.population(1);
    stateZone(i) = r.zone(1);
    stateHC(i)   = r.hc_access(1);
end
meanHC = mean(stateHC);
natPop = natT.population(1);

fprintf('Loaded %d states/FCT, %d national weekly rows, %d state-week rows.\n', ...
    nStates, height(natT), height(stT));

%% -------------------- 2. DATA-DRIVEN SEVERITY COVARIATE ------------
% Calculate cumulative cases per 100,000 for each training season.
% Standardise the two training-season severity values.
% Forecast severity is sampled between the training values with jitter.
sevRaw = containers.Map('KeyType','char','ValueType','double');
for si = 1:numel(TRAIN_SEASONS)
    s = TRAIN_SEASONS{si};
    sub = natT(natT.season==s,:);
    sevRaw(s) = sum(sub.cases) / natPop * 1e5;
end
trainSevVals = cellfun(@(s) sevRaw(s), TRAIN_SEASONS);
sevMu = mean(trainSevVals); sevSd = std(trainSevVals); if sevSd==0, sevSd=1; end
sevZ = containers.Map('KeyType','char','ValueType','double');
for si = 1:numel(TRAIN_SEASONS)
    s = TRAIN_SEASONS{si};
    sevZ(s) = (sevRaw(s) - sevMu) / sevSd;
end
trainSevZVals = cellfun(@(s) sevZ(s), TRAIN_SEASONS);
sevLo = min(trainSevZVals); sevHi = max(trainSevZVals);
fprintf('Training-season severity (cum. cases/100k): ');
for si=1:numel(TRAIN_SEASONS), fprintf('%s=%.0f  ',TRAIN_SEASONS{si},sevRaw(TRAIN_SEASONS{si})); end
fprintf('\n');

%% -------------------- 3. NATIONAL MODEL: FIT ------------------------
% Fit the national harmonic model with severity interactions.
% Fit a pooled AR(1) process to the training residuals.
[betaN, residTrainN] = fitHarmonic(natT, TRAIN_SEASONS, NWK, NHARM, NINTERACT, sevZ);

tTest = (1:NWK)';
HdTest = harmonicDesign(tTest, NWK, NHARM);

[phiN, sigmaN] = fitAR1(residTrainN, NWK);
fprintf('National residual AR(1): phi=%.3f  sigma=%.3f  (used as-fitted)\n', phiN, sigmaN);

% Generate national case trajectories using severity sampling and AR(1) residuals.
simCasesN = simulateAR1Severity(HdTest, betaN, NINTERACT, sevLo, sevHi, SEV_JITTER_SD, phiN, sigmaN, NSIM, NWK);

%% -------------------- 4. OUTCOME CASCADE: HOSP & DEATHS -------------
% Fit harmonic outcome-to-case ratios and AR(1) residuals.
[hospRatioTest, phiH, sigmaH] = fitRatioHarmonicAR(natT, TRAIN_SEASONS, NWK, 'hospitalizations', tTest);
[deathRatioTest, phiD, sigmaD] = fitRatioHarmonicAR(natT, TRAIN_SEASONS, NWK, 'deaths', tTest);
fprintf('Hosp.-ratio AR(1): phi=%.3f sigma=%.3f | Death-ratio AR(1): phi=%.3f sigma=%.3f\n', phiH, sigmaH, phiD, sigmaD);

% Apply the outcome ratios to each national case trajectory.
simHospN  = simulateAR1Cascade(simCasesN, hospRatioTest,  phiH, sigmaH, NWK);
simDeathN = simulateAR1Cascade(simCasesN, deathRatioTest, phiD, sigmaD, NWK);

%% -------------------- 4b. HEALTHCARE-ACCESS SENSITIVITY -------------
% Fit state-level healthcare-access effects on hospitalisation and
% death ratios using training seasons only.
gammaDeath = fitHCAccessGamma(stT, allStates, TRAIN_SEASONS, stateHC, meanHC, 'deaths');
gammaHosp  = fitHCAccessGamma(stT, allStates, TRAIN_SEASONS, stateHC, meanHC, 'hospitalizations');
fprintf('Healthcare-access sensitivity (fit from training data only): death gamma=%.2f, hosp gamma=%.2f\n', ...
    gammaDeath, gammaHosp);

% Load the held-out test season for evaluation only.
actN = natT(natT.season==TEST_SEASON,:);
actN = sortrows(actN,'epi_week_of_season');
actCases = actN.cases; actHosp = actN.hospitalizations; actDeaths = actN.deaths;
weekStartTest = actN.week_start;

% Calculate forecast quantiles.
QcasesN = myQuantile(simCasesN, QLEVELS);
QhospN  = myQuantile(simHospN,  QLEVELS);
QdeathN = myQuantile(simDeathN, QLEVELS);
medCasesN  = QcasesN(QLEVELS==0.5,:)';
medHospN   = QhospN(QLEVELS==0.5,:)';
medDeathN  = QdeathN(QLEVELS==0.5,:)';

%% -------------------- 5. SEASONAL TARGETS (national) -----------------
% Calculate simulated seasonal peak, cumulative burden and attack rate.
[peakWkMed, peakValMed, peakWkDist] = peakStats(simCasesN);
cumCasesDist  = sum(simCasesN,2);
cumHospDist   = sum(simHospN,2);
cumDeathDist  = sum(simDeathN,2);
attackRateDist = cumCasesDist ./ natPop * 100;

% Calculate observed seasonal values.
[~,actPeakIdx] = max(actCases);
actPeakWk  = actPeakIdx;
actPeakVal = actCases(actPeakIdx);
actCumCases = sum(actCases); actCumHosp = sum(actHosp); actCumDeaths = sum(actDeaths);
actAttackRate = actCumCases/natPop*100;
actSeverity = actCumCases/natPop*1e5;

fprintf('\n--- National seasonal targets, %s (forecast vs actual) ---\n', TEST_SEASON);
fprintf('(Actual severity %.0f cum.cases/100k shown below for reference only.)\n', actSeverity);
fprintf('Peak week          : forecast %d (90%% PI %d-%d)  | actual %d\n', ...
    round(median(peakWkDist)), myQuantile(peakWkDist,0.05), myQuantile(peakWkDist,0.95), actPeakWk);
fprintf('Peak weekly cases   : forecast %.0f (90%% PI %.0f-%.0f) | actual %.0f\n', ...
    peakValMed, myQuantile(max(simCasesN,[],2),0.05), myQuantile(max(simCasesN,[],2),0.95), actPeakVal);
fprintf('Cumulative cases    : forecast %.0f (90%% PI %.0f-%.0f) | actual %.0f\n', ...
    median(cumCasesDist), myQuantile(cumCasesDist,0.05), myQuantile(cumCasesDist,0.95), actCumCases);
fprintf('Cumulative hosp.    : forecast %.0f | actual %.0f\n', median(cumHospDist), actCumHosp);
fprintf('Cumulative deaths   : forecast %.0f | actual %.0f\n', median(cumDeathDist), actCumDeaths);
fprintf('Attack rate (%%)     : forecast %.2f (90%% PI %.2f-%.2f) | actual %.2f\n', ...
    median(attackRateDist), myQuantile(attackRateDist,0.05), myQuantile(attackRateDist,0.95), actAttackRate);
fprintf('\n--- National cumulative relative error ---\n');
fprintf('Cumulative-cases relative error   : %+.1f%%\n', (median(cumCasesDist)-actCumCases)/actCumCases*100);
fprintf('Cumulative-hosp. relative error   : %+.1f%%\n', (median(cumHospDist)-actCumHosp)/actCumHosp*100);
fprintf('Cumulative-deaths relative error  : %+.1f%%\n', (median(cumDeathDist)-actCumDeaths)/actCumDeaths*100);

%% -------------------- 6. OPTIONAL PROBABILISTIC TARGETS --------------
% Calculate weekly increase/decrease probabilities and peak timing probability.
pIncrease = mean(diff(simCasesN,1,2) > 0, 1)';   % NWK-1 x 1
pDecrease = 1 - pIncrease;
peakWindow = 2;
pPeakNearMedian = mean(abs(peakWkDist - median(peakWkDist)) <= peakWindow);
fprintf('P(peak within +/-%d wks of median peak week): %.2f\n', peakWindow, pPeakNearMedian);

peakWeekProb = zeros(NWK,1);           % P(peak occurs in week w), all w
for w = 1:NWK
    peakWeekProb(w) = mean(peakWkDist==w);
end
writetable(table((1:NWK)', peakWeekProb, 'VariableNames',{'epi_week','P_peak_this_week'}), ...
    fullfile(OUT_DIR,'peak_week_probability.csv'));

%% -------------------- 7. NATIONAL EVALUATION METRICS -----------------
% Calculate point, probabilistic and interval forecast metrics.
evalN = scoreForecast(actCases, simCasesN, QLEVELS, QcasesN);
evalH = scoreForecast(actHosp,  simHospN,  QLEVELS, QhospN);
evalD = scoreForecast(actDeaths,simDeathN, QLEVELS, QdeathN);

fprintf('\n--- National weekly forecast accuracy (2025/2026, out-of-sample) ---\n');
fprintf('%-18s %10s %10s %10s %10s %10s %8s %8s\n','Target','MAE','RMSE','WIS','CRPS','QLoss','Cov50','Cov90');
printEvalRow('Cases',   evalN);
printEvalRow('Hosp.',   evalH);
printEvalRow('Deaths',  evalD);

%% -------------------- 8. STATE-LEVEL MODEL --------------------------
% Fit a harmonic model for each state.
% Compute population-weighted zone coefficients.
% Shrink state coefficients toward the corresponding zone coefficients.
% Use zone residuals for block-bootstrap simulation.
zoneList = unique(stateZone);
zoneBeta = containers.Map('KeyType','char','ValueType','any');

% Fit each state model and retain residuals separately by season.
rawBeta = cell(nStates,1); rawResidSeasons = cell(nStates,1);
for i=1:nStates
    Yall = []; Xall = [];
    for si = 1:numel(TRAIN_SEASONS)
        subS = stT(stT.state==allStates(i) & stT.season==TRAIN_SEASONS{si},:);
        subS = sortrows(subS,'epi_week_of_season');
        y = log1p(subS.cases_per_100k);
        Yall = [Yall; y]; %#ok<AGROW>
        Xall = [Xall; harmonicDesign(subS.epi_week_of_season, NWK, NHARM)]; %#ok<AGROW>
    end
    b = Xall \ Yall;
    rawBeta{i} = b;
    resAll = Yall - Xall*b;
    rawResidSeasons{i} = {resAll(1:NWK), resAll(NWK+1:2*NWK)};
end

% Calculate population-weighted zone coefficients.
for z = zoneList'
    idx = find(stateZone==z);
    w = statePop(idx)/sum(statePop(idx));
    B = [rawBeta{idx}]';
    zoneBeta(char(z)) = (w' * B)';
end

% Shrink state coefficients toward the zone and simulate state forecasts.
weekly100kMed = zeros(nStates,NWK); weekly100kLo50=weekly100kMed; weekly100kHi50=weekly100kMed;
weekly100kLo90=weekly100kMed; weekly100kHi90=weekly100kMed;
weeklyCasesMed = zeros(nStates,NWK);
weeklyHospMed  = zeros(nStates,NWK);
attackRateFcState = zeros(nStates,1); attackRateActState = zeros(nStates,1);
peakWeekFcState  = zeros(nStates,1); peakWeekActState = zeros(nStates,1);
cumDeathsFcState = zeros(nStates,1); deathRate100kFcState = zeros(nStates,1); deathRate100kActState = zeros(nStates,1);
cumHospFcState   = zeros(nStates,1); hospRate100kFcState  = zeros(nStates,1); hospRate100kActState  = zeros(nStates,1);
cumCasesActState = zeros(nStates,1); cumDeathsActState = zeros(nStates,1); cumHospActState = zeros(nStates,1);
maeState = zeros(nStates,1); rmseState = zeros(nStates,1);

for i=1:nStates
    % Shrink state coefficients toward the population-weighted zone estimate.
    w_s = statePop(i) / (statePop(i) + K_SHRINK);
    bZone = zoneBeta(char(stateZone(i)));
    bShrunk = w_s*rawBeta{i} + (1-w_s)*bZone;

    % Build the zone residual pool using season-separated state curves.
    zIdx = find(stateZone==stateZone(i));
    zoneChunks = {};
    for j = zIdx'
        zoneChunks = [zoneChunks, rawResidSeasons{j}]; %#ok<AGROW>
    end

    % Simulate state incidence per 100,000 using block bootstrap.
    muTest_s = harmonicDesign(tTest,NWK,NHARM) * bShrunk;
    sim100k = simulateBlockBootstrap(muTest_s, zoneChunks, NSIM_STATE, NWK, BLOCK_LEN, JITTER_FRAC);

    q = myQuantile(sim100k, [0.05 0.25 0.5 0.75 0.95]);
    weekly100kLo90(i,:) = q(1,:); weekly100kLo50(i,:) = q(2,:);
    weekly100kMed(i,:)  = q(3,:); weekly100kHi50(i,:)  = q(4,:); weekly100kHi90(i,:) = q(5,:);

    weeklyCasesMed(i,:) = weekly100kMed(i,:) * statePop(i) / 1e5;

    % Apply healthcare-access adjustments to outcome ratios.
    hospMult_s  = min(max(exp(gammaHosp *(stateHC(i)-meanHC)), 0.3), 3.0);
    deathMult_s = min(max(exp(gammaDeath*(stateHC(i)-meanHC)), 0.3), 3.0);
    hospRatio_s = min(hospRatioTest' * hospMult_s, 0.35);
    weeklyHospMed(i,:) = weeklyCasesMed(i,:) .* hospRatio_s;

    % Apply the adjusted death ratio.
    deathRatio_s = deathRatioTest' * deathMult_s;
    weeklyDeathsMed_i = weeklyCasesMed(i,:) .* deathRatio_s;
    cumDeathsFcState(i) = sum(weeklyDeathsMed_i);
    deathRate100kFcState(i) = cumDeathsFcState(i)/statePop(i)*1e5;
    cumHospFcState(i) = sum(weeklyHospMed(i,:));
    hospRate100kFcState(i) = cumHospFcState(i)/statePop(i)*1e5;

    % Calculate state seasonal targets.
    attackRateFcState(i) = sum(weeklyCasesMed(i,:))/statePop(i)*100;
    [~,pk] = max(weekly100kMed(i,:)); peakWeekFcState(i) = pk;

    % Calculate state-level observed values and errors.
    subA = stT(stT.state==allStates(i) & stT.season==TEST_SEASON,:);
    subA = sortrows(subA,'epi_week_of_season');
    attackRateActState(i) = sum(subA.cases)/statePop(i)*100;
    deathRate100kActState(i) = sum(subA.deaths)/statePop(i)*1e5;
    hospRate100kActState(i) = sum(subA.hospitalizations)/statePop(i)*1e5;
    cumCasesActState(i) = sum(subA.cases); cumDeathsActState(i) = sum(subA.deaths); cumHospActState(i) = sum(subA.hospitalizations);
    [~,pkA] = max(subA.cases_per_100k); peakWeekActState(i) = pkA;
    maeState(i)  = mean(abs(subA.cases - weeklyCasesMed(i,:)'));
    rmseState(i) = sqrt(mean((subA.cases - weeklyCasesMed(i,:)').^2));
end

fprintf('\n--- State-level accuracy summary (averaged over %d states/FCT) ---\n',nStates);
fprintf('Mean weekly-cases MAE : %.1f\n', mean(maeState));
fprintf('Mean weekly-cases RMSE: %.1f\n', mean(rmseState));
fprintf('Mean |attack-rate error| (pct pts): %.2f\n', mean(abs(attackRateFcState-attackRateActState)));
fprintf('Mean |peak-week error| (weeks)    : %.2f\n', mean(abs(peakWeekFcState-peakWeekActState)));
fprintf('Mean |cumulative-deaths relative error| (state-level): %.1f%%\n', ...
    mean(abs(cumDeathsFcState-cumDeathsActState)./max(cumDeathsActState,1))*100);
fprintf('Mean |cumulative-hospitalisations relative error| (state-level): %.1f%%\n', ...
    mean(abs(cumHospFcState-cumHospActState)./max(cumHospActState,1))*100);

%% -------------------- 9. NORTH vs SOUTH AGGREGATION -------------------
% Aggregate state forecasts into North and South groups.
northIdx = ismember(stateZone,NORTH_ZONES);
southIdx = ismember(stateZone,SOUTH_ZONES);
northFcCases = sum(weeklyCasesMed(northIdx,:),1);
southFcCases = sum(weeklyCasesMed(southIdx,:),1);

% Aggregate held-out state observations by zone and week.
actByZoneWeek = zeros(numel(zoneList),NWK);
for zi=1:numel(zoneList)
    memb = stateZone==zoneList(zi);
    sub = stT(ismember(stT.state,allStates(memb)) & stT.season==TEST_SEASON,:);
    wk = sub.epi_week_of_season; cs = sub.cases;
    v = zeros(NWK,1);
    for w = 1:NWK
        v(w) = sum(cs(wk==w));
    end
    actByZoneWeek(zi,:) = v';
end
northAct = sum(actByZoneWeek(ismember(zoneList,NORTH_ZONES),:),1);
southAct = sum(actByZoneWeek(ismember(zoneList,SOUTH_ZONES),:),1);

%% -------------------- 10. WRITE OUTPUT TABLES --------------------------
% Write national, state and evaluation tables.
probIncreaseVec = [pIncrease; NaN];
probDecreaseVec = [pDecrease; NaN];
natOut = table(weekStartTest, (1:NWK)', medCasesN, QcasesN(QLEVELS==0.25,:)', QcasesN(QLEVELS==0.75,:)', ...
    QcasesN(QLEVELS==0.05,:)', QcasesN(QLEVELS==0.95,:)', medHospN, medDeathN, actCases, actHosp, actDeaths, ...
    probIncreaseVec, probDecreaseVec, ...
    'VariableNames',{'week_start','epi_week','forecast_cases_median','cases_pi50_lo','cases_pi50_hi', ...
    'cases_pi90_lo','cases_pi90_hi','forecast_hosp_median','forecast_deaths_median','actual_cases','actual_hosp','actual_deaths', ...
    'P_cases_increase_nextweek','P_cases_decrease_nextweek'});
writetable(natOut, fullfile(OUT_DIR,'national_weekly_forecast.csv'));

stateOut = table(allStates, stateZone, statePop, attackRateFcState, attackRateActState, ...
    peakWeekFcState, peakWeekActState, deathRate100kFcState, deathRate100kActState, ...
    hospRate100kFcState, hospRate100kActState, maeState, rmseState, ...
    'VariableNames',{'state','zone','population','attack_rate_pct_forecast','attack_rate_pct_actual', ...
    'peak_week_forecast','peak_week_actual','death_rate_per100k_forecast','death_rate_per100k_actual', ...
    'hosp_rate_per100k_forecast','hosp_rate_per100k_actual','weekly_MAE','weekly_RMSE'});
writetable(stateOut, fullfile(OUT_DIR,'state_seasonal_forecast.csv'));

cumRelErrN = (median(cumCasesDist)-actCumCases)/actCumCases*100;
cumRelErrH = (median(cumHospDist)-actCumHosp)/actCumHosp*100;
cumRelErrD = (median(cumDeathDist)-actCumDeaths)/actCumDeaths*100;
metricsOut = table({'Cases';'Hospitalizations';'Deaths'}, ...
    [evalN.MAE;evalH.MAE;evalD.MAE], [evalN.RMSE;evalH.RMSE;evalD.RMSE], ...
    [evalN.WIS;evalH.WIS;evalD.WIS], [evalN.CRPS;evalH.CRPS;evalD.CRPS], ...
    [evalN.qloss;evalH.qloss;evalD.qloss], [evalN.cov50;evalH.cov50;evalD.cov50], [evalN.cov90;evalH.cov90;evalD.cov90], ...
    [cumRelErrN;cumRelErrH;cumRelErrD], ...
    'VariableNames',{'target','MAE','RMSE','WIS','CRPS','QuantileLoss','Coverage50','Coverage90','CumulativeRelativeError_pct'});
writetable(metricsOut, fullfile(OUT_DIR,'national_evaluation_metrics.csv'));

fprintf('\nForecast tables written to: %s\n', OUT_DIR);

%% =====================================================================
%  11. VISUALISATIONS
% =====================================================================
weekAxis = 1:NWK;

% --- Fig 1: National weekly forecasts and prediction intervals --------
f1 = figure('Position',[50 50 1100 850],'Color','w');
subplot(3,1,1); plotBandAndActual(weekAxis, QcasesN, QLEVELS, actCases, 'Weekly cases');
title('National weekly influenza CASES: forecast vs actual — 2025/2026'); ylabel('Cases');
subplot(3,1,2); plotBandAndActual(weekAxis, QhospN, QLEVELS, actHosp, 'Weekly hospitalisations');
title('National weekly HOSPITALISATIONS: forecast vs actual'); ylabel('Hospitalisations');
subplot(3,1,3); plotBandAndActual(weekAxis, QdeathN, QLEVELS, actDeaths, 'Weekly deaths');
title('National weekly DEATHS: forecast vs actual'); ylabel('Deaths'); xlabel('Epidemic week of season');
saveas(f1, fullfile(OUT_DIR,'fig1_national_weekly_cases_hosp_deaths.png'));

% --- Fig 2: Regional epidemic curves ---------------------------
f2 = figure('Position',[50 50 1100 700],'Color','w');
for zi=1:numel(zoneList)
    subplot(2,3,zi);
    memb = stateZone==zoneList(zi);
    fc = sum(weeklyCasesMed(memb,:),1);
    plot(weekAxis, fc,'b-','LineWidth',1.8); hold on;
    plot(weekAxis, actByZoneWeek(zi,:),'k--','LineWidth',1.3);
    title(char(zoneList(zi))); xlabel('Week'); ylabel('Cases'); legend('Forecast','Actual','Location','best'); grid on;
end
sgtitle('Regional epidemic curves by geopolitical zone — 2025/2026');
saveas(f2, fullfile(OUT_DIR,'fig2_regional_epidemic_curves.png'));

% --- Fig 3: North vs South comparison -----------------------------
f3 = figure('Position',[50 50 900 500],'Color','w');
plot(weekAxis, northFcCases,'r-','LineWidth',2); hold on;
plot(weekAxis, northAct,'r--','LineWidth',1.3);
plot(weekAxis, southFcCases,'b-','LineWidth',2);
plot(weekAxis, southAct,'b--','LineWidth',1.3);
legend('North forecast','North actual','South forecast','South actual','Location','best');
title('North vs South Nigeria — influenza wave comparison, 2025/2026');
xlabel('Epidemic week of season'); ylabel('Weekly cases'); grid on;
saveas(f3, fullfile(OUT_DIR,'fig3_north_vs_south.png'));

% --- Fig 4: State-level actual incidence heatmap ------------------
ordT = table(stateZone, allStates, (1:nStates)', 'VariableNames',{'zoneKey','stateKey','origIdx'});
ordT = sortrows(ordT, {'zoneKey','stateKey'});
ord  = ordT.origIdx;
heatMat = zeros(nStates,NWK);
for i=1:nStates
    subA = stT(stT.state==allStates(i) & stT.season==TEST_SEASON,:);
    subA = sortrows(subA,'epi_week_of_season');
    heatMat(i,:) = subA.cases_per_100k';
end
f4 = figure('Position',[50 50 1000 900],'Color','w');
imagesc(heatMat(ord,:)); colormap(parula); colorbar;
set(gca,'YTick',1:nStates,'YTickLabel',cellstr(allStates(ord)),'FontSize',7);
xlabel('Epidemic week of season'); title('State-level weekly incidence heatmap (cases per 100k) — 2025/2026 actual');
saveas(f4, fullfile(OUT_DIR,'fig4_state_weekly_heatmap.png'));

% --- Fig 5: State attack-rate comparison -------------------------
[~, sIdx] = sort(attackRateActState,'descend');
f5 = figure('Position',[50 50 900 950],'Color','w');
barh([attackRateActState(sIdx), attackRateFcState(sIdx)]);
set(gca,'YTick',1:nStates,'YTickLabel',cellstr(allStates(sIdx)),'YDir','reverse','FontSize',7);
legend('Actual','Forecast','Location','southeast');
xlabel('Seasonal attack rate (%)'); title('State attack-rate map (ranked) — 2025/2026');
saveas(f5, fullfile(OUT_DIR,'fig5_state_attack_rate_map.png'));

% --- Fig 6: State mortality comparison ----------------------------
[~, sIdx2] = sort(deathRate100kActState,'descend');
f6 = figure('Position',[50 50 900 950],'Color','w');
barh([deathRate100kActState(sIdx2), deathRate100kFcState(sIdx2)]);
set(gca,'YTick',1:nStates,'YTickLabel',cellstr(allStates(sIdx2)),'YDir','reverse','FontSize',7);
legend('Actual','Forecast','Location','southeast');
xlabel('Deaths per 100,000'); title('State mortality map (ranked) — 2025/2026');
saveas(f6, fullfile(OUT_DIR,'fig6_state_mortality_map.png'));

% --- Fig 7: Peak-week comparison ----------------------------------
f7 = figure('Position',[50 50 950 950],'Color','w');
scatter(peakWeekActState, 1:nStates, 45, 'k','filled'); hold on;
scatter(peakWeekFcState, 1:nStates, 45, 'r','filled');
set(gca,'YTick',1:nStates,'YTickLabel',cellstr(allStates),'FontSize',7);
legend('Actual peak week','Forecast peak week','Location','eastoutside');
xlabel('Epidemic week of season'); title('Peak-week summary by state — 2025/2026');
saveas(f7, fullfile(OUT_DIR,'fig7_peak_week_summary.png'));

% --- Fig 8: National forecast uncertainty -------------------------
f8 = figure('Position',[50 50 950 500],'Color','w');
plotBandAndActual(weekAxis, QcasesN, QLEVELS, actCases, 'Weekly cases');
title('National case forecast — 50% and 90% prediction intervals');
xlabel('Epidemic week of season'); ylabel('Weekly cases'); grid on;
saveas(f8, fullfile(OUT_DIR,'fig8_forecast_uncertainty_bands.png'));

fprintf('8 figures written to: %s\n', OUT_DIR);
fprintf('\n=== DONE ===\n');


%% =====================================================================
%                          LOCAL FUNCTIONS
% =====================================================================

function q = myQuantile(X, p)
    % Calculate quantiles using linear interpolation without toolboxes.
    [n, nc] = size(X);
    Xs = sort(X,1);
    nq = numel(p);
    q = zeros(nq, nc);
    for k = 1:nq
        h = p(k)*(n-1) + 1;
        lo = min(max(floor(h),1),n);
        hi = min(max(ceil(h),1),n);
        frac = h - lo;
        q(k,:) = Xs(lo,:) + frac*(Xs(hi,:)-Xs(lo,:));
    end
end

function gamma = fitHCAccessGamma(stT, allStates, trainSeasons, stateHC, meanHC, outcomeCol)
    % Fit the log outcome-to-case ratio against healthcare-access deviation
    % using training-season state totals.
    n = numel(allStates);
    X = zeros(n,1); Y = zeros(n,1);
    for i = 1:n
        num = 0; den = 0;
        for si = 1:numel(trainSeasons)
            sub = stT(stT.state==allStates(i) & stT.season==trainSeasons{si}, :);
            num = num + sum(sub.(outcomeCol));
            den = den + sum(sub.cases);
        end
        ratio = num / max(den,1);
        X(i) = stateHC(i) - meanHC;
        Y(i) = log(max(ratio, 1e-6));
    end
    A = [ones(n,1), X];
    b = A \ Y;
    gamma = b(2);
end

function fp = findFile(dataDir, pattern)
    % Find the first file matching the supplied pattern.
    d = dir(fullfile(dataDir,pattern));
    if isempty(d)
        error('MAEPIMS:fileNotFound', ...
            'Could not find a file matching "%s" in %s. Edit DATA_DIR at the top of the script.', pattern, dataDir);
    end
    fp = fullfile(d(1).folder, d(1).name);
end

function X = harmonicDesign(t, nWeeks, nHarm)
    % Build intercept, Fourier harmonics and week-1 indicator.
    X = ones(numel(t),1);
    for k = 1:nHarm
        X = [X, sin(2*pi*k*t(:)/nWeeks), cos(2*pi*k*t(:)/nWeeks)]; %#ok<AGROW>
    end
    X = [X, double(t(:)==1)];
end

function Xrow = addSeverityInteraction(Hd, sevValue, nInteract)
    % Add severity and severity interactions with the selected harmonics.
    n = size(Hd,1);
    sevCol = sevValue*ones(n,1);
    if nInteract > 0
        interactCols = Hd(:,2:1+2*nInteract) .* sevCol;
        Xrow = [Hd, sevCol, interactCols];
    else
        Xrow = [Hd, sevCol];
    end
end

function [beta, residCell] = fitHarmonic(natT, trainSeasons, NWK, nHarm, nInteract, sevZ)
    % Fit the national harmonic model and retain season-separated residuals.
    Y = []; Xd = []; residCell = cell(numel(trainSeasons),1);
    for si = 1:numel(trainSeasons)
        s = trainSeasons{si};
        sub = natT(natT.season==s,:);
        sub = sortrows(sub,'epi_week_of_season');
        t = sub.epi_week_of_season;
        y = log1p(sub.cases);
        Hd = harmonicDesign(t, NWK, nHarm);
        Xrow = addSeverityInteraction(Hd, sevZ(s), nInteract);
        Y = [Y; y]; Xd = [Xd; Xrow]; %#ok<AGROW>
    end
    beta = Xd \ Y;
    resAll = Y - Xd*beta;
    nps = NWK;
    for si = 1:numel(trainSeasons)
        residCell{si} = resAll((si-1)*nps+1 : si*nps);
    end
end

function [phi, sigma] = fitAR1(residCell, NWK)
    % Fit a pooled stationary AR(1) model to season-separated residuals.
    Y = []; X = [];
    for i = 1:numel(residCell)
        r = residCell{i}(:); n = numel(r);
        for tt = 2:n
            Y(end+1,1) = r(tt); %#ok<AGROW>
            X(end+1,1) = r(tt-1); %#ok<AGROW>
        end
    end
    phi = (X'*Y) / (X'*X);
    innov = Y - X*phi;
    sigma = std(innov);
end

function sim = simulateAR1Severity(HdTest, beta, nInteract, sevLo, sevHi, sevJitterSD, phi, sigma, nSim, NWK)
    % Simulate national case trajectories using sampled severity and AR(1) noise.
    sim = zeros(nSim, NWK);
    for s = 1:nSim
        wSev = rand();
        sevSim = wSev*sevLo + (1-wSev)*sevHi + sevJitterSD*randn();
        Xtest = addSeverityInteraction(HdTest, sevSim, nInteract);
        mu = Xtest * beta;
        e = zeros(NWK,1);
        e(1) = (sigma/sqrt(max(1-phi^2,1e-6))) * randn();
        for t = 2:NWK
            e(t) = phi*e(t-1) + sigma*randn();
        end
        sim(s,:) = max(expm1(mu+e), 0)';
    end
end

function [ratioTest, phi, sigma] = fitRatioHarmonicAR(natT, trainSeasons, NWK, outcomeCol, tTest)
    % Fit a one-harmonic outcome ratio and its pooled AR(1) residual model.
    Y = []; Xd = []; seasonLens = zeros(numel(trainSeasons),1);
    for si = 1:numel(trainSeasons)
        sub = natT(natT.season==trainSeasons{si},:);
        sub = sortrows(sub,'epi_week_of_season');
        r = sub.(outcomeCol) ./ max(sub.cases,1);
        r = min(max(r, 1e-5), 0.5);
        t = sub.epi_week_of_season;
        Hd = harmonicDesign(t, NWK, 1);
        Y = [Y; log(r)]; Xd = [Xd; Hd]; %#ok<AGROW>
        seasonLens(si) = numel(t);
    end
    b = Xd \ Y;
    resAll = Y - Xd*b;
    residCell = cell(numel(trainSeasons),1);
    idx0 = 0;
    for si = 1:numel(trainSeasons)
        residCell{si} = resAll(idx0+1:idx0+seasonLens(si));
        idx0 = idx0 + seasonLens(si);
    end
    [phi, sigma] = fitAR1(residCell, NWK);
    ratioTest = exp(harmonicDesign(tTest, NWK, 1) * b);
end

function sim = simulateAR1Cascade(simCases, ratioCurve, phi, sigma, NWK)
    % Apply AR(1) residual variation to outcome-to-case ratios.
    nSim = size(simCases,1);
    sim = zeros(nSim, NWK);
    for s = 1:nSim
        e = zeros(NWK,1);
        e(1) = (sigma/sqrt(max(1-phi^2,1e-6))) * randn();
        for t = 2:NWK
            e(t) = phi*e(t-1) + sigma*randn();
        end
        ratio_t = ratioCurve(:)' .* exp(e)';
        sim(s,:) = max(simCases(s,:) .* ratio_t, 0);
    end
end

function sim = simulateBlockBootstrap(mu, curvesPool, nSim, NWK, blockLen, jitterFrac)
    % Simulate state/zone trajectories using block-bootstrap residuals
    % with continuous mixing and Gaussian jitter.
    pool = curvesPool; nPool = numel(pool);
    poolMat = zeros(NWK, nPool);
    for k = 1:nPool, poolMat(:,k) = pool{k}; end
    jitterStd = jitterFrac * std(poolMat(:));
    nBlocks = ceil(NWK/blockLen);
    sim = zeros(nSim, NWK);
    for s = 1:nSim
        e = zeros(NWK,1);
        for b = 1:nBlocks
            i0 = (b-1)*blockLen + 1;
            i1 = min(i0+blockLen-1, NWK);
            if nPool >= 2
                idx = randi(nPool,1,2);
            else
                idx = [1 1];
            end
            w = rand();
            e(i0:i1) = w*poolMat(i0:i1,idx(1)) + (1-w)*poolMat(i0:i1,idx(2));
        end
        e = e + jitterStd*randn(NWK,1);
        y = mu + e;
        sim(s,:) = max(expm1(y), 0)';
    end
end

function [peakWkMed, peakValMed, peakWkDist] = peakStats(sim)
    % Extract peak week and peak value from simulated trajectories.
    [maxVals, maxIdx] = max(sim, [], 2);
    peakWkDist = maxIdx;
    peakWkMed = median(maxIdx);
    peakValMed = median(maxVals);
end

function out = scoreForecast(actual, sim, qlevels, Q)
    % Calculate point, quantile, interval and distributional forecast metrics.
    med = Q(qlevels==0.5,:)';
    out.MAE  = mean(abs(actual - med));
    out.RMSE = sqrt(mean((actual - med).^2));

    % Quantile loss.
    ql = 0;
    for qi = 1:numel(qlevels)
        tauv = qlevels(qi); qv = Q(qi,:)';
        diffv = actual - qv;
        ql = ql + mean(max(tauv*diffv, (tauv-1)*diffv));
    end
    out.qloss = ql / numel(qlevels);

    % Weighted interval score using 50% and 90% intervals.
    lo50 = Q(qlevels==0.25,:)'; hi50 = Q(qlevels==0.75,:)';
    lo90 = Q(qlevels==0.05,:)'; hi90 = Q(qlevels==0.95,:)';
    IS50 = (hi50-lo50) + (2/0.5).*(lo50-actual).*(actual<lo50) + (2/0.5).*(actual-hi50).*(actual>hi50);
    IS90 = (hi90-lo90) + (2/0.1).*(lo90-actual).*(actual<lo90) + (2/0.1).*(actual-hi90).*(actual>hi90);
    K = 2; w0 = 0.5; w50 = 0.5/2; w90 = 0.1/2;
    WIS = (1/(K+0.5)) .* (w0*abs(actual-med) + w50*IS50 + w90*IS90);
    out.WIS = mean(WIS);

    % Prediction-interval coverage.
    out.cov50 = mean(actual>=lo50 & actual<=hi50);
    out.cov90 = mean(actual>=lo90 & actual<=hi90);

    % Empirical CRPS.
    NWK = numel(actual); nSim = size(sim,1);
    crpsW = zeros(NWK,1);
    for w = 1:NWK
        x = sort(sim(:,w));
        n = numel(x);
        term1 = mean(abs(x - actual(w)));
        ranks = (1:n)';
        term2 = (2/(n^2)) * sum((2*ranks - n - 1) .* x);
        crpsW(w) = term1 - 0.5*term2;
    end
    out.CRPS = mean(crpsW);
end

function printEvalRow(name, ev)
    % Print one forecast-evaluation row, including quantile loss.
    fprintf('%-18s %10.1f %10.1f %10.2f %10.2f %10.2f %8.2f %8.2f\n', ...
        name, ev.MAE, ev.RMSE, ev.WIS, ev.CRPS, ev.qloss, ev.cov50, ev.cov90);
end

function plotBandAndActual(weekAxis, Q, qlevels, actual, ~)
    % Plot 50% and 90% prediction intervals, median forecast and actuals.
    lo90 = Q(qlevels==0.05,:); hi90 = Q(qlevels==0.95,:);
    lo50 = Q(qlevels==0.25,:); hi50 = Q(qlevels==0.75,:);
    med  = Q(qlevels==0.5,:);
    fill([weekAxis, fliplr(weekAxis)], [lo90, fliplr(hi90)], [0.75 0.85 1], 'EdgeColor','none','FaceAlpha',0.6); hold on;
    fill([weekAxis, fliplr(weekAxis)], [lo50, fliplr(hi50)], [0.35 0.6 0.95], 'EdgeColor','none','FaceAlpha',0.7);
    plot(weekAxis, med, 'b-', 'LineWidth', 2);
    plot(weekAxis, actual, 'k--', 'LineWidth', 1.5);
    legend('90% PI','50% PI','Median forecast','Actual','Location','best');
    grid on;
end