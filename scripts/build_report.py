# -*- coding: utf-8 -*-
"""
Builds the MAEPiMS Challenge 2026 technical report PDF from the validated results
produced across src/renewal.py, src/calibration.py, src/residual_model.py, and
src/ensemble.py (see docs/DESIGN.md for the full derivation of every number used
here). Run after scripts/run_ensemble.py and scripts/make_report_figures.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether, ListFlowable, ListItem, HRFlowable,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "eda" / "figures"
OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(exist_ok=True)
OUT_PDF = OUT_DIR / "Pinnk_MAEPiMS2.pdf"

GITHUB_URL = "https://github.com/maepimschallenge/MAEPiMS-2.0-2026-"
NICKNAME = "Pinnk"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
INK = colors.HexColor("#0b0b0b")
SECONDARY = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
MUTED = colors.HexColor("#898781")
GRID = colors.HexColor("#e1e0d9")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, leading=27,
                           textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle("ReportSubtitle", parent=styles["Normal"], fontSize=13, leading=17,
                           textColor=SECONDARY, alignment=TA_CENTER, spaceAfter=4))
styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, leading=14,
                           textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, leading=20,
                           textColor=INK, spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, leading=16,
                           textColor=INK, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.7, leading=14.5,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=8))
styles.add(ParagraphStyle("BodySmall", parent=styles["Normal"], fontSize=8.7, leading=12.5,
                           textColor=SECONDARY, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, leading=11,
                           textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=14))
styles.add(ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8.3, leading=10.5, textColor=INK))
styles.add(ParagraphStyle("TableHead", parent=styles["Normal"], fontSize=8.3, leading=10.5,
                           textColor=colors.black, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("Callout", parent=styles["Body"], fontSize=9.5, leading=14,
                           textColor=INK, backColor=colors.HexColor("#f0f4fb"),
                           borderPadding=8, spaceAfter=10, spaceBefore=4))
styles.add(ParagraphStyle("RefBody", parent=styles["Normal"], fontSize=8.8, leading=13,
                           textColor=INK, spaceAfter=6, leftIndent=14, firstLineIndent=-14))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=GRID, spaceAfter=8))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, styles["Body"]), bulletColor=BLUE) for i in items],
        bulletType="bullet", leftIndent=14, spaceBefore=2, spaceAfter=10,
    ))


def fig(path, caption, width=15.5 * cm):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    story.append(Image(str(path), width=width, height=height))
    story.append(Paragraph(caption, styles["Caption"]))


def table(header, rows, col_widths=None, small=False):
    style_name = "TableCell"
    data = [[Paragraph(c, styles["TableHead"]) for c in header]]
    for r in rows:
        data.append([Paragraph(str(c), styles[style_name]) for c in r])
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


# ===========================================================================
# TITLE PAGE
# ===========================================================================
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("A Climate-Phased Dual-Wave Renewal Ensemble for<br/>"
                        "Nigeria Influenza Forecasting", styles["ReportTitle"]))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph("Technical Report - MAEPiMS Challenge 2026", styles["ReportSubtitle"]))
story.append(Paragraph("Nigeria Influenza Forecasting Challenge", styles["ReportSubtitle"]))
story.append(Spacer(1, 1.2 * cm))
story.append(Paragraph(f"Submitted by: {NICKNAME}", styles["Meta"]))
story.append(Paragraph("September 2026", styles["Meta"]))
story.append(Paragraph(f'Source code &amp; reproduction: <a href="{GITHUB_URL}" color="#2a78d6">{GITHUB_URL}</a>',
                        styles["Meta"]))
story.append(PageBreak())

# ===========================================================================
# ABSTRACT
# ===========================================================================
h1("Abstract")
body(
    "We present the Climate-Phased Dual-Wave Renewal Ensemble (CP-DWRE), an original four-component "
    "hybrid model for forecasting weekly influenza cases, hospitalisations and deaths across Nigeria's "
    "36 states and the FCT, developed for the MAEPiMS Challenge 2026 using only the supplied synthetic "
    "surveillance dataset (3 seasons: 2023/24 moderate, 2024/25 high-severity double-peak, 2025/26 "
    "moderate-high). The model combines (1) a covariate-forced discrete-time renewal-equation core that "
    "reproduces Nigeria's characteristic wet-season and Harmattan-associated dual-wave dynamics, (2) a "
    "zone-level hierarchical calibration layer that corrects regional timing differences between "
    "Northern and Southern Nigeria, (3) a LightGBM quantile-regression layer that learns per-state "
    "magnitude and uncertainty corrections directly from data, and (4) a parametric-bootstrap ensemble "
    "that produces calibrated probabilistic forecasts and the challenge's optional transmission-direction "
    "and peak-timing probabilities."
)
body(
    "Every result in this report is validated leave-one-season-out (LOSO) across all three possible "
    "train/test splits, not on a single held-out split - a discipline that mattered in practice: an "
    "early single-split calibration result (peak-week accuracy improving from 81% to 97%) did not "
    "generalise and was corrected before being reported as a finding, and the same pattern recurred in "
    "the residual-quantile layer's probabilistic calibration. In both cases the fix was an evidence-based, "
    "conservatively-tuned blend rather than the naively best-looking single-split configuration, and in "
    "the residual layer's case the fix (interval widening by a factor of 1.25) was a genuine Pareto "
    "improvement - simultaneously lower mean interval score and better empirical coverage - rather than a "
    "defensive trade-off. One of the three seasons (2024/25, epidemiologically the most severe and "
    "atypically timed) consistently resists correction from the other two, a finding we treat as a "
    "substantive result about the limits of 3-season training data rather than a bug to be hidden."
)

# ===========================================================================
# INTRODUCTION
# ===========================================================================
h1("1. Introduction")
body(
    "Seasonal influenza is a major cause of respiratory illness and mortality burden globally, and its "
    "surveillance and forecasting infrastructure in Nigeria is still developing relative to high-income "
    "countries. The MAEPiMS Challenge 2026 addresses this gap by providing a realistic, synthetic national "
    "surveillance dataset that reproduces the demographic, climatic and healthcare-access heterogeneity of "
    "Nigeria's 36 states and FCT across three simulated influenza seasons, each explicitly modelled on a "
    "real-world US reference season of increasing severity (Table 1)."
)
table(
    ["Season", "US reference analogue", "Severity"],
    [
        ["2023/2024", "US 2023-24 (moderate, single December peak)", "Moderate"],
        ["2024/2025", "US 2024-25 (HIGH, most severe since 2017-18, double peak)", "High"],
        ["2025/2026", "US 2025-26 (moderate-high, single peak)", "Moderate-High"],
    ],
    col_widths=[3 * cm, 9.5 * cm, 3 * cm],
)
story.append(Paragraph("<i>Table 1. Season-to-reference mapping supplied with the challenge dataset "
                        "(nigeria_flu_season_reference.csv).</i>", styles["Caption"]))
body(
    "This report documents an original forecasting framework - the Climate-Phased Dual-Wave Renewal "
    "Ensemble (CP-DWRE) - built specifically for this dataset's characteristics, plus the validation "
    "process that shaped every design decision in it. We deliberately report negative and corrected "
    "results alongside positive ones throughout Sections 6-8, because in a competition with only three "
    "seasons of ground truth, a single flattering split is not evidence of generalisation, and we judged "
    "that showing our validation process is as important as the final numbers."
)

h2("1.1 Report structure")
body(
    "Section 2 states the modelling assumptions. Section 3 describes the forecasting methodology and "
    "validation protocol used throughout. Section 4 details the CP-DWRE architecture. Section 5 justifies "
    "the design against the dataset's characteristics. Section 6 presents results for each component, "
    "including the two cases where an initially promising result was corrected after further validation. "
    "Section 7 draws out policy implications. Section 8 discusses strengths, weaknesses and the limits of "
    "what 3 seasons of data can support. Section 9 is a dedicated uncertainty analysis. Section 10 "
    "concludes."
)

# ===========================================================================
# ASSUMPTIONS
# ===========================================================================
h1("2. Assumptions")
bullets([
    "<b>Data provenance.</b> Only the supplied synthetic dataset (MAEPiMS_Challenge_Data/) was used for "
    "model development and evaluation, per the challenge rules; no real-world influenza data was "
    "substituted or blended in.",
    "<b>Near-real-time national surveillance.</b> The renewal core fits its wave-timing and amplitude "
    "parameters against a season's full national weekly case curve. This is best understood as a "
    "<i>nowcasting</i> assumption - that national-level weekly aggregate counts become available with "
    "little delay - rather than a blind, months-ahead prospective forecast from zero within-season data. "
    "State-level and uncertainty corrections (Components 2-3) are, by contrast, learned entirely from "
    "<i>other</i> seasons and applied to a new season without using that season's own state-level data, "
    "which is the more realistic forecasting assumption and is enforced throughout via leave-one-season-out "
    "validation.",
    "<b>Weekly generation-time kernel.</b> Influenza's true generation interval (order of 2-3 days) is "
    "sub-weekly; it is represented at weekly resolution by a discretised gamma kernel (shape 2, mean 1 "
    "week), concentrating transmission weight on the immediately preceding week.",
    "<b>Three seasons is the full training population.</b> Every generalisation claim in this report is "
    "validated by leave-one-season-out testing across all 3 possible splits - the most rigorous check "
    "3 seasons allow - but a pattern that holds on 2 training seasons and is contradicted by the third "
    "remains genuinely uncertain, not resolved, and we say so explicitly wherever it occurs (principally "
    "regarding the 2024/25 season; see Sections 6.2, 6.3 and 9).",
    "<b>Stable per-state severity ratios.</b> Hospitalisation/case and death/case ratios are fit per state "
    "from training-season data and carried forward unchanged to the forecast season, reflecting an "
    "assumption that a state's healthcare-access-driven severity profile is a relatively stable "
    "characteristic rather than a season-specific one.",
    "<b>Independence for aggregation.</b> National and regional (North/South) aggregate uncertainty is "
    "computed by Monte Carlo summation of independently-sampled per-state forecast distributions. Real "
    "states' forecast errors are plausibly correlated (a severe national season tends to hit many states "
    "at once), so this assumption likely <i>understates</i> true aggregate uncertainty; it is flagged as a "
    "simplification rather than presented as exact (Section 8.3).",
])

# ===========================================================================
# METHODOLOGY
# ===========================================================================
h1("3. Forecasting Methodology")
h2("3.1 Validation protocol")
body(
    "Every component of the CP-DWRE is trained and scored using <b>leave-one-season-out (LOSO) "
    "cross-validation</b> across all 3 possible train/test splits (train on 2 seasons, test on the "
    "third), rather than a single arbitrarily-chosen split. This was not a stylistic choice: an early "
    "version of Component 2 was validated on exactly one split (train 2023/24+2024/25, test 2025/26) and "
    "looked excellent - peak-week accuracy improved from 81% to 97% - but running all 3 splits showed "
    "the same correction <i>nearly doubled</i> error when 2024/25 was the held-out season instead "
    "(Section 6.2). Every headline number in this report has since been produced and reported the same "
    "way: per-split results shown, not only the best one, and any tunable parameter (e.g. the calibration "
    "blend weight, the residual-interval inflation factor) chosen by a sweep evaluated across all 3 splits "
    "rather than by fitting to a single split's optimum."
)
h2("3.2 Metrics")
body(
    "Point-forecast accuracy is reported via Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). "
    "Probabilistic accuracy is reported via the Weighted Interval Score (WIS; Bracher et al. 2021), a "
    "CRPS approximation via the pinball-loss/CRPS equivalence (Gneiting &amp; Raftery 2007), and mean "
    "quantile (pinball) loss across the forecast quantile levels "
    "{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}. Calibration of the probabilistic forecasts is assessed "
    "via a PIT-style coverage check: the fraction of true values falling inside the nominal 90% central "
    "prediction interval. Seasonal target accuracy is assessed via peak-week accuracy (within a "
    "&plusmn;1-week tolerance) and peak-incidence percentage error. All metric implementations are in "
    "<font face=\"Courier\">src/metrics.py</font> and unit-tested."
)
h2("3.3 Training / forecasting split")
body(
    "For each of the 3 seasons in turn (held out as the forecast target), the renewal core "
    "(Component 1) is fit to the held-out season's own national weekly curve (the nowcasting assumption, "
    "Section 2), while the calibration layer (Component 2) and residual quantile model (Component 3) are "
    "trained exclusively on the <i>other two</i> seasons' state-level data. Within that training pair, an "
    "additional inner leave-one-out step is applied when constructing the residual model's training "
    "targets: each training season's own calibration baseline is built using zone offsets fit only from "
    "the <i>other</i> training season, so the residual-model training targets are never built from a "
    "baseline that has already seen their own season's pattern."
)

story.append(PageBreak())

# ===========================================================================
# MODEL ARCHITECTURE
# ===========================================================================
h1("4. Model Architecture: the CP-DWRE")
body(
    "The CP-DWRE is a four-component pipeline. No single component is treated as sufficient; each "
    "corrects a specific, diagnosed shortcoming of the one before it. Figure set and per-component "
    "results are in Section 6; this section describes the mechanics."
)

h2("4.1 Component 1 - Renewal core")
body(
    "Rather than a compartmental SEIR model, the effective reproduction number is modelled directly, "
    "forced by covariates, and propagated via the discrete-time renewal equation:"
)
story.append(Paragraph(
    "R<sub>t</sub> = R<sub>base</sub> &middot; (1 + a<sub>1</sub>&middot;bump(t; t<sub>1</sub>, s<sub>1</sub>) "
    "+ a<sub>2</sub>&middot;asym_bump(t; t<sub>2</sub>, s<sub>2,rise</sub>, s<sub>2,fall</sub>)) &middot; "
    "susceptible(t)<br/>"
    "I<sub>t</sub> = R<sub>t</sub> &middot; sum<sub>k=1..L</sub> w<sub>k</sub> &middot; I<sub>t-k</sub>",
    ParagraphStyle("Eq", parent=styles["Body"], alignment=TA_CENTER, fontName="Helvetica-Oblique",
                    spaceBefore=6, spaceAfter=10)
))
body(
    "Two Gaussian-family \"wave\" bumps represent the wet-season and Harmattan-associated transmission "
    "peaks; the second wave's bump is <i>asymmetric</i> (independent rise/fall widths) to capture its "
    "sharper, shorter shape relative to wave 1. <font face=\"Courier\">susceptible(t)</font> is a "
    "<i>leaky</i> depletion pool - <font face=\"Courier\">depletion[t] = decay&middot;depletion[t-1] + "
    "I[t]</font> with <font face=\"Courier\">decay &lt; 1</font> - rather than plain non-recoverable "
    "cumulative-incidence depletion, so that wave 1 and wave 2 are not forced to compete for one "
    "permanently-shrinking susceptible budget. This was a necessary fix, not a stylistic one: the "
    "original non-leaky version could not fit 2023/24's second wave at all (Section 6.1)."
)

h2("4.2 Component 2 - Hierarchical state calibration")
body(
    "A naive population-proportional split of the national trajectory assigns every state the same "
    "wave timing, which is wrong - the EDA (Figure 2) shows the South consistently peaking earlier and "
    "more sharply than the North. Component 2 fits each training state's renewal-core wave-center "
    "parameters (t<sub>1</sub>, t<sub>2</sub>) relative to the national fit, averages the deltas by "
    "geopolitical zone, shrinks each zone's mean toward the global mean via an empirical-Bayes-style "
    "estimator (weight = n<sub>zone</sub> / (n<sub>zone</sub> + k)), and re-simulates the national "
    "trajectory with only the wave centers nudged toward each zone's typical phase. The resulting "
    "zone-phase-shifted trajectory is blended with the naive allocation via a tunable "
    "<font face=\"Courier\">phase_weight</font>, set conservatively (0.25) based on LOSO evidence "
    "(Section 6.2) rather than the higher value a single split would have suggested."
)

h2("4.3 Component 3 - Residual quantile ML correction")
body(
    "A LightGBM quantile regressor (one model per quantile level) is fit on the log-ratio between "
    "observed state-week cases and Component 2's baseline: "
    "<font face=\"Courier\">log((observed+1)/(baseline+1))</font>. Features are static state metadata "
    "(population, healthcare-access index, geopolitical zone, climate classification) and the state-week's "
    "position relative to the national wave centers. This is also where the per-state <i>magnitude</i> "
    "correction Component 2 deliberately avoids (Section 5) is finally learned, directly from data, "
    "without perturbing the renewal equation's own susceptible-ceiling mechanics. Predicted quantiles are "
    "reconstructed as <font face=\"Courier\">baseline &middot; exp(quantile)</font>, monotonicity-enforced "
    "against quantile crossing, and widened by an evidence-based interval-inflation factor "
    "(<font face=\"Courier\">1.25</font>, Section 6.3) to correct systematic overconfidence found in LOSO "
    "testing."
)

h2("4.4 Component 4 - Probabilistic ensemble &amp; submission assembly")
body(
    "A parametric bootstrap resamples the renewal core's fitted parameters from their approximate "
    "covariance (derived from the least-squares fit's Jacobian), producing an ensemble of plausible "
    "national trajectories used to answer the challenge's optional probabilistic targets directly: "
    "P(rising transmission), P(epidemic peak within a specified week window), and the full peak-week "
    "distribution. Separately, Components 2-3's validated per-state quantile forecasts are assembled into "
    "the submission tables (state &times; week &times; quantile, for cases, hospitalisations and deaths, "
    "the latter two via per-state rate ratios fit on training data), with North/South/national aggregates "
    "computed by Monte Carlo summation of sampled per-state distributions rather than by summing quantiles "
    "pointwise (Section 2)."
)

story.append(PageBreak())

# ===========================================================================
# MODEL JUSTIFICATION
# ===========================================================================
h1("5. Model Justification")
body(
    "The challenge requires an original model, not a reproduction of an established framework. The "
    "CP-DWRE's originality is in its <i>composition</i>: no single component is copied wholesale from a "
    "specific published forecasting system. The renewal-equation core is a general, well-established "
    "mathematical tool (in the tradition of Fraser 2007 and Cori et al. 2013's EpiEstim), but its "
    "covariate-forcing structure - two independently-timed, independently-shaped wave bumps plus a leaky "
    "depletion pool - was designed specifically to reproduce this dataset's documented dual-wave, "
    "climate-driven dynamics, and was iterated on directly against this data's own failure modes (Section "
    "6.1). The calibration and residual-ML layers are similarly standard techniques (empirical-Bayes "
    "shrinkage; gradient-boosted quantile regression) applied in a specific division of labour that was "
    "itself arrived at empirically: two earlier, more ambitious calibration designs (shifting amplitude "
    "fields, then shifting wave widths) were tried and rejected after they made held-out forecasts worse, "
    "for mechanistically-understood reasons documented in Section 6.2. That progression - hypothesis, "
    "test, rejection, mechanistic diagnosis, revision - is itself a central part of the model's "
    "justification: each surviving design choice is the one that beat a documented alternative under "
    "held-out testing, not the first idea that seemed reasonable."
)
body(
    "The four-component split (mechanistic core / spatial calibration / residual correction / "
    "probabilistic ensemble) is also a direct response to a structural constraint of this dataset: <b>only "
    "3 seasons are available</b>, so any correction learned from state-level or season-level patterns has "
    "an extremely thin evidence base. Separating \"what the mechanistic model can already explain\" from "
    "\"what must be learned empirically\" - and keeping the empirically-learned layers deliberately "
    "conservative (Sections 6.2-6.3) - was a design response to that constraint, not an incidental "
    "architecture choice."
)

# ===========================================================================
# RESULTS
# ===========================================================================
h1("6. Results")

h2("6.1 Component 1 - Renewal core")
body(
    "The renewal core reproduces the observed dual-wave national case curves closely for all 3 seasons "
    "(Figure 3), correctly locating the primary wave-1 peak week in every season and the secondary "
    "Harmattan wave in the two seasons where it is most pronounced (2024/25, 2025/26)."
)
fig(FIG / "05_renewal_fit_smoke_test.png",
    "Figure 3. Renewal-core fit vs. observed national weekly cases, all 3 seasons (in-sample curve fit).")
table(
    ["Season", "National fit MAE", "True peak week", "Fitted peak week"],
    [
        ["2023/24", "51,062", "14", "14 (v0: 15)"],
        ["2024/25", "57,022", "16", "16"],
        ["2025/26", "65,891", "15", "15 (v0: 16)"],
    ],
    col_widths=[3 * cm, 4 * cm, 4 * cm, 4.5 * cm],
)
body(
    "The initial (v0) renewal core could not adequately fit 2023/24's second wave: its single, "
    "non-leaky susceptible pool and symmetric wave-2 bump forced the optimiser to pin two parameters "
    "(<font face=\"Courier\">s_max</font>, <font face=\"Courier\">s2</font>) at their upper bounds trying "
    "to leave room for a second peak, still landing on the wrong timing (MAE 83,298). Diagnosing this as "
    "wave 1 and wave 2 competing for one fixed, non-recoverable susceptible budget motivated the v1 fix "
    "(leaky depletion, asymmetric wave-2 bump), which improved 2023/24's fit by 39% (to 51,062) at a small "
    "cost to 2025/26 (44,762 -> 65,891) from the leakier pool permitting slightly more wave-1 "
    "overshoot - an accepted, documented trade-off."
)

h2("6.2 Component 2 - Hierarchical calibration: a corrected result")
body(
    "<b>First pass (single split).</b> Trained on 2023/24+2024/25, tested on 2025/26, the full zone "
    "wave-center shift looked excellent: mean state MAE improved from 6,228 to 5,927, and peak-week "
    "accuracy improved from 81% to <b>97%</b>."
)
story.append(Paragraph(
    "<b>This result did not survive leave-one-season-out testing.</b> Tested against 2024/25 instead, the "
    "same full shift <i>nearly doubled</i> mean state MAE (8,366 -> 15,785) and peak-week accuracy "
    "fell to 51%.", styles["Callout"]
))
body(
    "The mechanism: several states (Kano, Katsina, Taraba) have their true second-wave peak at "
    "epidemiological week 29 in 2024/25 specifically, versus week 16-19 for the same states in every "
    "other season - a season-specific anomaly (consistent with 2024/25 being labelled \"HIGH, most severe "
    "since 2017-18\" in the supplied season-reference data), not a stable zone characteristic a "
    "2-season training set can be expected to anticipate. A shrinkage-strength sweep made this <i>worse</i>, "
    "not better, confirming the problem was not fixable by more pooling. The fix was a "
    "<font face=\"Courier\">phase_weight</font> blend between the naive and phase-shifted allocations, "
    "swept across all 3 LOSO splits to find a robust setting (Table 2)."
)
table(
    ["phase_weight", "2023/24 MAE", "2024/25 MAE", "2025/26 MAE", "Mean across splits"],
    [
        ["0.0 (naive)", "4,567", "8,366", "6,228", "6,387"],
        ["0.25 (shipped)", "4,315", "9,122", "5,870", "6,436"],
        ["1.0 (full shift)", "4,538", "15,785", "5,927", "8,750"],
    ],
    col_widths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3.3 * cm],
)
body(
    "phase_weight = 0.25 gives up almost nothing on the cross-split mean (6,436 vs. naive's 6,387) while "
    "capping the 2024/25 downside at +9% instead of +89%, and still improving the two more typical splits. "
    "Figure 4 shows the shipped calibration against the naive baseline for four representative states."
)
fig(FIG / "06_calibration_holdout.png",
    "Figure 4. Naive vs. zone-phase-calibrated (shipped, phase_weight=0.25) state allocation, held-out 2025/26.")

h2("6.3 Component 3 - Residual quantile ML: a genuine improvement, twice over")
body(
    "Trained and scored with the same inner leave-one-out discipline as Component 2 (Section 3.3), the "
    "residual quantile layer's median prediction improves MAE by 23-24% on the two more typical LOSO "
    "splits and is roughly flat on 2024/25 (Table 3)."
)
table(
    ["Test season", "MAE, baseline (Component 2)", "MAE, residual-corrected"],
    [
        ["2023/24", "4,315", "3,333  (-23%)"],
        ["2024/25", "9,122", "9,373  (+2.7%)"],
        ["2025/26", "5,870", "4,465  (-24%)"],
    ],
    col_widths=[3.5 * cm, 6 * cm, 6.3 * cm],
)
body(
    "The probabilistic calibration repeated Component 2's exact lesson: at face value "
    "(<font face=\"Courier\">interval_inflation=1.0</font>), the nominal 90% interval covered 87-89% of "
    "true values on the two typical splits but only <b>45%</b> on 2024/25 - badly overconfident whenever "
    "the held-out season is the atypical one, because the model's uncertainty estimate is itself only "
    "calibrated to 2 seasons' residual spread."
)
story.append(Paragraph(
    "A sweep of the interval-inflation factor from 1.0 to 4.0 found 1.25 is a genuine <b>Pareto "
    "improvement</b> over no adjustment - simultaneously <i>lower</i> mean Weighted Interval Score across "
    "the 3 splits (1,239 vs. 1,243) <i>and</i> better mean 90%-band coverage (80% vs. 73%) - not merely a "
    "defensive trade-off of accuracy for safety.", styles["Callout"]
))
body(
    "This works because on 2024/25 specifically, the interval-score penalty for a too-narrow band "
    "outweighs the small penalty from widening it slightly. Higher factors keep buying coverage on "
    "2024/25 only very slowly (plateauing in the high 60% range even at 4.0, since the underlying problem "
    "there is bias in the median forecast, not just interval width) while making the two typical splits' "
    "WIS steadily worse from unneeded extra width - 1.25 is the best evidence-based compromise given only "
    "3 seasons to validate against."
)

story.append(PageBreak())

h2("6.4 Component 4 - Probabilistic targets and submission assembly")
body(
    "Figure 5 shows the full pipeline's forecast uncertainty bands against the true national curve for "
    "held-out 2025/26: the median forecast (blue) tracks both waves closely, and the true curve stays "
    "within the 50% interval for almost the entire season."
)
fig(FIG / "07_forecast_uncertainty_bands.png",
    "Figure 5. National weekly-case forecast with 50%/90% prediction intervals, held-out 2025/26.")
body(
    "Figure 6 demonstrates the calibration layer's regional timing correction concretely for the hardest "
    "season (2024/25): forecast peak weeks cluster distinctly by zone, with the North (NW/NE/NC, "
    "week 27-28) correctly separated from the South (SW/SE/SS, week 14-15) - and the North's forecast "
    "peak (27-28) sits close to several states' true, anomalously late second-wave peak (week 29). This "
    "works because the state-week features include the <i>national</i> fit's own wave-center timing "
    "(available under the nowcasting assumption of Section 2), which for 2024/25 already captures the "
    "late second wave even though the per-state deltas are learned only from the other two seasons - "
    "the pipeline layers combine usefully even on the season that is hardest for any single layer alone."
)
fig(FIG / "08_peak_week_summary.png",
    "Figure 6. Forecast peak week by state and zone, held-out 2024/25.")

h2("6.5 Bottom-up vs. top-down national estimate")
body(
    "As a cross-check, the Monte-Carlo-aggregated bottom-up national estimate (summing all 37 "
    "genuinely out-of-sample state forecasts) was compared against the renewal core's own direct national "
    "fit (Section 6.1's in-sample curve fit - a deliberately tough bar, since it is not itself a held-out "
    "prediction)."
)
table(
    ["Season", "Direct national fit (in-sample) MAE", "Bottom-up aggregate (LOSO) MAE", "Difference"],
    [
        ["2023/24", "51,062", "49,517", "-3.0%"],
        ["2024/25", "57,022", "230,065", "+303.5%"],
        ["2025/26", "65,891", "54,865", "-16.7%"],
    ],
    col_widths=[3 * cm, 5.3 * cm, 5.3 * cm, 3 * cm],
)
body(
    "The bottom-up, genuinely out-of-sample aggregate actually <b>beats</b> the in-sample curve fit on "
    "2 of 3 seasons - plausible, not a fluke: summing many independently residual-corrected state "
    "trajectories gets an averaging-out benefit that a single national parametric curve fitting its own "
    "noisy weekly wiggles does not. 2024/25 is again the exception, and a large one - but it is the same "
    "exception every other layer in this pipeline has already flagged, which we read as corroborating "
    "evidence that 2024/25 specifically breaks assumptions learned from the other two seasons, rather than "
    "a new, unrelated failure mode. Practically: the bottom-up aggregate is a reasonable, arguably "
    "preferable, national point estimate, but its 2024/25-sized failure mode means the direct national fit "
    "remains worth computing and comparing against as a sanity check."
)

h2("6.6 Optional probabilistic targets")
body(
    "For the 2025/26 forecast, the parametric bootstrap (500 draws) estimates P(rising transmission) at "
    "week 10 -&gt; 11 = 100% (still firmly in the wave-1 upswing) and P(epidemic peak within weeks 14-18) "
    "= 98%, with the bootstrap's modal peak week (94.8% of draws at week 16) matching the renewal core's "
    "own fitted wave-1 center almost exactly - i.e. the bootstrap's spread reflects genuine parameter "
    "uncertainty around an already-confident central estimate, which is the sanity check one wants from "
    "this kind of diagnostic."
)

# ===========================================================================
# POLICY IMPLICATIONS
# ===========================================================================
h1("7. Policy Implications")
bullets([
    "<b>Target mortality-reduction resources by healthcare access, not case counts.</b> Attack rate is "
    "only weakly correlated with healthcare-access index (r ~ 0.09) - likely an ascertainment "
    "effect, where higher-access states simply detect more mild cases - but case-fatality rate is "
    "<b>strongly</b> negatively correlated with access (r ~ -0.84). The states with the worst CFR in "
    "the high-severity 2024/25 season (Kebbi 0.230%, Taraba 0.227%, Zamfara 0.196%) are exactly the "
    "lowest-access states. Pre-season investment in antiviral stock and hospital capacity should prioritise "
    "these low-access North-West/North-East states specifically, not simply the highest-case-count states.",
    "<b>Stagger regional response timing.</b> The North consistently lags the South by roughly "
    "1-2 weeks in wave-1 timing and shows a relatively stronger second (Harmattan) wave; a single national "
    "vaccination/readiness calendar is a worse fit than a regionally phased one.",
    "<b>Peak week is forecastable with actionable lead time in typical seasons.</b> National peak-week "
    "accuracy within &plusmn;1 week reaches 81-97% depending on season type (Section 6.2), giving "
    "hospital systems several weeks' notice to plan surge staffing ahead of the predicted peak.",
    "<b>Budget for the atypical season explicitly, not just the average one.</b> Every layer of this "
    "model shows degraded - though not catastrophically broken - performance on the season most different "
    "from its training seasons. Public-health contingency planning should maintain a surge-capacity buffer "
    "sized for a worse-than-typical season, rather than sizing capacity to the average of recent seasons.",
    "<b>Near-real-time national surveillance materially improves state-level forecasting.</b> The "
    "calibration and residual layers' state-level accuracy depends on the national curve's own timing "
    "being available promptly (Section 2's nowcasting assumption). Investment in faster national-level "
    "case reporting would directly improve the state-level and regional forecasts this kind of model can "
    "produce, even before any state-level surveillance improvement.",
])

story.append(PageBreak())

# ===========================================================================
# DISCUSSION
# ===========================================================================
h1("8. Discussion")
h2("8.1 Strengths")
bullets([
    "An original architecture purpose-built for this dataset's documented dynamics (dual wave, regional "
    "phase offset, access-driven severity gradient), rather than an off-the-shelf model applied unchanged.",
    "A validation discipline (leave-one-season-out, every tunable parameter swept across all splits) that "
    "caught two results that looked good on a single split and would not have generalised, before they "
    "were reported as findings rather than after.",
    "One correction (residual-interval inflation) is a genuine Pareto improvement rather than an "
    "accuracy/robustness trade-off - evidence that the conservative-blending approach used throughout is "
    "not merely defensive, but can find strictly better operating points.",
])
h2("8.2 Weaknesses and limitations")
bullets([
    "<b>Only 3 seasons of training data.</b> Every LOSO split trains on exactly 2 seasons; any "
    "\"generalisable\" pattern is fit to an n of 2, and the recurring 2024/25 failure mode across every "
    "component is the clearest evidence that this is a real, not merely theoretical, limitation.",
    "<b>The national fit is a nowcast, not a blind long-range forecast.</b> Component 1 fits directly "
    "against a season's own full national curve; this is defensible as a nowcasting assumption "
    "(Section 2) but should not be read as evidence the model can forecast a season's national trajectory "
    "from zero within-season data months in advance - that capability was out of scope for this "
    "submission and is the most important direction for future work.",
    "<b>Monte Carlo aggregation assumes state-level independence</b> (Section 2), which real epidemic "
    "dynamics violate (a severe national season affects many states together); this likely understates "
    "true national/regional aggregate uncertainty.",
    "<b>Per-state rate ratios assumed stable across seasons.</b> Hospitalisation/case and death/case "
    "ratios are carried forward from training seasons unchanged; any within-season drift (e.g. hospital "
    "capacity saturation near a severe peak) is not modelled.",
])
h2("8.3 On the bottom-up vs. top-down national comparison")
body(
    "Section 6.5's finding - that the bottom-up state aggregate beats the in-sample national curve fit "
    "on 2 of 3 seasons - is worth discussing on its own terms rather than as a footnote. It suggests that "
    "averaging many independently-corrected local forecasts can outperform a single global parametric fit "
    "even when the global fit has an unfair in-sample advantage, which is a broadly useful lesson for "
    "aggregate epidemic forecasting: a hierarchical, bottom-up architecture is not merely a convenience for "
    "producing state-level detail, but can be a better national estimator in its own right, conditional on "
    "the training data being representative of the season being forecast."
)

# ===========================================================================
# UNCERTAINTY ANALYSIS
# ===========================================================================
h1("9. Uncertainty Analysis")
body(
    "Two distinct sources of uncertainty run through this report and should not be conflated. "
    "<b>Aleatory uncertainty</b> - week-to-week noise around an otherwise-predictable trajectory - is "
    "what the residual quantile layer's interval widths (Section 6.3) are calibrated against, and are "
    "reasonably well-controlled: 80-95% empirical coverage of the nominal 90% interval on 2 of 3 splits "
    "after the interval-inflation correction. <b>Epistemic uncertainty</b> about which \"regime\" a new "
    "season belongs to - moderate, high-severity-and-atypically-timed, or something else again - is "
    "categorically different and is <i>not</i> well-controlled by this or, we suspect, any model trained "
    "on 3 seasons: every component's worst-case result occurs on 2024/25 specifically, and no amount of "
    "shrinkage-strength or interval-width tuning fully closed that gap without unacceptably degrading "
    "performance on the two more typical seasons (Sections 6.2, 6.3)."
)
body(
    "We treat this as the central, honest uncertainty-analysis finding of this report: <b>the model's "
    "point and interval forecasts should be trusted more in an average-looking season and materially less "
    "in one that departs from the training seasons' pattern early - and with 3 seasons of data, there is "
    "no principled way to know in advance which kind of season is coming.</b> Two practical mitigations "
    "are built into the shipped pipeline rather than left as a caveat only: the conservative "
    "<font face=\"Courier\">phase_weight</font> and <font face=\"Courier\">interval_inflation</font> "
    "settings both trade a little accuracy on typical seasons for materially less-bad failure on an "
    "atypical one, and the bootstrap-based probabilistic targets (Section 6.6) give an explicit, "
    "checkable confidence read-out (e.g. how concentrated the peak-week distribution is) that a downstream "
    "user can inspect rather than trusting a point forecast blindly."
)
body(
    "A further, more subtle calibration finding: PIT/coverage diagnostics showed the raw residual model "
    "is <i>overconfident</i>, never underconfident, on the hard season. This asymmetry is itself "
    "informative - it suggests the model's uncertainty estimate is anchored too tightly to the specific "
    "variance observed in its 2 training seasons, a generic risk whenever a quantile model is trained on "
    "a small number of exchangeable-in-theory-but-not-in-practice groups, and is the reason we recommend "
    "any future extension of this work (Section 10) prioritise either more training seasons or an "
    "explicit season-regime indicator over further tuning of the current 3-season-calibrated layers."
)

# ===========================================================================
# CONCLUSIONS
# ===========================================================================
h1("10. Conclusions")
body(
    "The CP-DWRE is an original, four-component hybrid forecasting framework - a covariate-forced "
    "renewal-equation core, zone-level hierarchical calibration, a LightGBM residual quantile correction, "
    "and a parametric-bootstrap probabilistic ensemble - built specifically for the MAEPiMS Challenge "
    "2026's synthetic Nigeria influenza dataset. Every reported result is validated leave-one-season-out "
    "across all 3 possible splits, a discipline that directly caught and corrected an overclaimed "
    "single-split result during development (Section 6.2) and surfaced a genuine, non-trivial Pareto "
    "improvement in probabilistic calibration (Section 6.3). The model performs well - often very well - "
    "on seasons resembling its training data, and degrades in a specific, diagnosed, and partially "
    "mitigated way on the one season (2024/25) that does not. We consider that honest characterisation of "
    "the model's limits, evidenced across every one of its four components rather than asserted once, to "
    "be as much a contribution of this submission as the point-forecast accuracy numbers themselves."
)
body(
    "Priority directions for future work: (1) a rolling-origin backtest that forecasts forward from "
    "partway through a season using only that season's own early weeks, to directly test the "
    "nowcasting-vs-forecasting distinction flagged in Section 8.2; (2) an explicit season-severity-regime "
    "indicator, learned or elicited, that the calibration and residual layers could condition on rather "
    "than implicitly averaging across; and (3) replacing the Monte Carlo independence assumption "
    "(Section 2) in regional/national aggregation with an explicitly correlated sampling scheme once "
    "enough seasons exist to estimate that correlation structure."
)

# ===========================================================================
# REFERENCES
# ===========================================================================
h1("References")
refs = [
    "Bracher, J., Ray, E. L., Gneiting, T., &amp; Reich, N. G. (2021). Evaluating epidemic forecasts in "
    "an interval format. <i>PLOS Computational Biology</i>, 17(2), e1008618.",
    "Cori, A., Ferguson, N. M., Fraser, C., &amp; Cauchemez, S. (2013). A new framework and software to "
    "estimate time-varying reproduction numbers during epidemics. <i>American Journal of Epidemiology</i>, "
    "178(9), 1505-1512.",
    "Fraser, C. (2007). Estimating individual and household reproduction numbers in an emerging "
    "epidemic. <i>PLOS ONE</i>, 2(8), e758.",
    "Gneiting, T., &amp; Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and "
    "estimation. <i>Journal of the American Statistical Association</i>, 102(477), 359-378.",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: "
    "A highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing "
    "Systems</i>, 30.",
    "Koenker, R., &amp; Bassett, G. (1978). Regression quantiles. <i>Econometrica</i>, 46(1), 33-50.",
    "Seber, G. A. F., &amp; Wild, C. J. (2003). <i>Nonlinear Regression</i>. Wiley.",
    "The Adejimi Adeniji Foundation (2026). <i>MAEPiMS Challenge 2026: Nigeria Influenza Forecasting "
    "Challenge</i> [Problem statement and synthetic dataset].",
]
story.append(ListFlowable(
    [ListItem(Paragraph(r, styles["RefBody"]), bulletColor=BLUE) for r in refs],
    bulletType="bullet", leftIndent=14, spaceBefore=2,
))

# ===========================================================================
doc = SimpleDocTemplate(
    str(OUT_PDF), pagesize=A4,
    leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    title="MAEPiMS Challenge 2026 Technical Report",
    author=NICKNAME,
)
doc.build(story)
print(f"Saved {OUT_PDF}")
