"""
RCHOC INQ-10 study -- main analysis pipeline.

    python3 analyze_redcap.py --input redcap_export.csv --output results/

What it does, roughly in order:
  1. Splits a REDCap export into the Primary Analytic Sample (positive
     ASQ, PHQ-A + C-SSRS complete, INQ-10 administered) and the
     Screening-Cascade Descriptive Sample (all age-eligible patients).
  2. Scores the INQ-10 into PB/TB (config.INQ_SUBSCALE_MAP).
  3. Baseline descriptives on the Primary Analytic Sample.
  4. The primary four-step hierarchical ordinal logistic regression on
     C-SSRS Ideation Severity, with an LR test per step.
  5. An approximate proportional-odds check (see supplementary_stats.py).
  6. The supplementary hierarchical OLS regression.
  7. Secondary Aim 2 (screening-cascade proportions) and Aim 3
     (screened vs. unscreened), both descriptive/hypothesis-generating.
  8. Cohen's kappa on any configured dual-abstraction field pairs.
  9. Writes results/analysis_report.txt.

GAD-7 is summarized descriptively only and never enters the confirmatory
model. Requires numpy, pandas, scipy, scikit-learn (kappa only).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    COLUMNS,
    INQ10_ITEM_COLUMNS,
    INQ_SUBSCALE_MAP,
    REVERSE_SCORED_ITEMS,
    INQ_ITEM_SCALE_MIN,
    INQ_ITEM_SCALE_MAX,
    DUAL_ABSTRACTION_FIELD_PAIRS,
)
from ordinal_logit import OrdinalLogisticRegression, likelihood_ratio_test, mcfadden_pseudo_r2, fit_null_model
from supplementary_stats import run_hierarchical_ols, format_hierarchical_ols_results, proportional_odds_diagnostic, format_proportional_odds_diagnostic


def load_data(path):
    return pd.read_csv(path)


def define_primary_analytic_sample(df):
    """
    Inclusion: age 12-21, positive ASQ, PHQ-A + C-SSRS complete, INQ-10
    administered. Bedside-judgment exclusions (cognitive/developmental
    impairment, intoxication, ESI=1) aren't re-derived here -- there's
    no reliable proxy column for a clinical judgment call made at
    triage. If your export has explicit exclusion flags, add them.
    """
    c = COLUMNS
    mask = (
        (df[c["asq_result"]].astype(str).str.lower() == "positive")
        & df[c["phqa_total"]].notna()
        & df[c["cssrs_ideation_severity"]].notna()
    )
    mask &= df[INQ10_ITEM_COLUMNS].notna().any(axis=1)

    if c["age"] in df.columns:
        mask &= df[c["age"]].between(12, 21)

    return df.loc[mask].copy()


def define_screening_cascade_sample(df):
    """All age-eligible ED patients, regardless of screening completion -- Aims 2 & 3 only."""
    c = COLUMNS
    if c["age"] in df.columns:
        return df.loc[df[c["age"]].between(12, 21)].copy()
    return df.copy()


def score_inq10(df):
    """
    Adds inq10_pb / inq10_tb (mean of the mapped items, reverse-scored
    where applicable). The subscale assignment is sourced (config.py);
    what this can't check automatically is whether your REDCap field
    order matches config.INQ10_FIELD_TO_ORIGINAL_ITEM -- confirm that
    against your actual instrument.
    """
    print("Scoring INQ-10 -- confirm config.INQ10_FIELD_TO_ORIGINAL_ITEM matches your "
          "REDCap instrument's actual field order before trusting these results.")

    df = df.copy()
    scored = {}
    for item_col, subscale in INQ_SUBSCALE_MAP.items():
        if item_col not in df.columns:
            raise ValueError(f"'{item_col}' is in INQ_SUBSCALE_MAP but not in the data -- "
                              f"check config.COLUMNS / INQ10_FIELD_TO_ORIGINAL_ITEM against "
                              f"your export's actual field names.")
        vals = df[item_col].astype(float)
        if item_col in REVERSE_SCORED_ITEMS:
            vals = (INQ_ITEM_SCALE_MIN + INQ_ITEM_SCALE_MAX) - vals
        scored[item_col] = vals

    scored_df = pd.DataFrame(scored)
    pb_cols = [c for c, s in INQ_SUBSCALE_MAP.items() if s == "PB"]
    tb_cols = [c for c, s in INQ_SUBSCALE_MAP.items() if s == "TB"]

    df["inq10_pb"] = scored_df[pb_cols].mean(axis=1) if pb_cols else np.nan
    df["inq10_tb"] = scored_df[tb_cols].mean(axis=1) if tb_cols else np.nan
    return df


def describe_sample(df, report):
    c = COLUMNS
    report.append("=== Baseline Descriptive Statistics (Primary Analytic Sample) ===")
    report.append(f"N = {len(df)}")

    if c["age"] in df.columns:
        report.append(f"Age: mean={df[c['age']].mean():.2f}, SD={df[c['age']].std():.2f}, "
                       f"range=[{df[c['age']].min():.0f}, {df[c['age']].max():.0f}]")
    if c["sex"] in df.columns:
        report.append(f"Biological sex: {dict(df[c['sex']].value_counts(dropna=False))}")
    if c["medication_status"] in df.columns:
        report.append(f"Documented elevated-risk medication/substance use: "
                       f"{100 * df[c['medication_status']].mean():.1f}%")
    if c["phqa_total"] in df.columns:
        report.append(f"PHQ-A total: mean={df[c['phqa_total']].mean():.2f}, SD={df[c['phqa_total']].std():.2f}")
    if c["gad7_total"] in df.columns:
        report.append(f"GAD-7 total (descriptive only, not in confirmatory model): "
                       f"mean={df[c['gad7_total']].mean():.2f}, SD={df[c['gad7_total']].std():.2f}")
    if "inq10_pb" in df.columns:
        report.append(f"INQ-10 PB: mean={df['inq10_pb'].mean():.2f}, SD={df['inq10_pb'].std():.2f}")
        report.append(f"INQ-10 TB: mean={df['inq10_tb'].mean():.2f}, SD={df['inq10_tb'].std():.2f}")
    if c["cssrs_ideation_severity"] in df.columns:
        counts = df[c["cssrs_ideation_severity"]].value_counts(dropna=False).sort_index()
        report.append(f"C-SSRS Ideation Severity Score distribution: {dict(counts)}")
    report.append("")


def build_model_matrices(df):
    """
    Step 1: age, biological sex, medication/substance-use status
    Step 2: PHQ-A total, C-SSRS triage result
    Step 3: INQ-10 PB & TB
    Step 4: centered PB x TB interaction

    Categorical predictors are expected numeric/binary already, per
    config.py -- recode upstream if your export uses text labels.
    """
    c = COLUMNS
    step1 = df[[c["age"], c["sex"], c["medication_status"]]].to_numpy(dtype=float)
    step2 = df[[c["phqa_total"], c["cssrs_triage_result"]]].to_numpy(dtype=float)
    step3 = df[["inq10_pb", "inq10_tb"]].to_numpy(dtype=float)

    pb_c = df["inq10_pb"] - df["inq10_pb"].mean()
    tb_c = df["inq10_tb"] - df["inq10_tb"].mean()
    step4 = (pb_c * tb_c).to_numpy(dtype=float).reshape(-1, 1)

    return [
        ("Step 1: Demographics", step1, [c["age"], c["sex"], c["medication_status"]]),
        ("Step 2: Standard Care Benchmarks", step2, [c["phqa_total"], c["cssrs_triage_result"]]),
        ("Step 3: INQ-10 PB & TB (primary hypothesis)", step3, ["inq10_pb", "inq10_tb"]),
        ("Step 4: Centered PB x TB interaction (secondary aim 1)", step4, ["pb_x_tb_centered"]),
    ]


def run_primary_ordinal_model(df, report):
    c = COLUMNS
    y = df[c["cssrs_ideation_severity"]].to_numpy()
    step_blocks = build_model_matrices(df)

    report.append("=== Primary Analysis: Hierarchical Ordinal Logistic Regression ===")
    report.append("(Proportional-odds model; DV = C-SSRS Ideation Severity Score, 1-5)\n")

    X_cum, names_cum, prev_model, fitted = None, [], None, []
    null_model = fit_null_model(y)

    for label, X_block, block_names in step_blocks:
        X_cum = X_block if X_cum is None else np.column_stack([X_cum, X_block])
        names_cum = names_cum + block_names

        model = OrdinalLogisticRegression()
        model.fit(X_cum, y, predictor_names=names_cum)
        fitted.append(model)

        report.append(f"--- {label} ---")
        report.append(model.summary())

        pseudo_r2 = mcfadden_pseudo_r2(model.loglik_, null_model.loglik_)
        report.append(f"McFadden pseudo-R^2 (vs. null model): {pseudo_r2:.4f}")

        if prev_model is not None:
            df_diff = model.n_predictors_ - prev_model.n_predictors_
            chi2_stat, p_value = likelihood_ratio_test(model.loglik_, prev_model.loglik_, df_diff)
            report.append(f"LR test for this step's incremental contribution: "
                           f"chi2({df_diff}) = {chi2_stat:.3f}, p = {p_value:.4f}")
        report.append("")
        prev_model = model

    return fitted


def run_proportional_odds_check(df, report):
    c = COLUMNS
    y = df[c["cssrs_ideation_severity"]].to_numpy()
    step_blocks = build_model_matrices(df)
    X_full = np.column_stack([blk for _, blk, _ in step_blocks])
    names_full = sum([names for _, _, names in step_blocks], [])

    report.append("=== Proportional-Odds Assumption Check ===")
    diag = proportional_odds_diagnostic(X_full, y, names_full)
    report.append(format_proportional_odds_diagnostic(diag))
    report.append("")
    return diag


def run_supplementary_ols(df, report):
    c = COLUMNS
    y = df[c["cssrs_ideation_severity"]].to_numpy(dtype=float)
    step_blocks = build_model_matrices(df)
    ols_blocks = [(label, X_block) for label, X_block, _ in step_blocks]

    report.append("=== Supplementary Analysis: Hierarchical OLS Regression ===")
    report.append("(Interpretability analysis only; ordinal logistic regression above is primary)")
    results = run_hierarchical_ols(y, ols_blocks)
    report.append(format_hierarchical_ols_results(results))
    return results


def _age_bins(age_series):
    return pd.cut(age_series, bins=[11, 14, 17, 21], labels=["12-14", "15-17", "18-21"])


def secondary_aim_2_cascade(df, report):
    """
    Descriptive: proportion of PHQ-A completers who go on to receive
    the C-SSRS, stratified by age band, sex, race/ethnicity, and
    insurance type. Hypothesis-generating only, per protocol -- no
    causal claims.
    """
    c = COLUMNS
    report.append("=== Secondary Aim 2: Screening Cascade (Descriptive) ===")

    phqa_completers = df[df[c["phqa_total"]].notna()].copy()
    phqa_completers["received_cssrs"] = df[c["cssrs_ideation_severity"]].notna()

    report.append(f"PHQ-A completers, N = {len(phqa_completers)}")
    report.append(f"Overall proportion receiving C-SSRS: {phqa_completers['received_cssrs'].mean():.3f}\n")

    if c["age"] in phqa_completers.columns:
        phqa_completers["age_band"] = _age_bins(phqa_completers[c["age"]])
        strata_cols = {"age_band": "age_band", "sex": c["sex"],
                        "race_ethnicity": c["race_ethnicity"], "insurance_type": c["insurance_type"]}
    else:
        strata_cols = {"sex": c["sex"], "race_ethnicity": c["race_ethnicity"], "insurance_type": c["insurance_type"]}

    for stratum_label, col in strata_cols.items():
        if col not in phqa_completers.columns:
            continue
        report.append(f"-- Stratified by {stratum_label} --")
        crosstab = pd.crosstab(phqa_completers[col], phqa_completers["received_cssrs"])
        report.append(str(crosstab))
        if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(crosstab)
            report.append(f"Chi-square({dof}) = {chi2:.3f}, p = {p:.4f} (descriptive only)")
        report.append("")

    report.append("ZIP code (SES proxy) isn't auto-binned here -- map it to a meaningful grouping")
    report.append("(e.g. income tercile from a public crosswalk) before stratifying by it.\n")


def secondary_aim_3_screened_vs_unscreened(df_cascade, report):
    """
    Descriptive comparison of screened vs. unscreened age-eligible
    patients: bypass reason, ESI level, age, and presenting complaint.
    Hypothesis-generating only.
    """
    c = COLUMNS
    report.append("=== Secondary Aim 3: Screened vs. Unscreened Comparison (Descriptive) ===")

    screened = df_cascade[c["asq_result"]].notna()
    report.append(f"Screened (ASQ administered): N = {screened.sum()}")
    report.append(f"Unscreened / bypassed: N = {(~screened).sum()}\n")

    if c["bypass_reason"] in df_cascade.columns:
        report.append("Documented bypass reasons among unscreened patients:")
        report.append(str(df_cascade.loc[~screened, c["bypass_reason"]].value_counts(dropna=False)))
        report.append("")

    if c["esi_level"] in df_cascade.columns:
        report.append("ESI level by screening status:")
        report.append(str(pd.crosstab(df_cascade[c["esi_level"]], screened)))
        report.append("")

    if c["presenting_complaint"] in df_cascade.columns:
        report.append("Presenting complaint by screening status:")
        report.append(str(pd.crosstab(df_cascade[c["presenting_complaint"]], screened)))
        report.append("")

    if c["age"] in df_cascade.columns:
        screened_age = df_cascade.loc[screened, c["age"]].dropna()
        unscreened_age = df_cascade.loc[~screened, c["age"]].dropna()
        if len(screened_age) > 1 and len(unscreened_age) > 1:
            t_stat, p_val = stats.ttest_ind(screened_age, unscreened_age, equal_var=False)
            report.append(f"Age, screened vs. unscreened: t = {t_stat:.3f}, p = {p_val:.4f} (descriptive only)")
    report.append("")


def compute_inter_rater_reliability(df, report):
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        report.append("scikit-learn not installed -- skipping inter-rater reliability.")
        return

    report.append("=== Inter-Rater Reliability (10% Dual-Abstraction Subsample) ===")
    any_computed = False
    for field_label, (col_a, col_b) in DUAL_ABSTRACTION_FIELD_PAIRS.items():
        if col_a not in df.columns or col_b not in df.columns:
            continue
        paired = df[[col_a, col_b]].dropna()
        if len(paired) == 0:
            continue
        kappa = cohen_kappa_score(paired[col_a], paired[col_b])
        report.append(f"{field_label}: Cohen's kappa = {kappa:.3f} (N pairs = {len(paired)})")
        any_computed = True

    if not any_computed:
        report.append("No dual-abstraction pairs found -- update config.DUAL_ABSTRACTION_FIELD_PAIRS.")
    report.append("")


def main():
    parser = argparse.ArgumentParser(description="RCHOC INQ-10 study analysis pipeline")
    parser.add_argument("--input", required=True, help="Path to REDCap export CSV")
    parser.add_argument("--output", default="results", help="Output directory for the report")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    report = []

    df_raw = load_data(args.input)
    df_cascade = define_screening_cascade_sample(df_raw)
    df_primary = define_primary_analytic_sample(df_raw)

    try:
        df_primary = score_inq10(df_primary)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    describe_sample(df_primary, report)
    run_primary_ordinal_model(df_primary, report)
    run_proportional_odds_check(df_primary, report)
    run_supplementary_ols(df_primary, report)
    secondary_aim_2_cascade(df_cascade, report)
    secondary_aim_3_screened_vs_unscreened(df_cascade, report)
    compute_inter_rater_reliability(df_raw, report)

    report_path = os.path.join(args.output, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"Done. Report written to {report_path}")


if __name__ == "__main__":
    main()
