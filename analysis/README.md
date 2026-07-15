# RCHOC INQ-10 Study — Analysis Code

Code for the protocol testing whether INQ-10 PB/TB subscale scores
explain unique variance in C-SSRS Ideation Severity Score among 12-21
year olds screened in the RCHOC ED, beyond age, sex,
medication/substance-use status, PHQ-A, and C-SSRS triage result.

## Files

- `power_analysis.py` — sample size / power for the primary model. Run
  this first.
- `config.py` — **edit this** for your REDCap export's column names
  and to confirm the INQ-10 field order.
- `analyze_redcap.py` — the pipeline; run this against a real export.
- `ordinal_logit.py` — proportional-odds regression from scratch (no
  statsmodels dependency).
- `supplementary_stats.py` — hierarchical OLS + proportional-odds
  diagnostic.
- `generate_synthetic_data.py` — fabricated test CSV, no real data
  required to try the pipeline.

## Quick start

```bash
pip install numpy pandas scipy scikit-learn

python3 generate_synthetic_data.py --output synth.csv --n 500
python3 analyze_redcap.py --input synth.csv --output results/
```

Once real data collection is underway, edit `config.py` (column names,
INQ-10 field order) and point `--input` at your actual export.

## Before running on real data

`config.py` ships with the INQ-10 PB/TB scoring key already filled in
and checked against the source table (Hill et al., 2015, Table 1) — PB
= items 1, 3, 9, 12, 14; TB = items 17, 19, 20, 21, 24; TB items 17,
19, 24 are reverse-scored. Don't touch `INQ_SUBSCALE_BY_ORIGINAL_ITEM`
or `REVERSE_SCORED_ORIGINAL_ITEMS`.

What you do need to confirm: `INQ10_FIELD_TO_ORIGINAL_ITEM` assumes
your REDCap instrument presents the 10 items in original-item order,
renumbered 1-10. That's the standard convention for a derived short
form, but it's specific to how your instrument was actually built —
check it against the live REDCap form before trusting results.

## What the pipeline runs

1. Primary Analytic Sample (positive ASQ, PHQ-A + C-SSRS complete,
   INQ-10 administered) vs. Screening-Cascade Descriptive Sample (all
   age-eligible patients).
2. INQ-10 scoring into PB/TB means.
3. Baseline descriptives.
4. Four-step hierarchical ordinal logistic regression (age/sex/med →
   PHQ-A/triage → PB+TB → PB×TB interaction), with an LR test per step.
5. Approximate proportional-odds diagnostic.
6. Supplementary hierarchical OLS with F-tests for R² change.
7. Secondary Aim 2 (screening-cascade proportions by age band, sex,
   race/ethnicity, insurance) and Aim 3 (screened vs. unscreened, by
   bypass reason, ESI, presenting complaint, age) — both descriptive
   only.
8. Cohen's kappa for any configured dual-abstraction fields.

GAD-7 is descriptive only and never enters the confirmatory model.

## Statistical caveats

- The proportional-odds diagnostic is a heuristic (cutpoint-wise
  coefficient spread vs. average SE), not a formal Brant test. If it
  flags a predictor, check against `brant::brant()` on a `MASS::polr`
  fit, or `ordinal::clm`, before concluding the assumption is actually
  violated. Note the protocol itself says a genuine violation should
  trigger a partial-PO or multinomial model in place of the primary
  one — this pipeline flags the issue but doesn't auto-switch models;
  that decision needs a human look at the flagged predictors.
- Predictors are z-scored internally before the ordinal fit and
  converted back afterward, which fixes a real conditioning problem
  (unscaled columns were leaving some standard errors stuck near their
  BFGS starting value). SEs are still from an inverse-Hessian
  approximation, not analytic observed information — cross-check
  anything going into a publication or continuing review against
  `MASS::polr` in R.
- Sparse ordinal categories exclude cutpoints from the PO diagnostic
  outright (< 20 observations on either side) rather than reporting a
  falsely reassuring "no violation" on an unstable estimate.
- Secondary Aim 2/3 chi-square and t-tests are descriptive /
  hypothesis-generating, per the protocol's own framing — not causal
  claims.
- ZIP code isn't auto-binned for the SES-proxy stratification; map it
  to a real grouping (e.g. income tercile) first if you want that cut.

## Reference

Hill, R., Rey, Y., Marin, C. E., Sharp, C., Green, K. L., & Pettit, J.
(2015). Evaluating the Interpersonal Needs Questionnaire: Comparison of
the reliability, factor structure, and predictive validity across five
versions. *Suicide & Life-Threatening Behavior, 45*(3), 302-314.

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
Sciences* (2nd ed.). Lawrence Erlbaum Associates.
