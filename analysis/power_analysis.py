"""
Sample size / power for the RCHOC INQ-10 study's primary hierarchical
model:

    DV: C-SSRS Ideation Severity Score (ordinal, 1-5)
    Step 1: age, biological sex, medication/substance-use status  (3)
    Step 2: PHQ-A total, C-SSRS triage                            (2)
    Step 3: INQ-10 PB & TB                                        (2, primary)
    Step 4: centered PB x TB interaction                          (1, secondary aim 1)

Solves, for a grid of Cohen's f^2 values, the N needed to detect the
R^2 change from a new predictor block via the noncentral F
distribution (Cohen, 1988, ch. 9). This is the standard OLS-style
planning approximation for this kind of incremental-validity design --
closed-form power solutions for R^2-change tests in ordinal logistic
regression aren't well standardized, so this is a planning estimate,
not the power calc for the confirmatory ordinal model itself.

    python3 power_analysis.py
"""

import numpy as np
from scipy import stats


def solve_n_for_f2(f2, u, k_prior, alpha=0.05, power=0.80, n_max=5000):
    """
    Smallest N such that u new predictors added to k_prior existing
    ones reach `power` at level `alpha`, for effect size f2. Noncentral
    F with lambda = f2 * N, df1 = u, df2 = N - k_prior - u - 1.
    Returns (n, df1, df2, achieved_power), or None if n_max isn't enough.
    """
    for n in range(u + k_prior + 3, n_max):
        df2 = n - k_prior - u - 1
        if df2 <= 0:
            continue
        f_crit = stats.f.ppf(1 - alpha, u, df2)
        achieved = 1 - stats.ncf.cdf(f_crit, u, df2, f2 * n)
        if achieved >= power:
            return n, u, df2, achieved
    return None


def r2_change_from_f2(f2):
    return f2 / (1 + f2)


def main():
    alpha, target_power = 0.05, 0.80
    effect_sizes = [0.020, 0.035, 0.050, 0.100, 0.150]

    step3 = {"label": "Step 3: INQ-10 PB & TB subscales (PRIMARY)", "u": 2, "k_prior": 5}
    step4 = {"label": "Step 4: Centered PB x TB interaction (SECONDARY AIM 1)", "u": 1, "k_prior": 7}

    for step in (step3, step4):
        print(f"=== {step['label']} ===")
        print(f"{'f2':>6} | {'~R2 change':>10} | {'N required':>11} | {'df1':>4} | {'df2':>5} | {'achieved power':>14}")
        for f2 in effect_sizes:
            result = solve_n_for_f2(f2, step["u"], step["k_prior"], alpha, target_power)
            if result is None:
                print(f"{f2:>6.3f} | could not reach target power within search range")
                continue
            n, df1, df2, achieved = result
            print(f"{f2:>6.3f} | {r2_change_from_f2(f2):>10.3f} | {n:>11d} | {df1:>4d} | {df2:>5d} | {achieved:>14.3f}")
        print()

    # planning anchor used in the protocol
    anchor_f2 = 0.035
    n_anchor, _, _, _ = solve_n_for_f2(anchor_f2, step3["u"], step3["k_prior"], alpha, target_power)
    n_buffered = int(np.ceil(n_anchor * 1.15))

    print("=== Planning recommendation ===")
    print(f"Anchor effect size: f2 = {anchor_f2} (~R2 change = {r2_change_from_f2(anchor_f2):.3f})")
    print(f"Minimum N for 80% power at Step 3: {n_anchor}")
    print(f"With a 15% buffer for incomplete INQ-10 / partial chart data: {n_buffered}")
    print()
    print("Literature-anchored planning estimate -- confirm against RCHOC ED's actual prior-period")
    print("volume of ASQ-positive, PHQ-A/C-SSRS-complete adolescents ages 12-21 before finalizing")
    print("the enrollment target.")


if __name__ == "__main__":
    main()
