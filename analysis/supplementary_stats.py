"""
Two pieces of the stats plan that sit alongside the primary ordinal
model (ordinal_logit.py):

1. Hierarchical OLS regression, the interpretability companion to the
   ordinal model, done with plain numpy so it doesn't add a dependency.

2. An approximate proportional-odds diagnostic. This is NOT a formal
   Brant test -- it fits a separate binary logistic model at each
   cumulative cutpoint of the ordinal outcome and looks at how much
   each predictor's coefficient moves across cutpoints. Big, erratic
   swings are evidence against proportional odds for that predictor.
   Useful as a screen, not a substitute for R's brant::brant() on a
   MASS::polr fit (or ordinal::clm) if something gets flagged.
"""

import numpy as np
from scipy import stats


def ols_fit(X, y, add_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]

    if add_intercept:
        X = np.column_stack([np.ones(n), X])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    p = X.shape[1]

    return {"beta": beta, "r2": r2, "ss_res": ss_res, "ss_tot": ss_tot,
            "n": n, "p": p, "df_resid": n - p}


def r2_change_f_test(fit_reduced, fit_full):
    """F-test for R^2 change between nested models. Returns (F, df1, df2, p, delta_r2)."""
    df1 = fit_full["p"] - fit_reduced["p"]
    df2 = fit_full["df_resid"]
    f_stat = ((fit_reduced["ss_res"] - fit_full["ss_res"]) / df1) / (fit_full["ss_res"] / df2)
    p_value = stats.f.sf(f_stat, df1, df2)
    return f_stat, df1, df2, p_value, fit_full["r2"] - fit_reduced["r2"]


def run_hierarchical_ols(y, step_blocks):
    """step_blocks: list of (label, X_block), entered cumulatively in order."""
    results = []
    X_cumulative = None
    prev_fit = None

    for label, X_block in step_blocks:
        X_block = np.asarray(X_block, dtype=float)
        if X_block.ndim == 1:
            X_block = X_block.reshape(-1, 1)
        X_cumulative = X_block if X_cumulative is None else np.column_stack([X_cumulative, X_block])
        fit = ols_fit(X_cumulative, y)

        entry = {"label": label, "fit": fit}
        if prev_fit is not None:
            f_stat, df1, df2, p_value, r2_change = r2_change_f_test(prev_fit, fit)
            entry.update(r2_change=r2_change, f_stat=f_stat, df1=df1, df2=df2, p_value=p_value)
        results.append(entry)
        prev_fit = fit

    return results


def format_hierarchical_ols_results(results):
    lines = []
    for i, entry in enumerate(results, start=1):
        fit = entry["fit"]
        lines.append(f"Step {i}: {entry['label']}")
        lines.append(f"  N={fit['n']}, R^2={fit['r2']:.4f}")
        if "r2_change" in entry:
            lines.append(f"  Delta R^2 vs. prior step = {entry['r2_change']:.4f}  "
                          f"F({entry['df1']},{entry['df2']}) = {entry['f_stat']:.3f}, "
                          f"p = {entry['p_value']:.4f}")
        lines.append("")
    return "\n".join(lines)


def binary_logit_fit(x, y_binary, add_intercept=True):
    """Small binary logit via scipy, used only for the cutpoint diagnostic below."""
    from scipy import optimize

    x = np.asarray(x, dtype=float)
    y_binary = np.asarray(y_binary, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    X = np.column_stack([np.ones(x.shape[0]), x]) if add_intercept else x

    def neg_ll(beta):
        p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-10, 1 - 1e-10)
        return -np.sum(y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p))

    result = optimize.minimize(neg_ll, np.zeros(X.shape[1]), method="BFGS")
    try:
        se = np.sqrt(np.diag(result.hess_inv))
    except Exception:
        se = np.full(X.shape[1], np.nan)
    return result.x, se


def proportional_odds_diagnostic(X, y, predictor_names, min_cell_count=20):
    """
    Fits Y<=k vs Y>k at every cutpoint k and compares each predictor's
    coefficient across cutpoints. Cutpoints with fewer than
    min_cell_count observations on either side are excluded rather than
    included as-is -- sparse extreme categories cause quasi-separation,
    which inflates the coefficient and its SE together, so a naive
    spread-vs-SE check would call it "fine" when the estimate is garbage.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    cutpoints = np.sort(np.unique(y))[:-1]

    stable, unstable, coef_rows, se_rows = [], [], [], []

    for k in cutpoints:
        y_bin = (y <= k).astype(float)
        n_pos, n_neg = int(y_bin.sum()), int(len(y_bin) - y_bin.sum())
        if n_pos < min_cell_count or n_neg < min_cell_count:
            unstable.append((k, n_pos, n_neg))
            continue
        beta, se = binary_logit_fit(X, y_bin)
        stable.append(k)
        coef_rows.append(beta[1:])
        se_rows.append(se[1:])

    results = {}
    if len(stable) >= 2:
        coef_matrix, se_matrix = np.array(coef_rows), np.array(se_rows)
        for j, name in enumerate(predictor_names):
            coefs, ses = coef_matrix[:, j], se_matrix[:, j]
            spread = float(np.nanmax(coefs) - np.nanmin(coefs))
            avg_se = float(np.nanmean(ses))
            results[name] = {
                "coefs_by_cutpoint": dict(zip([int(c) for c in stable], coefs.tolist())),
                "unstable_cutpoints": unstable,
                "spread": spread,
                "avg_se": avg_se,
                "flag": bool(spread > avg_se),
            }
    else:
        for name in predictor_names:
            results[name] = {"coefs_by_cutpoint": {}, "unstable_cutpoints": unstable,
                              "spread": None, "avg_se": None, "flag": None}
    return results


def format_proportional_odds_diagnostic(results):
    lines = ["Approximate proportional-odds diagnostic (heuristic, not a formal Brant test)"]
    any_unstable = False
    for name, info in results.items():
        if info["flag"] is None:
            lines.append(f"  {name}: fewer than 2 stable cutpoints -- can't assess with this data")
            any_unstable = True
            continue
        verdict = "possible violation, cross-check with R" if info["flag"] else "no strong evidence of violation"
        lines.append(f"  {name}: spread={info['spread']:.4f}, avg SE={info['avg_se']:.4f} -> {verdict}")
        for cut, coef in info["coefs_by_cutpoint"].items():
            lines.append(f"      Y<={cut}: coef={coef:.4f}")
        if info["unstable_cutpoints"]:
            any_unstable = True
            for k, n_pos, n_neg in info["unstable_cutpoints"]:
                lines.append(f"      Y<={k}: excluded (n_pos={n_pos}, n_neg={n_neg}, below min_cell_count)")
    if any_unstable:
        lines += [
            "",
            "Some cutpoints had too few observations on one side to fit reliably -- expected",
            "with sparse ordinal categories at moderate N, not a bug. Collapsing adjacent sparse",
            "categories for this diagnostic only (never for the primary model) is one option if",
            "this keeps happening.",
        ]
    return "\n".join(lines)
