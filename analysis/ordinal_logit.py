"""
Proportional-odds (cumulative logit) ordinal regression, fit by direct
MLE via scipy.optimize. No statsmodels dependency.

Model: for outcome Y in {1..K} and predictors X (n x p, no intercept),

    logit(P(Y <= k | X)) = theta_k - X @ beta,   k = 1..K-1

with theta_1 < theta_2 < ... < theta_{K-1}. The optimizer works over
unconstrained params [a_1..a_{K-1}, beta_1..beta_p] and thresholds are
recovered as theta_1 = a_1, theta_k = theta_{k-1} + exp(a_k).

Predictors are z-scored internally before optimization and the fitted
coefficients/thresholds/SEs are converted back to the original scale
afterward. Without this, columns on very different scales (age ~12-21
vs a 0/1 flag vs a Likert mean ~1-7) leave BFGS's Hessian estimate
poorly conditioned, and some diagonal entries never move off their
identity-matrix starting value -- you get an SE of ~1.0 for a
coefficient that has nothing to do with a variance of 1. Standardizing
first, then unstandardizing after, fixes the conditioning without
changing what's reported to the caller.

Caveats that don't go away just because the scaling bug is fixed:
- SEs still come from the BFGS inverse-Hessian approximation, not an
  analytic observed-information matrix. Cross-check key coefficients
  against R's MASS::polr for anything going in a paper or continuing
  review.
- No regularization. Small N with many covariates will still throw
  convergence warnings -- that's the model telling you something, not
  a bug to suppress.
"""

import numpy as np
from scipy import optimize, stats


class OrdinalLogisticRegression:
    def __init__(self):
        self.beta_ = None
        self.thresholds_ = None
        self.n_categories_ = None
        self.n_predictors_ = None
        self.loglik_ = None
        self.n_obs_ = None
        self.param_names_ = None
        self.converged_ = None
        self.categories_ = None
        self._hess_inv_std = None
        self._x_mean = None
        self._x_std = None

    @staticmethod
    def _unpack_params(params, n_thresh, n_pred):
        a = params[:n_thresh]
        beta = params[n_thresh:n_thresh + n_pred]
        theta = np.cumsum(np.concatenate([[a[0]], np.exp(a[1:])]))
        return theta, beta

    @staticmethod
    def _neg_loglik(params, X, y_idx, n_thresh, n_pred, n_categories):
        theta, beta = OrdinalLogisticRegression._unpack_params(params, n_thresh, n_pred)
        eta = X @ beta

        thresholds_full = np.concatenate([[-np.inf], theta, [np.inf]])
        upper = thresholds_full[y_idx + 1] - eta
        lower = thresholds_full[y_idx] - eta

        p_upper = 1.0 / (1.0 + np.exp(-upper))
        p_lower = 1.0 / (1.0 + np.exp(-lower))
        p_upper = np.where(np.isinf(upper), 1.0, p_upper)
        p_lower = np.where(np.isinf(lower), 0.0, p_lower)

        prob = np.clip(p_upper - p_lower, 1e-12, 1.0)
        return -np.sum(np.log(prob))

    def fit(self, X, y, predictor_names=None, maxiter=500):
        """
        X: (n, p) array, no intercept column.
        y: (n,) ordinal outcome, integer-coded; category labels are
           inferred from sorted unique values so they don't need to
           start at 0 or 1.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std[x_std == 0] = 1.0  # guard constant columns rather than dividing by zero
        X_std = (X - x_mean) / x_std

        categories = np.sort(np.unique(y))
        n_categories = len(categories)
        n_thresh = n_categories - 1
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        y_idx = np.array([cat_to_idx[v] for v in y])

        init_a = np.zeros(n_thresh)
        init_a[0] = -1.0
        init_params = np.concatenate([init_a, np.zeros(p)])

        result = optimize.minimize(
            self._neg_loglik, init_params,
            args=(X_std, y_idx, n_thresh, p, n_categories),
            method="BFGS",
            options={"maxiter": maxiter, "gtol": 1e-6},
        )

        # BFGS reports success=False fairly often on a benign precision-loss
        # stop, even when it's already sitting at the MLE -- checking the
        # gradient norm directly is more reliable than trusting the flag.
        # Genuine non-convergence (separation, collinearity) still shows up
        # as a large gradient, so retry once with a longer budget before
        # accepting the result.
        if not result.success:
            retry = optimize.minimize(
                self._neg_loglik, result.x,
                args=(X_std, y_idx, n_thresh, p, n_categories),
                method="BFGS",
                options={"maxiter": maxiter * 4, "gtol": 1e-8},
            )
            if retry.fun <= result.fun:
                result = retry

        theta_std, beta_std = self._unpack_params(result.x, n_thresh, p)
        grad_norm = np.linalg.norm(result.jac) if getattr(result, "jac", None) is not None else np.inf
        converged = result.success or grad_norm < 1e-3

        # back out of standardized space
        beta = beta_std / x_std
        theta = theta_std + x_mean @ beta

        self.beta_ = beta
        self.thresholds_ = theta
        self.n_categories_ = n_categories
        self.n_predictors_ = p
        self.n_obs_ = n
        self.loglik_ = -result.fun
        self.categories_ = categories
        self.converged_ = converged
        self.param_names_ = predictor_names or [f"x{i}" for i in range(p)]
        self._hess_inv_std = getattr(result, "hess_inv", None)
        self._x_mean = x_mean
        self._x_std = x_std

        if not converged:
            import warnings
            warnings.warn(
                f"did not converge (grad norm {grad_norm:.2e}, threshold 1e-3) -- "
                f"treat these coefficients as a starting point, not a result. "
                f"Usually separation, collinearity, or a near-empty outcome category.",
                stacklevel=2,
            )

        return self

    def coef_table(self):
        n_thresh = self.n_categories_ - 1
        if self._hess_inv_std is not None:
            var_std = np.diag(self._hess_inv_std)[n_thresh:n_thresh + self.n_predictors_]
            se_std = np.sqrt(np.abs(var_std))
            se = se_std / self._x_std  # same linear rescaling as beta
        else:
            se = np.full(self.n_predictors_, np.nan)

        z = self.beta_ / se
        pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))

        return [
            {"predictor": name, "coef": b, "se": s, "z": zz, "p": pv, "odds_ratio": np.exp(b)}
            for name, b, s, zz, pv in zip(self.param_names_, self.beta_, se, z, pvals)
        ]

    def summary(self):
        lines = [
            f"Ordinal logistic regression (proportional odds), N={self.n_obs_}, "
            f"{self.n_categories_} categories, converged={self.converged_}",
            f"Log-likelihood: {self.loglik_:.3f}",
            f"{'predictor':<28}{'coef':>10}{'SE':>10}{'z':>8}{'p':>10}{'OR':>10}",
        ]
        for row in self.coef_table():
            lines.append(f"{row['predictor']:<28}{row['coef']:>10.4f}{row['se']:>10.4f}"
                          f"{row['z']:>8.2f}{row['p']:>10.4f}{row['odds_ratio']:>10.3f}")
        lines.append(f"Thresholds: {np.round(self.thresholds_, 3).tolist()}")
        return "\n".join(lines)


def likelihood_ratio_test(loglik_full, loglik_reduced, df_diff):
    chi2_stat = max(2 * (loglik_full - loglik_reduced), 0.0)
    p_value = stats.chi2.sf(chi2_stat, df_diff)
    return chi2_stat, p_value


def mcfadden_pseudo_r2(loglik_full, loglik_null):
    return 1 - (loglik_full / loglik_null)


def fit_null_model(y):
    """Intercept-only fit (thresholds only, no predictors) for pseudo-R^2 baselines."""
    y = np.asarray(y)
    categories = np.sort(np.unique(y))
    n_thresh = len(categories) - 1
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    y_idx = np.array([cat_to_idx[v] for v in y])
    X_null = np.zeros((len(y), 0))

    init_a = np.zeros(n_thresh)
    init_a[0] = -1.0
    result = optimize.minimize(
        OrdinalLogisticRegression._neg_loglik, init_a,
        args=(X_null, y_idx, n_thresh, 0, len(categories)),
        method="BFGS",
    )

    model = OrdinalLogisticRegression()
    model.loglik_ = -result.fun
    model.n_obs_ = len(y)
    model.n_categories_ = len(categories)
    model.n_predictors_ = 0
    model.beta_ = np.array([])
    model.thresholds_, _ = OrdinalLogisticRegression._unpack_params(result.x, n_thresh, 0)
    return model
