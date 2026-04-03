"""
inference.py
Compute all four estimands from fitted estimators.

v2: three-way variance-weighted ensemble
-----------------------------------------
Each of the three CATE estimators provides a point estimate and a pointwise
variance estimate.  The optimal linear combination of K unbiased estimators
is the inverse-variance (precision) weighted average:

    w_k(x_i)    = 1 / sigma_k^2(x_i)
    tau_ens(x_i) = sum_k w_k * tau_k  /  sum_k w_k
    sigma_ens^2  = 1 / sum_k w_k(x_i)

sigma_k^2 is derived from each estimator's 95% interval:
    sigma_k(x_i) = (U95_k(x_i) - L95_k(x_i)) / (2 * 1.96)

When USE_IF_CI = False only two estimators are combined (DRL + CF).

v5 change: SE floor in _variance_weighted_combine
--------------------------------------------------
Prior code used MIN_SE = 1e-6, effectively zero.  A bootstrap run returning
a very tight CI for one observation would give that point near-infinite
precision weight, collapsing the ensemble to a single estimator locally.

New floor: SE_k(x_i) >= SE_FLOOR_FACTOR * std(tau_k_all)
This caps the max precision ratio at (1/SE_FLOOR_FACTOR)^2 = 100 while
preserving the observation-level adaptivity that makes per-obs weighting
superior to the global scalar weights used in dr_r_cf_ensemble_var_weighted.

Public API
----------
get_icates(est_drl, drl_has_ci, est_lin, forests, X_feat, n)
compute_scate(icates, lowers, uppers, n)            -> pd.DataFrame
compute_subcate(icates, lowers, uppers, x12, n)     -> pd.DataFrame
compute_pate(est_drl, drl_has_ci, est_lin, X_feat)  -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from scipy.optimize import nnls  # New import for NNLS optimization
from config import TREATMENTS, CONTROL, ALPHA, ENSEMBLE_STRATEGY

from config import (
    CONTROL,
    TREATMENTS,
    ALPHA,
    USE_IF_CI,
    SE_FLOOR_FACTOR,
    USE_FOREST_DRL,
    ENSEMBLE_STRATEGY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _se_from_ci(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Recover standard error from symmetric 95% CI half-width."""
    return (upper - lower) / (2 * 1.96)


def _get_rloss_weights(Y, T_str, M, est_drl, target_z):
    """
    Computes optimal weights for the ensemble by minimizing the Robinson Loss.
    Fixes AttributeError: 'list' object has no attribute 'model_propensity'
    Fixes AttributeError: 'DRLearner' object has no attribute 'classes_'
    """
    # 1. Correctly extract treatment categories to find the column index
    try:
        # EconML fitted learners store the OHE transformer in 'transformer_'
        categories = list(est_drl.transformer_.categories_[0])
    except AttributeError:
        # Fallback for older versions or LinearDRLearner
        categories = getattr(est_drl, "classes_", list(TREATMENTS))

    if target_z not in categories:
        return np.ones(M.shape[1]) / M.shape[1]

    z_idx = categories.index(target_z)

    # 2. Extract Cross-Fitted Nuisance Scores
    # These are already computed during est_drl.fit()
    try:
        # mu_hat: E[Y|X, T] for the observed treatment
        mu_hat = est_drl.nuisance_scores_regression_.flatten()
        # e_hat: P(T=t|X) for all treatment arms
        e_hat = est_drl.nuisance_scores_propensity_[:, z_idx]
    except (AttributeError, TypeError):
        # Safety fallback if nuisances are missing or malformed
        return np.ones(M.shape[1]) / M.shape[1]

    # 3. Robinson / DML Residuals
    Y_res = Y - mu_hat
    T_res = (T_str == target_z).astype(float) - e_hat

    # 4. Construct Design Matrix: Y_res ~ sum(w_k * tau_k * T_res)
    # M has shape (n, 4) containing point estimates from the candidate models
    X_nnls = M * T_res[:, np.newaxis]

    # Filter any NaNs to ensure NNLS convergence
    mask = ~np.isnan(X_nnls).any(axis=1) & ~np.isnan(Y_res)
    if not np.any(mask):
        return np.ones(M.shape[1]) / M.shape[1]

    # 5. Solve Non-Negative Least Squares (SuperLearner optimization)
    weights, _ = nnls(X_nnls[mask], Y_res[mask])

    # Normalize weights so they sum to 1
    sum_w = np.sum(weights)
    return weights / sum_w if sum_w > 0 else np.ones(M.shape[1]) / M.shape[1]


def get_icates(
    est_drl, drl_has_ci, est_lin, forests, X_feat, n, est_fdrl=None, Y=None, T=None
):
    """
    Ensemble iCATEs using R-Loss weights and analytic confidence intervals.
    """
    icates, lowers, uppers, weight_log = {}, {}, {}, {}

    for z in TREATMENTS:
        # 1. Point Estimates from all 4 ensemble members
        tau_drl = est_drl.effect(X_feat, T0=CONTROL, T1=z)
        tau_lin = est_lin.effect(X_feat, T0=CONTROL, T1=z) if est_lin else tau_drl
        tau_cf = forests[z][0].effect(X_feat)
        tau_fdrl = est_fdrl.effect(X_feat, T0=CONTROL, T1=z) if est_fdrl else tau_drl

        M = np.column_stack([tau_drl, tau_lin, tau_cf, tau_fdrl])

        # 2. Optimized Weights
        if ENSEMBLE_STRATEGY == "RLOSS" and Y is not None and T is not None:
            # Note: We use the main DRLearner's residuals as the reference
            w = _get_rloss_weights(Y, T, M, est_drl, z)
            weight_log[z] = w
            tau_ens = np.dot(M, w)
        else:
            tau_ens = M.mean(axis=1)

        # 3. Confidence Interval Pooling (using the 3 models with analytic intervals)
        se_pool, w_pool = [], []

        if est_lin:
            lb, ub = est_lin.effect_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            se_pool.append(((ub - lb) / 3.92) ** 2)
            w_pool.append(w[1] if ENSEMBLE_STRATEGY == "RLOSS" else 1.0)

        lb_cf, ub_cf = forests[z][0].effect_interval(X_feat, alpha=ALPHA)
        se_pool.append(((ub_cf - lb_cf) / 3.92) ** 2)
        w_pool.append(w[2] if ENSEMBLE_STRATEGY == "RLOSS" else 1.0)

        if est_fdrl:
            lb, ub = est_fdrl.effect_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            se_pool.append(((ub - lb) / 3.92) ** 2)
            w_pool.append(w[3] if ENSEMBLE_STRATEGY == "RLOSS" else 1.0)

        # Weighted variance aggregation
        w_pool = np.array(w_pool) / (np.sum(w_pool) + 1e-10)
        var_ens = np.sum([wp**2 * sp for wp, sp in zip(w_pool, se_pool)], axis=0)
        se_ens = np.sqrt(var_ens)

        icates[z] = tau_ens
        lowers[z] = tau_ens - 1.96 * se_ens
        uppers[z] = tau_ens + 1.96 * se_ens

    return icates, lowers, uppers, weight_log


# ─────────────────────────────────────────────────────────────────────────────
# sCATE
# ─────────────────────────────────────────────────────────────────────────────


def compute_scate(
    icates: dict,
    lowers: dict,
    uppers: dict,
    n: int,
) -> pd.DataFrame:
    """
    sCATE(z) = (1/n) sum_i iCATE(x_i, z)

    CI via delta method: SE[mean] = sqrt(sum SE_i^2) / n
    """
    rows = []
    for z in TREATMENTS:
        tau = icates[z]
        se_i = _se_from_ci(lowers[z], uppers[z])
        est = float(tau.mean())
        se_avg = float(np.sqrt((se_i**2).sum()) / n)
        rows.append(
            {
                "z": z,
                "Estimate": est,
                "L95": est - 1.96 * se_avg,
                "U95": est + 1.96 * se_avg,
            }
        )
    return pd.DataFrame(rows)[["z", "Estimate", "L95", "U95"]]


# ─────────────────────────────────────────────────────────────────────────────
# subCATE
# ─────────────────────────────────────────────────────────────────────────────


def compute_subcate(
    icates: dict,
    lowers: dict,
    uppers: dict,
    x12: np.ndarray,
    n: int,
) -> pd.DataFrame:
    """
    subCATE(z, x) = (1/n) sum_i iCATE(x_i, z) * 1(x12_i = x)
    Normalised by full n so subCATE(z,0) + subCATE(z,1) = sCATE(z).
    """
    rows = []
    for x_val in [0, 1]:
        mask = (x12 == x_val).astype(float)
        for z in TREATMENTS:
            tau = icates[z]
            se_i = _se_from_ci(lowers[z], uppers[z])
            est = float((tau * mask).mean())
            se_sub = float(np.sqrt(((se_i * mask) ** 2).sum()) / n)
            rows.append(
                {
                    "z": z,
                    "x": int(x_val),
                    "Estimate": est,
                    "L95": est - 1.96 * se_sub,
                    "U95": est + 1.96 * se_sub,
                }
            )
    return pd.DataFrame(rows)[["z", "x", "Estimate", "L95", "U95"]]


# ─────────────────────────────────────────────────────────────────────────────
# PATE  (uses DRL influence function)
# ─────────────────────────────────────────────────────────────────────────────


def compute_pate(
    est_drl,
    drl_has_ci: bool,
    est_lin,
    X_feat: np.ndarray,
) -> pd.DataFrame:
    """
    PATE(z) via the AIPW ATE estimator.

    Point estimate
    --------------
    Always taken from the nonlinear DRLearner (est_drl.ate()), which gives
    the most flexible doubly-robust ATE estimate regardless of whether
    bootstrap inference was attached.

    Confidence interval
    -------------------
    drl_has_ci=True  (USE_AUTOML=False, sklearn nuisance models):
        est_drl.ate_interval() — bootstrap CI from BootstrapInference.

    drl_has_ci=False (USE_AUTOML=True, FLAML nuisance — no bootstrap):
        est_lin.ate_interval() — HC1 sandwich CI from the LinearDRLearner
        with StatsModelsInferenceDiscrete.  The linear model is less
        flexible but its ATE CI is analytically valid and fast.

    If neither source is available (est_lin is None and drl_has_ci=False),
    a NaN interval is returned rather than crashing.
    """
    rows = []
    for z in TREATMENTS:
        ate_val = float(est_drl.ate(X_feat, T0=CONTROL, T1=z))

        if drl_has_ci:
            ate_lb, ate_ub = est_drl.ate_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            ate_lb, ate_ub = float(ate_lb), float(ate_ub)
        elif est_lin is not None:
            ate_lb, ate_ub = est_lin.ate_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            ate_lb, ate_ub = float(ate_lb), float(ate_ub)
        else:
            ate_lb, ate_ub = float("nan"), float("nan")

        rows.append(
            {
                "z": z,
                "Estimate": ate_val,
                "L95": ate_lb,
                "U95": ate_ub,
            }
        )
    return pd.DataFrame(rows)[["z", "Estimate", "L95", "U95"]]
