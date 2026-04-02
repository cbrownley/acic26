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

from config import (
    CONTROL,
    TREATMENTS,
    ALPHA,
    USE_IF_CI,
    ENSEMBLE_W_AUTOML_NOBOOT,
    SE_FLOOR_FACTOR,
    USE_FOREST_DRL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _se_from_ci(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Recover standard error from symmetric 95% CI half-width."""
    return (upper - lower) / (2 * 1.96)


def _variance_weighted_combine(
    estimates: list[np.ndarray],
    ses: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal (minimum-variance) linear combination of K estimators.

    SE floor  (v5)
    --------------
    For each estimator k, the per-observation SE is floored at:
        SE_floor_k = SE_FLOOR_FACTOR * std(tau_k)

    Motivation: with only N_BOOT=50 bootstrap replicates, individual CI
    widths are noisy.  An observation that happens to get a very tight
    bootstrap interval would receive near-infinite precision weight,
    effectively ignoring all other estimators at that point.

    The floor caps the maximum precision ratio across observations at
    (1 / SE_FLOOR_FACTOR)^2 — with the default 0.1 this is 100:1.
    This is far less extreme than the prior 1e-6 floor (ratio of ~10^12),
    while still allowing the ensemble to upweight locally confident
    estimators by up to 100× relative to locally uncertain ones.

    Parameters
    ----------
    estimates : list of K arrays, each shape (n,) — point estimates
    ses       : list of K arrays, each shape (n,) — per-obs standard errors

    Returns
    -------
    tau_ens  : (n,) ensemble point estimate
    lb_ens   : (n,) lower 95% CI bound
    ub_ens   : (n,) upper 95% CI bound
    """
    floored_ses = []
    for tau_k, se_k in zip(estimates, ses):
        # Global std of this estimator's point predictions across all obs
        global_se_k = float(np.std(tau_k)) * SE_FLOOR_FACTOR
        # Floor must also be strictly positive even when std(tau_k) ~ 0
        floor = max(global_se_k, 1e-8)
        floored_ses.append(np.maximum(se_k, floor))

    precisions = [1.0 / se**2 for se in floored_ses]
    total_precision = sum(precisions)
    tau_ens = sum(p * est for p, est in zip(precisions, estimates)) / total_precision
    se_ens = np.sqrt(1.0 / total_precision)

    lb_ens = tau_ens - 1.96 * se_ens
    ub_ens = tau_ens + 1.96 * se_ens
    return tau_ens, lb_ens, ub_ens


# ─────────────────────────────────────────────────────────────────────────────
# iCATE  (variance-weighted ensemble of 2 or 3 estimators)
# ─────────────────────────────────────────────────────────────────────────────


def get_icates(
    est_drl,
    drl_has_ci: bool,  # False when FLAML was used (no BootstrapInference)
    est_lin,  # LinearDRLearner or None when USE_IF_CI = False
    forests: dict,
    X_feat: np.ndarray,
    n: int,
    forest_drl=None,  # ForestDRLearner or None when USE_FOREST_DRL=False
) -> tuple[dict, dict, dict]:
    """
    Variance-weighted ensemble iCATE point estimates and 95% CIs.

    Estimator roles
    ---------------
    est_drl (always)       point estimate — the most flexible nuisance fit
    est_drl (drl_has_ci)   also provides bootstrap CIs when sklearn nuisance
                           models are used (thread-safe → BootstrapInference ok)
    est_lin (USE_IF_CI)    influence-function / sandwich CIs — always analytic,
                           always fast, unaffected by the FLAML thread issue
    forests                GRF honesty-based variance — always analytic

    FLAML / BootstrapInference incompatibility
    ------------------------------------------
    When drl_has_ci=False (USE_AUTOML=True), est_drl was fitted without
    inference=BootstrapInference.  Calling effect_interval() would raise:
        RuntimeError: No inference was performed — call fit with inference first

    In this case the nonlinear DRL's point estimate is folded into the ensemble
    mean with a fixed conservative weight (ENSEMBLE_W_AUTOML_NOBOOT = 0.25).
    The CI computation uses only est_lin + forests, which together still give
    a valid variance-weighted interval.

    Parameters
    ----------
    est_drl     : fitted DRLearner (inference may or may not be attached)
    drl_has_ci  : whether effect_interval() is callable on est_drl
    est_lin     : fitted LinearDRLearner, or None
    forests     : dict {z: (CausalForestDML, mask)}
    X_feat      : (n, p) preprocessed features
    n           : sample size

    Returns
    -------
    icates  : dict {z: np.ndarray (n,)}  — ensemble point estimates
    lowers  : dict {z: np.ndarray (n,)}  — lower 95% CI bounds
    uppers  : dict {z: np.ndarray (n,)}  — upper 95% CI bounds
    """
    icates, lowers, uppers = {}, {}, {}
    ci_sources = []
    print("  [ENS] Computing variance-weighted ensemble iCATEs …")

    for z in TREATMENTS:

        # — always: nonlinear DRL point estimate ──────────────────────────────
        tau_drl = est_drl.effect(X_feat, T0=CONTROL, T1=z)

        # Lists fed into _variance_weighted_combine for CI computation only
        ci_est_list: list[np.ndarray] = []
        ci_se_list: list[np.ndarray] = []

        # — estimator 1 CIs: bootstrap (only when thread-safe nuisance) ──────
        if drl_has_ci:
            lb_drl, ub_drl = est_drl.effect_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            ci_est_list.append(tau_drl)
            ci_se_list.append(_se_from_ci(lb_drl, ub_drl))
            if z == TREATMENTS[0]:  # log once per arm set
                ci_sources.append("DRL-bootstrap")

        # — estimator 2 CIs: LinearDRL influence function ─────────────────────
        if USE_IF_CI and est_lin is not None:
            tau_lin = est_lin.effect(X_feat, T0=CONTROL, T1=z)
            lb_lin, ub_lin = est_lin.effect_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            ci_est_list.append(tau_lin)
            ci_se_list.append(_se_from_ci(lb_lin, ub_lin))
            if z == TREATMENTS[0]:
                ci_sources.append("LinearDRL-IF")

        # — estimator 3 CIs: CausalForest GRF variance ───────────────────────
        cf, _mask = forests[z]
        tau_cf = cf.effect(X_feat)
        lb_cf, ub_cf = cf.effect_interval(X_feat, alpha=ALPHA)
        ci_est_list.append(tau_cf)
        ci_se_list.append(_se_from_ci(lb_cf, ub_cf))
        if z == TREATMENTS[0]:
            ci_sources.append("CausalForest-GRF")

        # — estimator 4 CIs: ForestDRLearner (DR + honest forest) ────────────
        if USE_FOREST_DRL and forest_drl is not None:
            tau_fdrl = forest_drl.effect(X_feat, T0=CONTROL, T1=z)
            lb_fdrl, ub_fdrl = forest_drl.effect_interval(X_feat, T0=CONTROL, T1=z, alpha=ALPHA)
            ci_est_list.append(tau_fdrl)
            ci_se_list.append(_se_from_ci(lb_fdrl, ub_fdrl))
            if z == TREATMENTS[0]:
                ci_sources.append("ForestDRL-GRF")

        # — variance-weighted CI ───────────────────────────────────────────────
        # CIs are derived entirely from the analytic estimators above.
        # The nonlinear DRL point estimate is blended into the final mean
        # via a fixed weight when it has no bootstrap CI.
        tau_ci_ens, lb_ens, ub_ens = _variance_weighted_combine(ci_est_list, ci_se_list)

        if drl_has_ci:
            # DRL already included in ci_est_list → tau_ci_ens is the full mean
            tau_ens = tau_ci_ens
        else:
            # Blend nonlinear DRL point estimate with conservative fixed weight
            w_drl = ENSEMBLE_W_AUTOML_NOBOOT  # e.g. 0.25
            w_rest = 1.0 - w_drl
            tau_ens = w_drl * tau_drl + w_rest * tau_ci_ens

        icates[z] = tau_ens
        lowers[z] = lb_ens
        uppers[z] = ub_ens

    print(f"  [ENS] Done.  CI sources: {', '.join(ci_sources)}")
    return icates, lowers, uppers


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
