"""
estimators.py
Fitting the three CATE estimators used in the variance-weighted ensemble.

v2 additions
------------
fit_linear_drlearner(X_feat, Y, T)
    A LinearDRLearner with a polynomial featurizer and StatsModels sandwich
    inference — gives influence-function CIs in seconds instead of minutes,
    at the cost of assuming τ(x) is linear in the (expanded) feature space.
    Used as the third member of the variance-weighted ensemble.

fit_drlearner  and  fit_causal_forest  now use AutoML nuisance models
(when config.USE_AUTOML = True) and respect config.INNER_N_JOBS.

Public API
----------
fit_drlearner(X_feat, Y, T)           -> fitted DRLearner
fit_linear_drlearner(X_feat, Y, T)    -> fitted LinearDRLearner  [new]
fit_causal_forest(X_feat, Y, T)       -> dict {z: (fitted_CF, mask)}
"""
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
)

from econml.dr  import DRLearner, LinearDRLearner
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference, StatsModelsInferenceDiscrete

from config import (
    ALL_ARMS, CONTROL, TREATMENTS,
    N_CV_FOLDS, N_BOOT, CF_N_TREES,
    USE_AUTOML, AUTOML_BOOTSTRAP,
    RANDOM_STATE, INNER_N_JOBS, USE_IF_CI,
)
from models import (
    make_outcome_model, make_propensity_model, make_cate_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nonlinear DR-Learner  (bootstrap CIs, AutoML nuisance)
# ─────────────────────────────────────────────────────────────────────────────

def fit_drlearner(
    X_feat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> tuple:
    """
    Fit a nonlinear doubly-robust learner with cross-fitting.

    Returns
    -------
    (fitted_DRLearner, has_bootstrap_ci: bool)

    has_bootstrap_ci = True   when USE_AUTOML=False (sklearn models, thread-safe)
    has_bootstrap_ci = False  when USE_AUTOML=True  (FLAML, not thread-safe)

    Bootstrap / FLAML incompatibility
    ----------------------------------
    BootstrapInference runs N_BOOT DRLearner fits inside joblib threads.
    Each thread calls FLAML.fit(), which internally calls flaml.tune.run()
    and creates a trial runner.  Inside a joblib thread the runner is None,
    causing:
        AttributeError: 'NoneType' object has no attribute 'stop_trial'

    Fix: when USE_AUTOML=True, attach inference=None.  The point estimate
    is still correct and still used in the ensemble.  CIs come from the
    LinearDRLearner (IF/sandwich) and CausalForest (GRF), which have their
    own analytic CI paths that are completely unaffected by this issue.

    When USE_AUTOML=False, sklearn estimators are thread-safe so
    BootstrapInference(n_jobs=-1) works normally.
    """
    use_bootstrap = (not USE_AUTOML) or AUTOML_BOOTSTRAP

    est = DRLearner(
        model_regression      = make_outcome_model(),
        model_propensity      = make_propensity_model(),
        model_final           = make_cate_model(),
        cv                    = N_CV_FOLDS,
        mc_iters              = 1,
        random_state          = RANDOM_STATE,
        categories            = ALL_ARMS,
        multitask_model_final = False,
    )

    if use_bootstrap:
        inference_obj = BootstrapInference(
            n_bootstrap_samples=N_BOOT,
            n_jobs=INNER_N_JOBS,
            random_state=RANDOM_STATE,
        )
        print("  [DRL] Fitting nonlinear DR-Learner "
              f"(AutoML nuisance, bootstrap CIs n={N_BOOT}) …")
    else:
        inference_obj = None
        print("  [DRL] Fitting nonlinear DR-Learner "
              "(AutoML nuisance, no bootstrap — FLAML/thread incompatibility) …")
        print("        CIs will come from LinearDRLearner + CausalForest.")

    est.fit(Y, T, X=X_feat, inference=inference_obj)
    print("  [DRL] Done.")
    return est, use_bootstrap


# ─────────────────────────────────────────────────────────────────────────────
# 2. Linear DR-Learner  (StatsModelsInferenceDiscrete HC1 CIs)
#
# Previous attempts used:
#   (a) StatsModelsInference       → AssertionError: fit_intercept is True
#       Wrong class: StatsModelsInference is for continuous-treatment DML.
#       LinearDRLearner is a discrete-treatment DR estimator and requires
#       StatsModelsInferenceDiscrete.
#
#   (b) _RecordingRidgeCV + model_final + _HC1SandwichEstimator
#       → TypeError: unexpected keyword argument 'model_final'
#       LinearDRLearner has no model_final param (that's only DRLearner).
#       LinearDRLearner's final stage is always an internal linear model.
#
# Correct approach: use StatsModelsInferenceDiscrete(cov_type='HC1').
# This is the inference class EconML designed for LinearDRLearner and
# produces fast O(n) analytic HC1 CIs — no bootstrap, no custom classes.
# ─────────────────────────────────────────────────────────────────────────────

def fit_linear_drlearner(
    X_feat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> LinearDRLearner:
    """
    Fit a LinearDRLearner with StatsModelsInferenceDiscrete HC1 CIs.

    LinearDRLearner is the discrete-treatment variant of the DR estimator
    whose final stage is an internal OLS/WLS linear model.  EconML provides
    StatsModelsInferenceDiscrete specifically for this class: it computes
    the asymptotic normal variance of the linear CATE coefficients using
    heteroskedasticity-consistent (HC1) sandwich standard errors — the same
    estimator as the semiparametric efficiency bound for the DR estimator.

    This gives valid pointwise CIs in O(n) time after fitting, with no
    bootstrap required.

    Parameters
    ----------
    X_feat : preprocessed feature matrix (n x p)
    Y      : outcome vector (n,)
    T      : treatment arm strings (n,)

    Returns
    -------
    Fitted LinearDRLearner with StatsModelsInferenceDiscrete attached.
    Supports .effect(X, T0, T1) and .effect_interval(X, T0, T1, alpha).
    """
    est = LinearDRLearner(
        model_regression  = make_outcome_model(),
        model_propensity  = make_propensity_model(),
        featurizer        = None,        # linear in X; avoids rank deficiency
                                         # when preprocessed p > n
        fit_cate_intercept= True,        # include intercept in linear CATE
        cv                = N_CV_FOLDS,
        random_state      = RANDOM_STATE,
        categories        = ALL_ARMS,
    )
    print("  [IF]  Fitting linear DR-Learner (StatsModelsInferenceDiscrete HC1) …")
    est.fit(Y, T, X=X_feat,
            inference=StatsModelsInferenceDiscrete(cov_type='HC1'))
    print("  [IF]  Done.")
    return est


# ─────────────────────────────────────────────────────────────────────────────
# 3. CausalForest  (honesty-based GRF CIs, AutoML nuisance)
# ─────────────────────────────────────────────────────────────────────────────

def fit_causal_forest(
    X_feat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> dict:
    """
    Fit one CausalForestDML per treatment arm vs. control.

    v2 change: nuisance models are now AutoML (FLAML) when USE_AUTOML=True.
    The forest itself is unchanged — CF_N_TREES honest trees, GRF variance.

    Parameters
    ----------
    X_feat : preprocessed feature matrix (n x p) — full sample
    Y      : outcome vector (n,)
    T      : treatment arm strings (n,)

    Returns
    -------
    dict {z: (fitted_CausalForestDML, mask_bool)}
    """
    forests   = {}
    mask_ctrl = (T == CONTROL)

    for z in TREATMENTS:
        mask_trt = (T == z)
        mask     = mask_ctrl | mask_trt
        Ys = Y[mask]
        Ts = (T[mask] == z).astype(float)
        Xs = X_feat[mask]

        cf = CausalForestDML(
            model_y          = make_outcome_model(),
            model_t          = GradientBoostingRegressor(
                                   n_estimators=200, max_depth=3,
                                   random_state=RANDOM_STATE),
            n_estimators     = CF_N_TREES,
            min_samples_leaf = 5,
            max_depth        = None,
            cv               = N_CV_FOLDS,
            random_state     = RANDOM_STATE,
        )
        print(f"  [CF]  Fitting CausalForest for '{z}' vs '{CONTROL}' …")
        cf.fit(Ys, Ts, X=Xs, inference='auto')
        forests[z] = (cf, mask)
        print(f"  [CF]  Done '{z}'.")

    return forests
