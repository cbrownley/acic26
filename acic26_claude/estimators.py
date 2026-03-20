"""
estimators.py
Fitting the four CATE estimators used in the variance-weighted ensemble.

v6 additions
------------
fit_forest_drlearner(X_feat, Y, T)
    ForestDRLearner: DR pseudo-outcomes fed into an honest causal forest.
    This is the natural 4th estimator — it combines:
      - Double robustness from the DR orthogonalization stage
      - Non-parametric CATE flexibility from an honest causal forest
      - GRF-style CIs without bootstrap
    It is a single, tightly integrated pipeline, unlike our current approach
    of running DRLearner and CausalForestDML separately and averaging.

    Why it adds genuine diversity to the ensemble
    ----------------------------------------------
    DRLearner (estimator 1): DR pseudo-outcomes → LightGBM stack final stage
    LinearDRLearner (est. 2): DR pseudo-outcomes → polynomial linear final stage
    CausalForestDML (est. 3): DML residuals → honest causal forest
    ForestDRLearner (est. 4): DR pseudo-outcomes → honest causal forest

    Estimators 3 and 4 both use honest forests but arrive via different
    orthogonalization paths (DML vs. DR), so their errors are not perfectly
    correlated — the ensemble benefits from both.

v5 fixes retained
-----------------
- discrete_treatment=True on CausalForestDML
- LGBMClassifier as model_t
- ClippedClassifier on all propensity models (via make_propensity_model())
- AUTOML_BOOTSTRAP guard

Public API
----------
fit_drlearner(X_feat, Y, T)           -> (DRLearner, has_ci: bool)
fit_linear_drlearner(X_feat, Y, T)    -> LinearDRLearner
fit_causal_forest(X_feat, Y, T)       -> dict {z: (CausalForestDML, mask)}
fit_forest_drlearner(X_feat, Y, T)    -> ForestDRLearner  [new]
"""

import numpy as np
from lightgbm import LGBMClassifier

from econml.dr import DRLearner, LinearDRLearner, ForestDRLearner
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference, StatsModelsInferenceDiscrete

import config as _config
from models import (
    make_outcome_model,
    make_propensity_model,
    make_cate_model,
)

# Convenience aliases for values that never change at runtime
# (structural constants, not CLI-overridable flags)
ALL_ARMS = _config.ALL_ARMS
CONTROL = _config.CONTROL
TREATMENTS = _config.TREATMENTS
LGBM_CLS = _config.LGBM_CLS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nonlinear DR-Learner  (LightGBM stack, bootstrap CIs)
# ─────────────────────────────────────────────────────────────────────────────


def fit_drlearner(X_feat, Y, T):
    """
    Fit a nonlinear doubly-robust learner with cross-fitting.

    Returns
    -------
    (fitted_DRLearner, has_bootstrap_ci: bool)

    has_bootstrap_ci=True  when _config.USE_BOOTSTRAP=True and USE_AUTOML=False.
    has_bootstrap_ci=False when _config.USE_BOOTSTRAP=False (fast mode — CIs come
                           from LinearDRLearner / CausalForest / ForestDRLearner)
                           or when USE_AUTOML=True (FLAML breaks in threads).
    """
    use_bootstrap = _config.USE_BOOTSTRAP and ((not _config.USE_AUTOML) or _config.AUTOML_BOOTSTRAP)

    est = DRLearner(
        model_regression=make_outcome_model(),
        model_propensity=make_propensity_model(),
        model_final=make_cate_model(),
        cv=_config.N_CV_FOLDS,
        mc_iters=1,
        random_state=_config.RANDOM_STATE,
        categories=ALL_ARMS,
        multitask_model_final=False,
    )
    inference = (
        BootstrapInference(n_bootstrap_samples=_config.N_BOOT, n_jobs=_config.INNER_N_JOBS) if use_bootstrap else None
    )
    tag = "bootstrap" if use_bootstrap else "point-only"
    print(f"  [DRL] Fitting nonlinear DR-Learner ({tag}) …")
    est.fit(Y, T, X=X_feat, inference=inference)
    print("  [DRL] Done.")
    return est, use_bootstrap


# ─────────────────────────────────────────────────────────────────────────────
# 2. Linear DR-Learner  (influence-function / sandwich CIs)
# ─────────────────────────────────────────────────────────────────────────────


def fit_linear_drlearner(X_feat, Y, T):
    """
    Fit a LinearDRLearner with HC1 sandwich (influence-function) CIs.

    Linear in X (no polynomial featurizer) to avoid rank deficiency when
    ordinal-encoded + OHE features push the column count above n/5.
    StatsModelsInferenceDiscrete provides HC1 heteroskedasticity-robust SEs.
    """
    est = LinearDRLearner(
        model_regression=make_outcome_model(),
        model_propensity=make_propensity_model(),
        featurizer=None,
        fit_cate_intercept=True,
        cv=_config.N_CV_FOLDS,
        random_state=_config.RANDOM_STATE,
        categories=ALL_ARMS,
    )
    print("  [IF]  Fitting linear DR-Learner (HC1 sandwich CIs) …")
    est.fit(Y, T, X=X_feat, inference=StatsModelsInferenceDiscrete(cov_type="HC1"))
    print("  [IF]  Done.")
    return est


# ─────────────────────────────────────────────────────────────────────────────
# 3. CausalForest  (DML path, GRF honest CIs)
# ─────────────────────────────────────────────────────────────────────────────


def fit_causal_forest(X_feat, Y, T):
    """
    Fit one CausalForestDML per treatment arm vs. control.

    discrete_treatment=True: uses LGBMClassifier internally (correct for
    binary arm-vs-control contrasts, not a continuous regressor).
    """
    forests = {}
    mask_ctrl = T == CONTROL
    lgbm_cls_params = {**LGBM_CLS, "n_jobs": _config.INNER_N_JOBS, "random_state": _config.RANDOM_STATE}

    for z in TREATMENTS:
        mask_trt = T == z
        mask = mask_ctrl | mask_trt
        Ys = Y[mask]
        Ts = (T[mask] == z).astype(int)
        Xs = X_feat[mask]

        cf = CausalForestDML(
            model_y=make_outcome_model(),
            model_t=LGBMClassifier(**lgbm_cls_params),
            discrete_treatment=_config.CF_DISCRETE_TREATMENT,
            n_estimators=_config.CF_N_TREES,
            min_samples_leaf=5,
            max_depth=None,
            cv=_config.N_CV_FOLDS,
            random_state=_config.RANDOM_STATE,
        )
        print(f"  [CF]  Fitting CausalForest '{z}' vs '{CONTROL}' …")
        cf.fit(Ys, Ts, X=Xs, inference="auto")
        forests[z] = (cf, mask)
        print(f"  [CF]  Done '{z}'.")

    return forests


# ─────────────────────────────────────────────────────────────────────────────
# 4. ForestDRLearner  (DR path + honest forest, GRF CIs)  [v6]
# ─────────────────────────────────────────────────────────────────────────────


def fit_forest_drlearner(X_feat, Y, T):
    """
    Fit a ForestDRLearner: DR pseudo-outcomes into an honest causal forest.

    Ensemble role
    -------------
    This is the 4th ensemble member.  It shares the DR orthogonalization path
    with DRLearner (est. 1) and LinearDRLearner (est. 2), but uses an honest
    causal forest as the final stage instead of LightGBM or ridge regression.
    CausalForestDML (est. 3) also uses an honest forest but via DML
    residuals.  The two forest estimators arrive at the same covariate point
    via different pseudo-outcome constructions, so their errors are partially
    independent — the ensemble gains from both.

    CI method
    ---------
    GRF-style honest variance (same as CausalForestDML).  No bootstrap needed.
    effect_interval() is callable immediately after fit().
    """
    est = ForestDRLearner(
        model_regression=make_outcome_model(),
        model_propensity=make_propensity_model(),
        n_estimators=_config.FDRL_N_TREES,
        min_samples_leaf=_config.FDRL_MIN_LEAF,
        max_depth=None,
        cv=_config.N_CV_FOLDS,
        random_state=_config.RANDOM_STATE,
        categories=ALL_ARMS,
    )
    print("  [FDR] Fitting ForestDRLearner (DR + honest forest, GRF CIs) …")
    est.fit(Y, T, X=X_feat, inference="auto")
    print("  [FDR] Done.")
    return est
