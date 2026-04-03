"""
models.py
Factory functions for nuisance and CATE models.

Public API
----------
make_outcome_model()    -> StackingRegressor  (LightGBM + RF + ElasticNet → Ridge)
make_propensity_model() -> ClippedClassifier  (StackingClassifier wrapped with clipping)
make_cate_model()       -> StackingRegressor  (lighter LightGBM stack for CATE stage)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    StackingRegressor,
    StackingClassifier,
)
from sklearn.linear_model import (
    RidgeCV,
    LogisticRegressionCV,
    ElasticNetCV,
)
from lightgbm import LGBMRegressor, LGBMClassifier

from config import (
    USE_AUTOML,
    AUTOML_TIME_BUDGET_REG,
    AUTOML_TIME_BUDGET_CLF,
    LGBM_REG,
    LGBM_CLS,
    LGBM_REG_CF,
    PROPENSITY_CLIP,
    RANDOM_STATE,
    INNER_N_JOBS,
)

# =============================================================================
# Propensity clipping wrapper
# =============================================================================


class ClippedClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper that clips predict_proba output.

    Clips each class probability to [clip, 1-clip] and renormalises so
    probabilities still sum to 1.  The base classifier is fitted without
    modification — only predictions are clipped.

    Parameters
    ----------
    base : unfitted sklearn classifier
    clip : float in (0, 0.5) — symmetric probability floor/ceiling.
           Default from config.PROPENSITY_CLIP (= 0.025), bounding the
           maximum inverse-probability weight in DR pseudo-outcomes at 40.
    """

    def __init__(self, base, clip: float = PROPENSITY_CLIP):
        self.base = base
        self.clip = clip

    def fit(self, X, y):
        self.base.fit(X, y)
        self.classes_ = self.base.classes_ if hasattr(self.base, "classes_") else np.unique(y)
        return self

    def predict(self, X):
        return self.base.predict(X)

    def predict_proba(self, X):
        proba = self.base.predict_proba(X)
        proba = np.clip(proba, self.clip, 1.0 - self.clip)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def get_params(self, deep=True):
        params = {"base": self.base, "clip": self.clip}
        if deep:
            params.update({"base__" + k: v for k, v in self.base.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        clip = params.pop("clip", None)
        if clip is not None:
            self.clip = clip
        base_params = {k[5:]: v for k, v in params.items() if k.startswith("base__")}
        if base_params:
            self.base.set_params(**base_params)
        return self


# =============================================================================
# LightGBM-primary stacked factories
# =============================================================================


def _make_lgbm_outcome_model(rs=RANDOM_STATE):
    lgbm_params = {**LGBM_REG, "random_state": rs, "n_jobs": INNER_N_JOBS}
    return StackingRegressor(
        estimators=[
            ("lgbm", LGBMRegressor(**lgbm_params)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=150, max_features="sqrt", min_samples_leaf=5, n_jobs=INNER_N_JOBS, random_state=rs
                ),
            ),
            ("en", ElasticNetCV(cv=5, max_iter=5000, random_state=rs)),
        ],
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
        cv=5,
        n_jobs=INNER_N_JOBS,
    )


def _make_lgbm_propensity_model(rs=RANDOM_STATE):
    lgbm_params = {**LGBM_CLS, "random_state": rs, "n_jobs": INNER_N_JOBS}
    base = StackingClassifier(
        estimators=[
            ("lgbm", LGBMClassifier(**lgbm_params)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150, max_features="sqrt", min_samples_leaf=5, n_jobs=INNER_N_JOBS, random_state=rs
                ),
            ),
        ],
        final_estimator=LogisticRegressionCV(cv=5, max_iter=2000, random_state=rs),
        cv=5,
        n_jobs=INNER_N_JOBS,
    )
    return ClippedClassifier(base, clip=PROPENSITY_CLIP)


def _make_lgbm_cate_model(rs=RANDOM_STATE):
    lgbm_params = {**LGBM_REG_CF, "random_state": rs, "n_jobs": INNER_N_JOBS}
    return StackingRegressor(
        estimators=[
            ("lgbm", LGBMRegressor(**lgbm_params)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=150, max_features="sqrt", min_samples_leaf=10, n_jobs=INNER_N_JOBS, random_state=rs
                ),
            ),
            ("en", ElasticNetCV(cv=5, max_iter=5000, random_state=rs)),
        ],
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
        cv=5,
        n_jobs=INNER_N_JOBS,
    )


# =============================================================================
# AutoML wrappers  (used only when USE_AUTOML = True)
# =============================================================================


class AutoMLRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, time_budget=AUTOML_TIME_BUDGET_REG, metric="rmse", seed=RANDOM_STATE, n_jobs=INNER_N_JOBS):
        self.time_budget = time_budget
        self.metric = metric
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from flaml import AutoML

        self._model = AutoML()
        self._model.fit(
            X,
            y,
            task="regression",
            time_budget=self.time_budget,
            metric=self.metric,
            seed=self.seed,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        return self

    def predict(self, X):
        return self._model.predict(X)


class AutoMLClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, time_budget=AUTOML_TIME_BUDGET_CLF, seed=RANDOM_STATE, n_jobs=INNER_N_JOBS):
        self.time_budget = time_budget
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from flaml import AutoML

        self._model = AutoML()
        self._model.fit(
            X,
            y,
            task="classification",
            metric="log_loss",
            time_budget=self.time_budget,
            seed=self.seed,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# =============================================================================
# Unified public factories
# =============================================================================


def make_outcome_model():
    if USE_AUTOML:
        return AutoMLRegressor(time_budget=AUTOML_TIME_BUDGET_REG)
    return _make_lgbm_outcome_model()


def make_propensity_model():
    """Always returns a ClippedClassifier regardless of backend."""
    if USE_AUTOML:
        return ClippedClassifier(
            AutoMLClassifier(time_budget=AUTOML_TIME_BUDGET_CLF),
            clip=PROPENSITY_CLIP,
        )
    return _make_lgbm_propensity_model()


def make_cate_model():
    if USE_AUTOML:
        return AutoMLRegressor(time_budget=max(20, AUTOML_TIME_BUDGET_REG // 2))
    return _make_lgbm_cate_model()
