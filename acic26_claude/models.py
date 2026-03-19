"""
models.py
Factory functions for nuisance and CATE models.

v2 additions
------------
AutoMLRegressor   : sklearn-compatible FLAML wrapper for regression
AutoMLClassifier  : sklearn-compatible FLAML wrapper for classification
make_automl_outcome_model()     -> AutoMLRegressor
make_automl_propensity_model()  -> AutoMLClassifier
make_automl_cate_model()        -> AutoMLRegressor (shorter budget, more regularised)

All v1 factories are kept so callers can switch via config.USE_AUTOML.

Public API
----------
make_outcome_model()     -> estimator   (automl or stacked sklearn)
make_propensity_model()  -> estimator
make_cate_model()        -> estimator
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    GradientBoostingClassifier, RandomForestClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, ElasticNetCV

from config import (
    USE_AUTOML,
    AUTOML_TIME_BUDGET_REG, AUTOML_TIME_BUDGET_CLF,
    GBM_N_TREES, RF_N_TREES, MIN_LEAF, MIN_LEAF_CATE,
    RANDOM_STATE, INNER_N_JOBS,
)


# =============================================================================
# FLAML AutoML wrappers
# =============================================================================

class AutoMLRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible wrapper around FLAML AutoML for regression.

    FLAML searches LightGBM, XGBoost, RandomForest, ExtraTree, and linear
    models within `time_budget` seconds, returning the best pipeline.

    Parameters
    ----------
    time_budget : int — wall-clock seconds for the search
    metric      : str — FLAML metric; 'rmse' or 'mse' recommended
    seed        : int — random seed passed to FLAML
    n_jobs      : int — parallelism inside FLAML (-1 = all cores)
    """
    def __init__(
        self,
        time_budget: int = AUTOML_TIME_BUDGET_REG,
        metric: str = "rmse",
        seed: int = RANDOM_STATE,
        n_jobs: int = INNER_N_JOBS,
    ):
        self.time_budget = time_budget
        self.metric      = metric
        self.seed        = seed
        self.n_jobs      = n_jobs

    def fit(self, X, y):
        from flaml import AutoML
        self._model = AutoML()
        self._model.fit(
            X, y,
            task        = "regression",
            time_budget = self.time_budget,
            metric      = self.metric,
            seed        = self.seed,
            n_jobs      = self.n_jobs,
            verbose     = 0,
        )
        return self

    def predict(self, X):
        return self._model.predict(X)


class AutoMLClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around FLAML AutoML for classification.

    Used for the propensity model π(Z | X) over the 5 treatment arms.

    Parameters
    ----------
    time_budget : int — wall-clock seconds for the search
    seed        : int — random seed
    n_jobs      : int — parallelism inside FLAML
    """
    def __init__(
        self,
        time_budget: int = AUTOML_TIME_BUDGET_CLF,
        seed: int = RANDOM_STATE,
        n_jobs: int = INNER_N_JOBS,
    ):
        self.time_budget = time_budget
        self.seed        = seed
        self.n_jobs      = n_jobs

    def fit(self, X, y):
        from flaml import AutoML
        self._model = AutoML()
        self._model.fit(
            X, y,
            task        = "classification",
            time_budget = self.time_budget,
            metric      = "log_loss",
            seed        = self.seed,
            n_jobs      = self.n_jobs,
            verbose     = 0,
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# =============================================================================
# AutoML factory functions
# =============================================================================

def make_automl_outcome_model() -> AutoMLRegressor:
    """FLAML outcome model μ(X, Z)."""
    return AutoMLRegressor(time_budget=AUTOML_TIME_BUDGET_REG)


def make_automl_propensity_model() -> AutoMLClassifier:
    """FLAML propensity model π(Z | X)."""
    return AutoMLClassifier(time_budget=AUTOML_TIME_BUDGET_CLF)


def make_automl_cate_model() -> AutoMLRegressor:
    """
    FLAML CATE final-stage model τ(X).

    Uses half the time budget of the outcome model.  The pseudo-outcome
    regression is noisier, so a shorter budget with lighter models is often
    better regularised than an exhaustive search.
    """
    return AutoMLRegressor(time_budget=max(20, AUTOML_TIME_BUDGET_REG // 2))


# =============================================================================
# v1 stacked sklearn factories  (kept as fallback when USE_AUTOML = False)
# =============================================================================

def _make_stacked_outcome_model(rs: int = RANDOM_STATE) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("gbm", GradientBoostingRegressor(
                n_estimators=GBM_N_TREES, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                min_samples_leaf=MIN_LEAF, random_state=rs)),
            ("rf",  RandomForestRegressor(
                n_estimators=RF_N_TREES, max_features="sqrt",
                min_samples_leaf=MIN_LEAF, random_state=rs)),
            ("en",  ElasticNetCV(cv=5, max_iter=5000, random_state=rs)),
        ],
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
        cv=5, n_jobs=INNER_N_JOBS,
    )


def _make_stacked_propensity_model(rs: int = RANDOM_STATE) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("gbm", GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8, random_state=rs)),
            ("rf",  RandomForestClassifier(
                n_estimators=200, max_features="sqrt",
                min_samples_leaf=MIN_LEAF, random_state=rs)),
        ],
        final_estimator=LogisticRegressionCV(
            cv=5, multi_class="multinomial",
            max_iter=2000, random_state=rs),
        cv=5, n_jobs=INNER_N_JOBS,
    )


def _make_stacked_cate_model(rs: int = RANDOM_STATE) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("gbm", GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                min_samples_leaf=MIN_LEAF_CATE, random_state=rs)),
            ("rf",  RandomForestRegressor(
                n_estimators=200, max_features="sqrt",
                min_samples_leaf=MIN_LEAF_CATE, random_state=rs)),
            ("en",  ElasticNetCV(cv=5, max_iter=5000, random_state=rs)),
        ],
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
        cv=5, n_jobs=INNER_N_JOBS,
    )


# =============================================================================
# Unified public factories  (callers always use these three)
# =============================================================================

def make_outcome_model():
    """Return outcome model selected by config.USE_AUTOML."""
    return make_automl_outcome_model() if USE_AUTOML else _make_stacked_outcome_model()


def make_propensity_model():
    """Return propensity model selected by config.USE_AUTOML."""
    return make_automl_propensity_model() if USE_AUTOML else _make_stacked_propensity_model()


def make_cate_model():
    """Return CATE final-stage model selected by config.USE_AUTOML."""
    return make_automl_cate_model() if USE_AUTOML else _make_stacked_cate_model()
