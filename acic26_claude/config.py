"""
config.py
All tuneable knobs in one place. Import this module everywhere.

v4 changes
----------
- USE_AUTOML now defaults to False. A fixed LightGBM-primary stack is
  4-6x faster than sklearn GBM and comparably accurate, removing the
  need for FLAML's unpredictable search overhead in most cases. Set
  True only when you have a large time budget and want FLAML to search
  beyond the LightGBM defaults.
- LightGBM hyperparameter blocks replace the old GBM_N_TREES / RF_N_TREES
  blocks. LightGBM's leaf-wise growth + GOSS + EFB gives sklearn-GBM-
  quality predictions at roughly 5-10x the speed.
- N_BOOT reduced to 50. With LightGBM nuisance models the pseudo-outcomes
  are cleaner, so fewer bootstrap replicates are needed to stabilise CIs.
  The linear DRL (influence-function) and CausalForest provide independent
  CI estimates that jointly anchor the variance-weighted ensemble.

v5 changes  (from analysis of dr_r_cf_ensemble_var_weighted)
-------------------------------------------------------------
- CF_DISCRETE_TREATMENT = True.  CausalForestDML's arm-vs-control contrasts
  use a 0/1 treatment indicator; setting discrete_treatment=True tells
  the forest to use a classifier for its internal propensity model instead
  of a regressor.  Omitting this flag causes the forest to fit a continuous
  regression on a binary variable, producing biased propensity scores and
  systematically over-smoothed CATE estimates.

- SE_FLOOR_FACTOR = 0.1.  The per-observation inverse-variance weights in
  the ensemble use each estimator's CI width as the SE.  With MIN_SE = 1e-6
  (the prior floor), a single observation with an artificially tight
  bootstrap CI receives ~10^12 times the weight of an average observation,
  collapsing the ensemble to a single estimator at that point.
  The new floor sets SE_k(x_i) >= SE_FLOOR_FACTOR * std(tau_k), which
  caps the maximum precision ratio across observations at (1/0.1)^2 = 100.
  This makes the per-obs weighting robust without abandoning its
  observation-level adaptivity advantage over global variance weighting.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("curated_data")
OUT_DIR = Path("submissions")

# ── Competition identifiers ───────────────────────────────────────────────────
TEAM_ID = "the_unconfounded"
SUBM_ID = "1"

# ── Treatment arms ────────────────────────────────────────────────────────────
CONTROL = "a"
TREATMENTS = ["b", "c", "d", "e"]
ALL_ARMS = [CONTROL] + TREATMENTS

# ── Cross-fitting / inference ─────────────────────────────────────────────────
N_CV_FOLDS = 5
N_BOOT = 50  # LightGBM nuisance → cleaner pseudo-outcomes → fewer reps needed
ALPHA = 0.05  # => 95% intervals

# ── Variance-weighted ensemble ────────────────────────────────────────────────
# Three estimators combined via inverse-variance weighting per observation:
#   (1) NonlinearDRLearner  — LightGBM CATE stage, bootstrap CIs
#   (2) LinearDRLearner     — polynomial features, IF sandwich CIs  [USE_IF_CI]
#   (3) CausalForestDML     — GRF honesty-based variance
USE_IF_CI = True

# When USE_AUTOML=True the nonlinear DRL has no bootstrap CIs (FLAML uses
# multiprocessing, which conflicts with BootstrapInference's threading).
# In that case the DRL point estimate is blended with this fixed weight;
# the CI computation uses only LinearDRL + CausalForest.
ENSEMBLE_W_AUTOML_NOBOOT = 0.25

# ── Model backend ─────────────────────────────────────────────────────────────
# False (default): LightGBM-primary stacked ensemble — fast, reproducible
# True           : FLAML AutoML search — use only with large time budgets
USE_AUTOML = False

# ── LightGBM hyperparameters (used when USE_AUTOML = False) ───────────────────
# Note: feature_name is intentionally NOT set here.  Setting it on the
# sklearn estimator causes LightGBM to warn that the param is ignored
# (it belongs on the Dataset constructor, not the estimator).
# Instead, models.py wraps every LGBMRegressor/LGBMClassifier in a
# Pipeline([("to_numpy", ...), ("lgbm", ...)]) that converts any
# DataFrame to a plain ndarray before it reaches LightGBM, eliminating
# the sklearn "X does not have valid feature names" warning at source.

# Outcome / CATE regression
LGBM_REG = dict(
    n_estimators=300,
    max_depth=-1,  # unlimited depth; controlled by num_leaves
    num_leaves=63,  # 2^6-1; good default for tabular data
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    n_jobs=-1,  # overridden per-worker by main.py
    verbosity=-1,
    random_state=42,
)
# Propensity classification
LGBM_CLS = dict(
    n_estimators=300,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    n_jobs=-1,
    verbosity=-1,
    random_state=42,
)
# CausalForest nuisance (lighter — forest orthogonalisation is less
# sensitive to propensity quality than DR pseudo-outcome construction)
LGBM_REG_CF = dict(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    n_jobs=-1,
    verbosity=-1,
    random_state=42,
)

# ── CausalForest ─────────────────────────────────────────────────────────────
CF_N_TREES = 500

# discrete_treatment=True: use a classifier (not regressor) for the forest's
# internal propensity model.  Correct for binary arm-vs-control contrasts.
# Sourced from dr_r_cf_ensemble_var_weighted — our prior code left this unset,
# causing CausalForestDML to fit a continuous regression on a 0/1 variable.
CF_DISCRETE_TREATMENT = True

# ── Ensemble SE floor ─────────────────────────────────────────────────────────
# Per-observation inverse-variance weights are floored so that
#   SE_k(x_i) >= SE_FLOOR_FACTOR * std(tau_k)
# This prevents any single observation from receiving disproportionate weight
# due to an artificially tight CI from a small bootstrap sample.
# 0.1 caps the max precision ratio across observations at 100:1.
SE_FLOOR_FACTOR = 0.1

# ── AutoML (used only when USE_AUTOML = True) ─────────────────────────────────
AUTOML_TIME_BUDGET_REG = 60
AUTOML_TIME_BUDGET_CLF = 60

# ── Bootstrap / inference ─────────────────────────────────────────────────────
# USE_BOOTSTRAP controls whether the NonlinearDRLearner attaches
# BootstrapInference (N_BOOT refits of the full CATE model).
#
# True  (default): bootstrap CIs from DRLearner + analytic CIs from
#                  LinearDRLearner / CausalForest / ForestDRLearner are all
#                  combined in the variance-weighted ensemble.
#
# False (fast):    DRLearner still contributes its point estimate to the
#                  ensemble (with weight ENSEMBLE_W_AUTOML_NOBOOT), but CIs
#                  come entirely from the three analytic estimators:
#                    - LinearDRLearner  → HC1 sandwich / influence function
#                    - CausalForestDML  → GRF honest variance
#                    - ForestDRLearner  → GRF honest variance
#                  These are closed-form, essentially free after fitting, and
#                  asymptotically valid under their respective assumptions.
#                  Use --no-bootstrap at the CLI to set this at runtime.
USE_BOOTSTRAP = True

# Whether to attach BootstrapInference to the nonlinear DRLearner when
# USE_AUTOML=True.  Default False because FLAML internally spawns its own
# subprocesses and is not safe to call from inside joblib threads
# (BootstrapInference uses joblib).  Set True only if you have confirmed
# that your FLAML version and OS handle nested parallelism without error.
AUTOML_BOOTSTRAP = False

# ── Parallel dataset processing ───────────────────────────────────────────────
N_PARALLEL_DATASETS = 4
INNER_N_JOBS = -1  # overridden automatically inside run_batch_parallel

RANDOM_STATE = 42

# ── Propensity score clipping ─────────────────────────────────────────────────
# DR pseudo-outcome = (1(T=t)/pi_t - 1(T=0)/pi_0) * (Y - mu)
# Near-zero propensities amplify noise by 1/pi — clip before inversion.
# 0.025 is the Crump et al. (2009) standard; 0.01 for looser trimming.
PROPENSITY_CLIP = 0.025

# ── ForestDRLearner ───────────────────────────────────────────────────────────
# 4th ensemble member: DR pseudo-outcomes fed into an honest causal forest.
# Combines DR's double-robustness with GRF's honest CI — no bootstrap needed.
USE_FOREST_DRL = True
FDRL_N_TREES = 500
FDRL_MIN_LEAF = 5
