"""
config.py
All tuneable knobs in one place. Import this module everywhere.
To run a quick experiment, change a value here — nothing else needs touching.

New in v2
---------
USE_AUTOML             : swap stacked sklearn models for FLAML AutoML
AUTOML_TIME_BUDGET_REG : seconds FLAML searches for regression tasks
AUTOML_TIME_BUDGET_CLF : seconds FLAML searches for classification tasks
USE_IF_CI              : add LinearDRLearner with IF-based CIs as a third
                         estimator in the variance-weighted ensemble
N_PARALLEL_DATASETS    : number of datasets to process simultaneously;
                         set to 1 to disable parallel processing
N_BOOT                 : reduced from 300 to 100 because the nonlinear
                         DRLearner is now one of three ensemble members
                         rather than the sole source of CIs
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("curated_data")
OUT_DIR  = Path("submissions")

# ── Competition identifiers ───────────────────────────────────────────────────
TEAM_ID  = "the_unconfounded"
SUBM_ID  = "1"

# ── Treatment arms ────────────────────────────────────────────────────────────
CONTROL    = "a"
TREATMENTS = ["b", "c", "d", "e"]
ALL_ARMS   = [CONTROL] + TREATMENTS

# ── Cross-fitting / inference ─────────────────────────────────────────────────
N_CV_FOLDS = 5
N_BOOT     = 100     # bootstrap replicates — only used when USE_AUTOML=False

# ── AutoML / bootstrap compatibility ─────────────────────────────────────────
# FLAML's internal trial runner is not thread-safe: it returns None when
# called from a joblib thread, causing AttributeError inside tune.run().
# BootstrapInference(n_jobs=-1) spawns threads, which triggers this crash.
#
# When USE_AUTOML=True we therefore skip BootstrapInference on the nonlinear
# DRLearner entirely.  CIs still come from two sources:
#   • LinearDRLearner  — influence-function (sandwich) CIs, O(n), exact
#   • CausalForestDML  — GRF honesty-based variance, exact
# The nonlinear DRL still contributes its point estimate to the ensemble
# via ENSEMBLE_W_AUTOML_NOBOOT (a fixed conservative weight).
#
# When USE_AUTOML=False (sklearn stacked models), sklearn estimators ARE
# thread-safe so BootstrapInference works normally.
AUTOML_BOOTSTRAP      = False   # always False when USE_AUTOML=True; set True
                                 # only if you replace FLAML with a thread-safe
                                 # AutoML backend (e.g. auto-sklearn 2)
ENSEMBLE_W_AUTOML_NOBOOT = 0.25  # conservative weight for the no-CI DRL term
ALPHA      = 0.05    # => 95% intervals

# ── Variance-weighted ensemble ────────────────────────────────────────────────
# Point estimate and CI both computed from:
#   (1) NonlinearDRLearner  — N_BOOT bootstrap CIs
#   (2) LinearDRLearner     — influence-function (sandwich) CIs  [if USE_IF_CI]
#   (3) CausalForestDML     — honesty-based GRF CIs
# Weight for estimator k at observation i: w_k(x_i) = 1 / sigma_k^2(x_i)
# Ensemble variance: sigma_ens^2 = 1 / sum_k w_k(x_i)
USE_IF_CI  = True    # set False to drop the LinearDRLearner (faster, less safe)

# ── AutoML ────────────────────────────────────────────────────────────────────
USE_AUTOML             = True  # False falls back to v1 stacked sklearn models
AUTOML_TIME_BUDGET_REG = 60    # seconds per AutoML regression task
AUTOML_TIME_BUDGET_CLF = 60    # seconds per AutoML classification task
# FLAML searches over LightGBM, XGBoost, RF, ExtraTree, etc. automatically.
# Increase budgets for larger datasets or when runtime permits.

# ── Fallback sklearn hyperparameters (used when USE_AUTOML = False) ───────────
GBM_N_TREES   = 250
RF_N_TREES    = 250
CF_N_TREES    = 500
MIN_LEAF      = 5
MIN_LEAF_CATE = 10

# ── Parallel dataset processing ───────────────────────────────────────────────
N_PARALLEL_DATASETS = 4   # concurrent datasets in run_batch_parallel
                           # set to 1 for sequential (debug-friendly) mode
# When N_PARALLEL_DATASETS > 1, each worker uses INNER_N_JOBS internally
# to avoid CPU oversubscription.  When = 1, n_jobs=-1 is used.
INNER_N_JOBS = -1          # overridden automatically inside run_batch_parallel

RANDOM_STATE = 42
