"""
evaluate.py
Cross-validation diagnostics and causal model evaluation.

v2 additions
------------
compute_gain_curves(icates, T, Y, min_rows, steps)
    Builds fklearn-format DataFrames per arm and returns relative cumulative
    gain curves for each treatment vs. control comparison.

evaluate_aucs(icates, T, Y, min_rows, steps)
    Returns the scalar AURC (area under the relative cumulative gain curve)
    per arm — the primary model-quality metric used in Causal Inference in
    Python / Chapter 7.  Higher = better CATE ranking.

Why relative cumulative gain?
-----------------------------
Sorting individuals by iCATE score and asking "if we treated only the top k%,
how much gain do we capture vs. the average ATE?" gives the relative
cumulative gain curve.  The area under it (AURC) is 0 for a random ranker
and positive for a good ranker — directly measuring whether the iCATEs
correctly identify who benefits most from treatment.

Since the DRLearner uses cross-fitting (K=5 folds), the iCATEs are already
out-of-sample predictions: observation i's iCATE was estimated by a model
trained on folds ≠ fold(i).  We can therefore use all n rows for the curve
without worrying about in-sample overfitting.

Public API (unchanged from v1)
------------------------------
cv_evaluate(X_feat, Y, T)    -> (rmse_mean, rmse_std)

New public API
--------------
compute_gain_curves(icates, T, Y) -> dict {z: pd.DataFrame}
evaluate_aucs(icates, T, Y)       -> dict {z: float}
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

from config import N_CV_FOLDS, RANDOM_STATE, CONTROL, TREATMENTS


# ─────────────────────────────────────────────────────────────────────────────
# v1 — outcome model CV RMSE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def cv_evaluate(
    X_feat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_folds: int = N_CV_FOLDS,
) -> tuple[float, float]:
    """5-fold CV RMSE of a GBM outcome model (light surrogate for speed)."""
    T_ohe = pd.get_dummies(T, drop_first=False).values.astype(float)
    X_aug = np.hstack([X_feat, T_ohe])

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=RANDOM_STATE,
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X_aug, Y,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_mean = float(-scores.mean())
    rmse_std  = float(scores.std())
    print(f"  [CV] Outcome model RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}"
          f"  (σ_Y = {Y.std():.4f}, {n_folds}-fold)")
    return rmse_mean, rmse_std


# ─────────────────────────────────────────────────────────────────────────────
# v2 — relative cumulative gain curves
# ─────────────────────────────────────────────────────────────────────────────

def _build_arm_df(
    z: str,
    icates: dict,
    T: np.ndarray,
    Y: np.ndarray,
) -> pd.DataFrame:
    """
    Build the fklearn-format DataFrame for arm z vs. control.

    Filters to rows where treatment ∈ {CONTROL, z}, creates a binary
    'treatment' column, and uses icates[z] as the 'prediction' score.

    The full iCATE array is used (not just the arm-z rows) because the
    prediction score for a control-arm row is still a valid CATE estimate —
    it answers "how much would this person benefit if assigned to z?"
    """
    mask = (T == CONTROL) | (T == z)
    return pd.DataFrame({
        "treatment":  (T[mask] == z).astype(int),
        "outcome":    Y[mask],
        "prediction": icates[z][mask],
    })


def compute_gain_curves(
    icates: dict,
    T: np.ndarray,
    Y: np.ndarray,
    min_rows: int = 30,
    steps: int = 100,
) -> dict:
    """
    Compute the relative cumulative gain curve for each treatment arm.

    The relative cumulative gain curve answers:
        "If we treat only the top-k fraction (ranked by predicted iCATE),
         how much better is our average effect vs. treating everyone?"

    Interpretation
    --------------
    - x-axis: fraction of population targeted (0 → 1)
    - y-axis: cumulative effect gain relative to the flat ATE baseline
    - A curve above y=0 means the model correctly identifies high-responders
    - The area under this curve (AURC) summarises ranking quality

    Parameters
    ----------
    icates   : dict {z: np.ndarray (n,)} — ensemble iCATE point estimates
    T        : treatment arm array (n,) — raw strings ('a'..'e')
    Y        : outcome array (n,)
    min_rows : minimum arm-z observations before computing a step
    steps    : number of evaluation points along the curve

    Returns
    -------
    dict {z: pd.DataFrame}  — fklearn curve DataFrames per arm.
    The DataFrame columns include at least 'relative_cumulative_gain'
    and 'n_samples' (see fklearn docs for full schema).
    """
    try:
        from fklearn.causal.validation.curves import (
            relative_cumulative_gain_curve,
        )
    except ImportError as exc:
        raise ImportError(
            "fklearn is required for gain curves. "
            "Install with: pip install fklearn"
        ) from exc

    curves = {}
    for z in TREATMENTS:
        df_arm = _build_arm_df(z, icates, T, Y)
        n_trt = (df_arm["treatment"] == 1).sum()
        if n_trt < min_rows:
            print(f"  [AUC] Arm '{z}': only {n_trt} treated obs — skipping curve.")
            continue
        curves[z] = relative_cumulative_gain_curve(
            df=df_arm,
            treatment="treatment",
            outcome="outcome",
            prediction="prediction",
            min_rows=min_rows,
            steps=steps,
        )
        print(f"  [AUC] Gain curve computed for arm '{z}'.")
    return curves


def evaluate_aucs(
    icates: dict,
    T: np.ndarray,
    Y: np.ndarray,
    min_rows: int = 30,
    steps: int = 100,
) -> dict:
    """
    Compute the area under the relative cumulative gain curve (AURC) per arm.

    AURC = ∫ |relative_cumulative_gain(k) - 0| dk

    This is the primary causal ranking metric from Facure (2023),
    Chapter 7: Meta-Learners.  A higher AURC means the model's iCATEs
    more accurately rank individuals from highest to lowest benefit.

    Parameters
    ----------
    icates   : dict {z: np.ndarray (n,)}
    T        : treatment arm array (n,)
    Y        : outcome array (n,)
    min_rows : minimum arm-z obs for a valid computation
    steps    : curve resolution

    Returns
    -------
    dict {z: float}  — AURC per arm (NaN if arm had too few observations).
    """
    try:
        from fklearn.causal.validation.auc import (
            area_under_the_relative_cumulative_gain_curve,
        )
    except ImportError as exc:
        raise ImportError(
            "fklearn is required for AUC computation. "
            "Install with: pip install fklearn"
        ) from exc

    aucs = {}
    for z in TREATMENTS:
        df_arm = _build_arm_df(z, icates, T, Y)
        n_trt = (df_arm["treatment"] == 1).sum()
        if n_trt < min_rows:
            print(f"  [AUC] Arm '{z}': only {n_trt} treated obs — AUC = NaN.")
            aucs[z] = float("nan")
            continue
        aucs[z] = float(
            area_under_the_relative_cumulative_gain_curve(
                df=df_arm,
                treatment="treatment",
                outcome="outcome",
                prediction="prediction",
                min_rows=min_rows,
                steps=steps,
            )
        )
        print(f"  [AUC] Arm '{z}': AURC = {aucs[z]:.4f}")
    return aucs
