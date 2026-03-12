import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor, LGBMClassifier

from econml.dml import CausalForestDML, NonParamDML

# -------------------------------
# CONFIG
# -------------------------------
TEAM_ID = "the_unconfounded"
SUBMISSION_ID = "1"

INPUT_DIR = "../data/inputs/curated_data"
OUTPUT_DIR = f"../data/outputs/super_learner_stabilized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TREATMENTS = ["b", "c", "d", "e"]
N_BOOT = 100      # bootstrap iterations
N_JOBS = -1       # parallel jobs (-1 uses all cores)
N_FOLDS = 5       # cross-fitting folds

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

def crossfit_superlearner(X, y, n_folds=N_FOLDS):
    """Cross-fitted super learner for nuisance estimation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]
        models = [
            LGBMRegressor(n_estimators=300, max_depth=6),
            RandomForestRegressor(n_estimators=200, max_depth=8),
            ElasticNet(alpha=0.01)
        ]
        fold_preds = np.column_stack([m.fit(Xtr, ytr).predict(Xte) for m in models])
        preds[test_idx] = fold_preds.mean(axis=1)
    return preds

def bootstrap_forest(X, y, T, n_estimators=400):
    """Bootstrap sample for uncertainty intervals with CausalForestDML."""
    n = len(y)
    idx = np.random.choice(n, n, replace=True)
    Xb, yb, Tb = X[idx], y[idx], T[idx]
    forest = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=200),
        model_t=LGBMClassifier(n_estimators=200),
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_depth=15,
        cv=2
    )
    forest.fit(yb, Tb, X=Xb)
    return forest.effect(X)

# -------------------------------
# PROCESS SINGLE DATASET
# -------------------------------

def process_dataset(file):
    dataset_id = os.path.basename(file).split("_")[1].split(".")[0]
    df = pd.read_csv(file)

    y = df["y"].values
    z = df["z"]
    X = df[[f"x{i}" for i in range(1, 41)]]

    # Treatment encoding
    le = LabelEncoder()
    T = le.fit_transform(z)

    # Covariate preprocessing
    numeric_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    transformer = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    Xp = transformer.fit_transform(X)

    # -------------------------------------
    # Cross-fitted super learner nuisance
    # -------------------------------------
    mu_hat = crossfit_superlearner(Xp, y)

    # -------------------------------------
    # Orthogonal forest
    # -------------------------------------
    forest = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=400),
        model_t=LGBMClassifier(n_estimators=400),
        n_estimators=800,
        min_samples_leaf=10,
        max_depth=15,
        cv=2
    )
    forest.fit(y, T, X=Xp)
    tau_forest = forest.effect(Xp)

    # -------------------------------------
    # R-learner
    # -------------------------------------
    rlearner = NonParamDML(
        model_y=LGBMRegressor(n_estimators=300),
        model_t=LGBMClassifier(n_estimators=300),
        model_final=LGBMRegressor(n_estimators=300),
        cv=2
    )
    rlearner.fit(y, T, X=Xp)
    tau_r = rlearner.effect(Xp)

    # -------------------------------------
    # Hybrid CATE
    # -------------------------------------
    tau = 0.6 * tau_forest + 0.4 * tau_r

    # -------------------------------------
    # Multi-task LightGBM stabilization
    # -------------------------------------
    multitask = LGBMRegressor(n_estimators=300)
    multitask.fit(Xp, tau)
    tau = multitask.predict(Xp)

    # -------------------------------------
    # Bootstrap intervals
    # -------------------------------------
    boot = Parallel(n_jobs=N_JOBS)(
        delayed(bootstrap_forest)(Xp, y, T) for _ in range(N_BOOT)
    )
    boot = np.stack(boot)
    lb = np.percentile(boot, 2.5, axis=0)
    ub = np.percentile(boot, 97.5, axis=0)

    # -------------------------------------
    # Ranking stabilization
    # -------------------------------------
    sigma = np.std(tau)
    probs = np.exp(tau / sigma)
    probs /= probs.sum(axis=1, keepdims=True)
    best_idx = np.argmax(probs, axis=1)

    # -------------------------------------
    # WRITE ESTIMANDS & BEST FILES
    # -------------------------------------
    id_vec = df["ID"].values
    rows = []
    for i, treat in enumerate(TREATMENTS):
        rows.append(pd.DataFrame({
            "ID": id_vec,
            "z": treat,
            "Estimate": tau[:, i],
            "L95": lb[:, i],
            "U95": ub[:, i]
        }))
    ic_df = pd.concat(rows)
    ic_df.to_csv(f"{OUTPUT_DIR}/iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # Sample average CATE (sCATE)
    scate = tau.mean(axis=0)
    scate_df = pd.DataFrame({
        "z": TREATMENTS,
        "Estimate": scate,
        "L95": lb.mean(axis=0),
        "U95": ub.mean(axis=0)
    })
    scate_df.to_csv(f"{OUTPUT_DIR}/sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # Subgroup CATE (subCATE) on x1
    sub_rows = []
    for g in [0,1]:
        idx = df["x1"] == g
        for i, treat in enumerate(TREATMENTS):
            sub_rows.append({
                "z": treat,
                "x": g,
                "Estimate": tau[idx, i].mean(),
                "L95": lb[idx, i].mean(),
                "U95": ub[idx, i].mean()
            })
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(f"{OUTPUT_DIR}/subCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # Population average treatment effect (PATE)
    pate_df = pd.DataFrame({
        "z": TREATMENTS,
        "Estimate": tau.mean(axis=0),
        "L95": lb.mean(axis=0),
        "U95": ub.mean(axis=0)
    })
    pate_df.to_csv(f"{OUTPUT_DIR}/PATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # BEST_iCATE
    pd.DataFrame({
        "ID": id_vec,
        "best_z": [TREATMENTS[i] for i in best_idx]
    }).to_csv(f"{OUTPUT_DIR}/BEST_iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # BEST_sCATE
    best_s = TREATMENTS[np.argmax(scate)]
    pd.DataFrame({"best_z": [best_s]}).to_csv(f"{OUTPUT_DIR}/BEST_sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # BEST_subCATE
    best_sub_rows = []
    for g in [0,1]:
        sub = sub_df[sub_df["x"] == g]
        best_sub_rows.append({
            "x": g,
            "best_z": sub.loc[sub["Estimate"].idxmax(), "z"]
        })
    pd.DataFrame(best_sub_rows).to_csv(f"{OUTPUT_DIR}/BEST_subCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # BEST_PATE
    pd.DataFrame({"best_z": [best_s]}).to_csv(f"{OUTPUT_DIR}/BEST_PATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    return dataset_id

# -------------------------------
# RUN PIPELINE ON ALL FILES
# -------------------------------

files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))
print(f"\nProcessing {len(files)} datasets with advanced ACIC pipeline...\n")

Parallel(n_jobs=N_JOBS)(
    delayed(process_dataset)(f) for f in tqdm(files, desc="Datasets")
)

print("\nAll datasets completed successfully.")