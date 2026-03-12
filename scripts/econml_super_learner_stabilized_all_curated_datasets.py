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


# ---------------------------------------------
# CONFIG
# ---------------------------------------------

TEAM_ID = "the_unconfounded"
SUBMISSION_ID = "3"

INPUT_DIR = "../data/inputs/curated_data"
OUTPUT_DIR = "../data/outputs/super_learner_stabilized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TREATMENTS = ["b","c","d","e"]

N_BOOT = 60
N_JOBS = -1


# ---------------------------------------------
# CROSS-FITTED SUPER LEARNER
# ---------------------------------------------

def crossfit_superlearner(X, y):

    kf = KFold(n_splits=5, shuffle=True)

    preds = np.zeros(len(y))

    for train, test in kf.split(X):

        Xtr, Xte = X[train], X[test]
        ytr = y[train]

        models = [
            LGBMRegressor(n_estimators=300),
            RandomForestRegressor(n_estimators=200),
            ElasticNet(alpha=0.01)
        ]

        fold_preds = []

        for m in models:

            m.fit(Xtr, ytr)

            fold_preds.append(m.predict(Xte))

        preds[test] = np.mean(fold_preds, axis=0)

    return preds


# ---------------------------------------------
# BOOTSTRAP FOREST
# ---------------------------------------------

def bootstrap_forest(X, y, T):

    n = len(y)

    idx = np.random.choice(n, n, replace=True)

    Xb = X[idx]
    yb = y[idx]
    Tb = T[idx]

    forest = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=200),
        model_t=LGBMClassifier(n_estimators=200),
        n_estimators=400,
        cv=2
    )

    forest.fit(yb, Tb, X=Xb)

    return forest.effect(X)


# ---------------------------------------------
# DATASET PIPELINE
# ---------------------------------------------

def process_dataset(file):

    dataset_id = os.path.basename(file).split("_")[1].split(".")[0]

    df = pd.read_csv(file)

    y = df["y"].values
    z = df["z"]

    X = df[[f"x{i}" for i in range(1,41)]]

    # treatment encoding
    le = LabelEncoder()
    T = le.fit_transform(z)

    # preprocessing
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    transformer = ColumnTransformer([
        ("num","passthrough",num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ])

    Xp = transformer.fit_transform(X)

    # -------------------------------------
    # SUPER LEARNER NUISANCE
    # -------------------------------------

    mu_hat = crossfit_superlearner(Xp, y)

    # -------------------------------------
    # ORTHOGONAL FOREST
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
    # R LEARNER
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
    # HYBRID COMBINATION
    # -------------------------------------

    tau = 0.6 * tau_forest + 0.4 * tau_r

    # -------------------------------------
    # MULTI-TASK STABILIZATION
    # -------------------------------------

    multitask = LGBMRegressor(n_estimators=300)

    multitask.fit(Xp, tau)

    tau = multitask.predict(Xp)

    # -------------------------------------
    # BOOTSTRAP INTERVALS
    # -------------------------------------

    boot = Parallel(n_jobs=N_JOBS)(
        delayed(bootstrap_forest)(Xp,y,T)
        for _ in range(N_BOOT)
    )

    boot = np.stack(boot)

    lb = np.percentile(boot,2.5,axis=0)
    ub = np.percentile(boot,97.5,axis=0)

    # -------------------------------------
    # RANKING STABILIZATION
    # -------------------------------------

    sigma = np.std(tau)

    probs = np.exp(tau/sigma)

    probs /= probs.sum(axis=1,keepdims=True)

    best_idx = np.argmax(probs,axis=1)

    # -------------------------------------
    # WRITE OUTPUTS
    # -------------------------------------

    id_vec = df["ID"].values

    rows = []

    for i,treat in enumerate(TREATMENTS):

        rows.append(pd.DataFrame({
            "ID":id_vec,
            "z":treat,
            "Estimate":tau[:,i],
            "L95":lb[:,i],
            "U95":ub[:,i]
        }))

    ic_df = pd.concat(rows)

    ic_df.to_csv(
        f"{OUTPUT_DIR}/iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # sCATE

    scate = tau.mean(axis=0)

    scate_df = pd.DataFrame({
        "z":TREATMENTS,
        "Estimate":scate,
        "L95":lb.mean(axis=0),
        "U95":ub.mean(axis=0)
    })

    scate_df.to_csv(
        f"{OUTPUT_DIR}/sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # BEST_iCATE

    pd.DataFrame({
        "ID":id_vec,
        "best_z":[TREATMENTS[i] for i in best_idx]
    }).to_csv(
        f"{OUTPUT_DIR}/BEST_iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    best_s = TREATMENTS[np.argmax(scate)]

    pd.DataFrame({"best_z":[best_s]}).to_csv(
        f"{OUTPUT_DIR}/BEST_sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    return dataset_id


# ---------------------------------------------
# RUN PIPELINE
# ---------------------------------------------

files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))

print(f"\nRunning advanced ACIC pipeline on {len(files)} datasets\n")

Parallel(n_jobs=N_JOBS)(
    delayed(process_dataset)(f)
    for f in tqdm(files,desc="Datasets")
)

print("\nAll datasets completed.")