import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from econml.dml import CausalForestDML
from econml.rlearner import RLearner

from flaml import AutoML

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
TEAM_ID = "the_unconfounded"
SUBMISSION_ID = "2"
INPUT_DIR = "../data/inputs/curated_data"
OUTPUT_DIR = "../data/outputs/causal_forest_rlearner_flaml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TREATMENTS = ["b","c","d","e"]
files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))

# ------------------------------------------------
# FUNCTION: FLAML nuisance model training
# ------------------------------------------------
def flaml_model(X, y, task="regression"):
    automl = AutoML()
    automl_settings = {
        "time_budget": 60,  # seconds per model
        "metric": "rmse" if task=="regression" else "accuracy",
        "task": task,
        "log_file_name": None,
        "verbosity": 0
    }
    automl.fit(X_train=X, y_train=y, **automl_settings)
    return automl.model.estimator

# ------------------------------------------------
# FUNCTION: PROCESS SINGLE DATASET
# ------------------------------------------------
def process_dataset(file):
    dataset_id = os.path.basename(file).split("_")[1].split(".")[0]
    print(f"Processing dataset {dataset_id}")

    df = pd.read_csv(file)
    y = df["y"].values
    z = df["z"]
    X = df[[f"x{i}" for i in range(1,41)]]

    # Encode treatments
    le = LabelEncoder()
    T = le.fit_transform(z)
    control = le.transform(["a"])[0]

    # Identify numeric / categorical columns
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    transformer = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    Xp = transformer.fit_transform(X)
    n = Xp.shape[0]

    # -----------------------------
    # Cross-fitted nuisance models
    # -----------------------------
    X_train, X_val, y_train, y_val, T_train, T_val = train_test_split(
        Xp, y, T, test_size=0.2, random_state=42
    )

    model_y = flaml_model(X_train, y_train, task="regression")
    model_t = flaml_model(X_train, T_train, task="classification")

    # -----------------------------
    # CausalForestDML (multi-treatment)
    # -----------------------------
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=10,
        cv=5,
        random_state=42,
        multi_output=True
    )
    cf.fit(y, T, X=Xp)
    cf_cate = cf.effect(Xp)
    cf_lb, cf_ub = cf.effect_interval(Xp)

    # -----------------------------
    # R-learner hybrid
    # -----------------------------
    rlearner = RLearner(model_regression=model_y, model_propensity=model_t)
    rlearner.fit(Y=y, T=T, X=Xp)
    r_cate = rlearner.effect(Xp)
    r_lb = r_cate - np.std(r_cate)*1.96
    r_ub = r_cate + np.std(r_cate)*1.96

    # -----------------------------
    # Ensemble stacking
    # -----------------------------
    cate_stack = np.stack([cf_cate, r_cate])
    cate_mean = np.mean(cate_stack, axis=0)
    lb_stack = np.stack([cf_lb, r_lb])
    ub_stack = np.stack([cf_ub, r_ub])
    lb_mean = np.mean(lb_stack, axis=0)
    ub_mean = np.mean(ub_stack, axis=0)

    # ------------------------------------------
    # iCATE
    # ------------------------------------------
    rows = []
    for i, treat in enumerate(TREATMENTS):
        for j in range(n):
            rows.append({
                "ID": df["ID"].iloc[j],
                "z": treat,
                "Estimate": cate_mean[j, i],
                "L95": lb_mean[j, i],
                "U95": ub_mean[j, i]
            })
    ic_df = pd.DataFrame(rows)
    ic_df.to_csv(f"{OUTPUT_DIR}/iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # ------------------------------------------
    # sCATE
    # ------------------------------------------
    scate = []
    for i, treat in enumerate(TREATMENTS):
        scate.append({
            "z": treat,
            "Estimate": cate_mean[:, i].mean(),
            "L95": lb_mean[:, i].mean(),
            "U95": ub_mean[:, i].mean()
        })
    scate_df = pd.DataFrame(scate)
    scate_df.to_csv(f"{OUTPUT_DIR}/sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # ------------------------------------------
    # subCATE
    # ------------------------------------------
    rows = []
    for g in [0, 1]:
        idx = df["x1"] == g
        for i, treat in enumerate(TREATMENTS):
            rows.append({
                "z": treat,
                "x": g,
                "Estimate": cate_mean[idx, i].mean(),
                "L95": lb_mean[idx, i].mean(),
                "U95": ub_mean[idx, i].mean()
            })
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(f"{OUTPUT_DIR}/subCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # ------------------------------------------
    # PATE (population-level)
    # ------------------------------------------
    scate_df.to_csv(f"{OUTPUT_DIR}/PATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False)

    # ------------------------------------------
    # BEST metrics
    # ------------------------------------------
    best_idx = np.argmax(cate_mean, axis=1)
    best = [TREATMENTS[i] for i in best_idx]
    pd.DataFrame({"ID": df["ID"], "best_z": best}).to_csv(
        f"{OUTPUT_DIR}/BEST_iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False
    )

    best_s = scate_df.loc[scate_df["Estimate"].idxmax(), "z"]
    pd.DataFrame({"best_z": [best_s]}).to_csv(
        f"{OUTPUT_DIR}/BEST_sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False
    )

    best_rows = []
    for g in [0, 1]:
        sub = sub_df[sub_df["x"] == g]
        best_rows.append({
            "x": g,
            "best_z": sub.loc[sub["Estimate"].idxmax(), "z"]
        })
    pd.DataFrame(best_rows).to_csv(
        f"{OUTPUT_DIR}/BEST_subCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False
    )

    pd.DataFrame({"best_z": [best_s]}).to_csv(
        f"{OUTPUT_DIR}/BEST_PATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv", index=False
    )

# ------------------------------------------------
# RUN ALL DATASETS IN PARALLEL WITH PROGRESS
# ------------------------------------------------
Parallel(n_jobs=-1)(
    delayed(process_dataset)(f) for f in tqdm(files)
)

print("All datasets completed.")