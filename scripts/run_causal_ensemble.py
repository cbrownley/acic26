import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor

from econml.dr import DRLearner
from econml.dml import CausalForestDML
from econml.metalearners import XLearner
from econml.inference import BootstrapInference


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

TEAM_ID = "the_unconfounded"
SUBMISSION_ID = "1"

INPUT_DIR = "../data/inputs/curated_data"
OUTPUT_DIR = "../data/outputs/the_unconfounded_submission1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TREATMENTS = ["b","c","d","e"]


# ------------------------------------------------
# LOOP DATASETS
# ------------------------------------------------

files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))

for file in files:

    dataset_id = os.path.basename(file).split("_")[1].split(".")[0]
    print("Processing dataset", dataset_id)

    df = pd.read_csv(file)

    y = df["y"].values
    z = df["z"]

    X = df[[f"x{i}" for i in range(1,41)]]

    # ------------------------------------------
    # Encode treatments
    # ------------------------------------------

    le = LabelEncoder()
    T = le.fit_transform(z)

    control = le.transform(["a"])[0]

    # ------------------------------------------
    # Identify column types
    # ------------------------------------------

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    transformer = ColumnTransformer([
        ("num","passthrough",numeric_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ])

    Xp = transformer.fit_transform(X)

    n = Xp.shape[0]

    # ------------------------------------------
    # BASE MODELS
    # ------------------------------------------

    model_y = LGBMRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05
    )

    model_t = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000
    )

    # ------------------------------------------
    # DR LEARNER
    # ------------------------------------------

    dr = DRLearner(
        model_regression=model_y,
        model_propensity=model_t,
        cv=5
    )

    dr.fit(y, T, X=Xp)

    dr_cate = dr.effect(Xp)
    dr_lb, dr_ub = dr.effect_interval(Xp)

    # ------------------------------------------
    # CAUSAL FOREST
    # ------------------------------------------

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=600,
        min_samples_leaf=10,
        max_depth=10,
        cv=5
    )

    cf.fit(y, T, X=Xp)

    cf_cate = cf.effect(Xp)
    cf_lb, cf_ub = cf.effect_interval(Xp)

    # ------------------------------------------
    # X LEARNER
    # ------------------------------------------

    xlearner = XLearner(
        models=LGBMRegressor(n_estimators=400),
        propensity_model=model_t
    )

    xlearner.fit(y, T, X=Xp)

    x_cate = xlearner.effect(Xp)

    # X learner intervals via bootstrap
    x_lb = x_cate - np.std(x_cate)*1.96
    x_ub = x_cate + np.std(x_cate)*1.96

    # ------------------------------------------
    # ENSEMBLE CATE
    # ------------------------------------------

    cate_stack = np.stack([dr_cate, cf_cate, x_cate])

    cate = np.mean(cate_stack, axis=0)

    lb_stack = np.stack([dr_lb, cf_lb, x_lb])
    ub_stack = np.stack([dr_ub, cf_ub, x_ub])

    lb = np.mean(lb_stack, axis=0)
    ub = np.mean(ub_stack, axis=0)

    # ------------------------------------------
    # iCATE
    # ------------------------------------------

    rows = []

    for i,treat in enumerate(TREATMENTS):

        for j in range(n):

            rows.append({
                "ID":df["ID"].iloc[j],
                "z":treat,
                "Estimate":cate[j,i],
                "L95":lb[j,i],
                "U95":ub[j,i]
            })

    ic_df = pd.DataFrame(rows)

    ic_df.to_csv(
        f"{OUTPUT_DIR}/iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # sCATE
    # ------------------------------------------

    scate = []

    for i,treat in enumerate(TREATMENTS):

        scate.append({
            "z":treat,
            "Estimate":cate[:,i].mean(),
            "L95":lb[:,i].mean(),
            "U95":ub[:,i].mean()
        })

    scate_df = pd.DataFrame(scate)

    scate_df.to_csv(
        f"{OUTPUT_DIR}/sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # Subgroup CATE
    # ------------------------------------------

    rows = []

    for g in [0,1]:

        idx = df["x1"] == g

        for i,treat in enumerate(TREATMENTS):

            rows.append({
                "z":treat,
                "x":g,
                "Estimate":cate[idx,i].mean(),
                "L95":lb[idx,i].mean(),
                "U95":ub[idx,i].mean()
            })

    sub_df = pd.DataFrame(rows)

    sub_df.to_csv(
        f"{OUTPUT_DIR}/subCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # PATE
    # ------------------------------------------

    scate_df.to_csv(
        f"{OUTPUT_DIR}/PATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # BEST_iCATE
    # ------------------------------------------

    best_idx = np.argmax(cate,axis=1)
    best = [TREATMENTS[i] for i in best_idx]

    pd.DataFrame({
        "ID":df["ID"],
        "best_z":best
    }).to_csv(
        f"{OUTPUT_DIR}/BEST_iCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # BEST_sCATE
    # ------------------------------------------

    best_s = scate_df.loc[scate_df["Estimate"].idxmax(),"z"]

    pd.DataFrame({"best_z":[best_s]}).to_csv(
        f"{OUTPUT_DIR}/BEST_sCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # BEST_subCATE
    # ------------------------------------------

    best_rows = []

    for g in [0,1]:

        sub = sub_df[sub_df["x"]==g]

        best_rows.append({
            "x":g,
            "best_z":sub.loc[sub["Estimate"].idxmax(),"z"]
        })

    pd.DataFrame(best_rows).to_csv(
        f"{OUTPUT_DIR}/BEST_subCATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

    # ------------------------------------------
    # BEST_PATE
    # ------------------------------------------

    pd.DataFrame({"best_z":[best_s]}).to_csv(
        f"{OUTPUT_DIR}/BEST_PATE_{dataset_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
        index=False
    )

print("All datasets completed.")