# ============================================================
# ACIC TOP-TIER PIPELINE (FAST VERSION)
# ============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from zipfile import ZipFile

from joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

TEAM_ID = "1"
SUBMISSION_ID = "1"

INPUT_DIR = "../data/inputs/curated_data"
OUTPUT_DIR = f"../data/outputs/econml_super_learner_team{TEAM_ID}_submission{SUBMISSION_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

treatments = ["a", "b", "c", "d", "e"]
subgroups = [0, 1]
x_col = "x12"

N_JOBS = -1

# silent LightGBM
LGBM_REG = dict(n_estimators=200, max_depth=6, verbosity=-1)
LGBM_CLS = dict(n_estimators=200, verbosity=-1)

# ============================================================
# PREPROCESSING
# ============================================================


def preprocess_covariates(data):

    X = data[[f"x{i}" for i in range(1, 41)]]

    numeric_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    transformer = ColumnTransformer(
        [("num", "passthrough", numeric_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    Xp = transformer.fit_transform(X)

    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    return np.asarray(Xp)


# ============================================================
# SUPER LEARNER NUISANCE MODELS
# ============================================================


def fit_superlearner_regression(X, y):

    models = [LGBMRegressor(**LGBM_REG), RandomForestRegressor(n_estimators=150, max_depth=10), ElasticNet(alpha=0.01)]

    preds = []

    for m in models:
        m.fit(X, y)
        preds.append(m.predict(X))

    return np.mean(np.column_stack(preds), axis=1)


def fit_propensity_model(X, T):

    model = LGBMClassifier(**LGBM_CLS)

    model.fit(X, T)

    return model.predict_proba(X)


# ============================================================
# FAST META-LEARNER CATE
# ============================================================


def estimate_cates_fast(data, X):

    y = data["y"].values

    le = LabelEncoder()
    T = le.fit_transform(data["z"])

    n = len(y)

    # ------------------------------------------------
    # SHARED NUISANCE MODELS
    # ------------------------------------------------

    mu_hat = fit_superlearner_regression(X, y)

    prop = fit_propensity_model(X, T)

    # ------------------------------------------------
    # DR pseudo-outcomes
    # ------------------------------------------------

    tau_list = []

    for t in range(1, 5):

        treat = (T == t).astype(int)
        control = (T == 0).astype(int)

        p_t = prop[:, t]
        p_0 = prop[:, 0]

        y_tilde = treat * (y - mu_hat) / p_t - control * (y - mu_hat) / p_0

        tau_model = LGBMRegressor(**LGBM_REG)

        tau_model.fit(X, y_tilde)

        tau = tau_model.predict(X)

        tau_list.append(tau)

    tau = np.column_stack(tau_list)

    return tau


# ============================================================
# BUILD iCATE DATAFRAME
# ============================================================


def build_iCATE(data, tau):

    ids = data["ID"].values

    tau_full = np.column_stack([np.zeros(len(tau)), tau])

    rows = []

    for i, z in enumerate(treatments):

        rows.append(
            pd.DataFrame({"ID": ids, "z": z, "Estimate": tau_full[:, i], "L95": tau_full[:, i], "U95": tau_full[:, i]})
        )

    return pd.concat(rows, ignore_index=True)


# ============================================================
# SAVE SUBMISSION FILES
# ============================================================


def save_submission_files(dataID, iCATE_df, data):

    files = []

    # --------------------
    # sCATE
    # --------------------

    sCATE_rows = []

    for t in treatments:

        df_t = iCATE_df[iCATE_df["z"] == t]

        sCATE_rows.append(
            {"z": t, "Estimate": df_t["Estimate"].mean(), "L95": df_t["L95"].mean(), "U95": df_t["U95"].mean()}
        )

    sCATE_df = pd.DataFrame(sCATE_rows)

    sfile = f"sCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"
    sCATE_df.to_csv(os.path.join(OUTPUT_DIR, sfile), index=False)

    files.append(sfile)

    # --------------------
    # subCATE
    # --------------------

    sub_rows = []

    for t, x_val in product(treatments, subgroups):

        sub = iCATE_df[(iCATE_df["z"] == t) & (data[x_col] == x_val)]

        sub_rows.append(
            {"z": t, "x": x_val, "Estimate": sub["Estimate"].mean(), "L95": sub["L95"].mean(), "U95": sub["U95"].mean()}
        )

    subCATE_df = pd.DataFrame(sub_rows)

    subfile = f"subCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    subCATE_df.to_csv(os.path.join(OUTPUT_DIR, subfile), index=False)

    files.append(subfile)

    # --------------------
    # PATE
    # --------------------

    PATE_df = sCATE_df.copy()

    pfile = f"PATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    PATE_df.to_csv(os.path.join(OUTPUT_DIR, pfile), index=False)

    files.append(pfile)

    # --------------------
    # iCATE
    # --------------------

    icfile = f"iCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    iCATE_df.to_csv(os.path.join(OUTPUT_DIR, icfile), index=False)

    files.append(icfile)

    # --------------------
    # BEST_iCATE
    # --------------------

    best_i = iCATE_df.pivot(index="ID", columns="z", values="Estimate")

    best_i["best_z"] = best_i[treatments].idxmax(axis=1)

    best_file = f"BEST_iCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    best_i[["best_z"]].to_csv(os.path.join(OUTPUT_DIR, best_file))

    files.append(best_file)

    # --------------------
    # BEST_sCATE
    # --------------------

    best_s = sCATE_df.loc[sCATE_df["Estimate"].idxmax(), "z"]

    best_s_file = f"BEST_sCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    pd.DataFrame({"best_z": [best_s]}).to_csv(os.path.join(OUTPUT_DIR, best_s_file), index=False)

    files.append(best_s_file)

    # --------------------
    # BEST_subCATE
    # --------------------

    best_sub_rows = []

    for x_val in subgroups:

        df_sub = subCATE_df[subCATE_df["x"] == x_val]

        best_z = df_sub.loc[df_sub["Estimate"].idxmax(), "z"]

        best_sub_rows.append({"x": x_val, "best_z": best_z})

    best_sub_file = f"BEST_subCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    pd.DataFrame(best_sub_rows).to_csv(os.path.join(OUTPUT_DIR, best_sub_file), index=False)

    files.append(best_sub_file)

    # --------------------
    # BEST_PATE
    # --------------------

    best_p = PATE_df.loc[PATE_df["Estimate"].idxmax(), "z"]

    best_p_file = f"BEST_PATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    pd.DataFrame({"best_z": [best_p]}).to_csv(os.path.join(OUTPUT_DIR, best_p_file), index=False)

    files.append(best_p_file)

    return files


# ============================================================
# PROCESS DATASETS
# ============================================================

files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))

all_files = []

for file in tqdm(files, desc="Processing datasets"):

    dataID = os.path.basename(file).split("_")[1].split(".")[0]

    data = pd.read_csv(file)

    X = preprocess_covariates(data)

    tau = estimate_cates_fast(data, X)

    iCATE_df = build_iCATE(data, tau)

    new_files = save_submission_files(dataID, iCATE_df, data)

    all_files.extend([os.path.join(OUTPUT_DIR, f) for f in new_files])


# ============================================================
# CREATE ZIP
# ============================================================

zip_filename = f"super_learner_II_team{TEAM_ID}_submission{SUBMISSION_ID}.zip"

with ZipFile(zip_filename, "w") as zipf:

    for filepath in all_files:
        zipf.write(filepath, arcname=os.path.basename(filepath))

print(f"\nAll files packaged into {zip_filename}")
