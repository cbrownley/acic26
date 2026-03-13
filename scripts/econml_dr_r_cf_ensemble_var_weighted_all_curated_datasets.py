# ============================================================
# TOP-TIER ACIC PIPELINE
# DR + R + CAUSAL FOREST ENSEMBLE
# WITH META-LEARNING + VARIANCE WEIGHTS
# ============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

from lightgbm import LGBMRegressor, LGBMClassifier

from econml.dml import CausalForestDML

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

TEAM_ID="1"
SUBMISSION_ID="1"

INPUT_DIR="../data/inputs/curated_data"
OUTPUT_DIR=f"../data/outputs/econml_ensemble_var_weighted_team{TEAM_ID}_submission{SUBMISSION_ID}"

os.makedirs(OUTPUT_DIR,exist_ok=True)

treatments=["b","c","d","e"]
control="a"

subgroups=[0,1]
x_col="x12"

SOFTMAX_TEMP=0.5

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_covariates(data):

    X=data[[f"x{i}" for i in range(1,41)]]

    num=X.select_dtypes(include=np.number).columns
    cat=X.select_dtypes(exclude=np.number).columns

    ct=ColumnTransformer([
        ("num","passthrough",num),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat)
    ])

    Xp=ct.fit_transform(X)

    if hasattr(Xp,"toarray"):
        Xp=Xp.toarray()

    return np.asarray(Xp)

# ============================================================
# NUISANCE MODELS
# ============================================================

def fit_mu(X,y):

    models=[
        LGBMRegressor(n_estimators=200,max_depth=6,verbosity=-1),
        RandomForestRegressor(n_estimators=200,max_depth=10),
        ElasticNet(alpha=.01)
    ]

    preds=[]

    for m in models:
        m.fit(X,y)
        preds.append(m.predict(X))

    return np.mean(np.column_stack(preds),axis=1)

def fit_propensity(X,T):

    model=LGBMClassifier(n_estimators=200,verbosity=-1)

    model.fit(X,T)

    return model.predict_proba(X)

# ============================================================
# DR META-LEARNER
# ============================================================

def dr_cate(X,y,T,mu_hat,prop):

    tau_list=[]
    var_list=[]

    for t in range(1,5):

        treat=(T==t).astype(int)
        control=(T==0).astype(int)

        p_t=prop[:,t]
        p_0=prop[:,0]

        pseudo=(
            treat*(y-mu_hat)/p_t -
            control*(y-mu_hat)/p_0
        )

        model=LGBMRegressor(n_estimators=200,verbosity=-1)

        model.fit(X,pseudo)

        tau=model.predict(X)

        tau_list.append(tau)

        var_list.append(np.var(pseudo))

    return np.column_stack(tau_list),np.array(var_list)

# ============================================================
# R LEARNER
# ============================================================

def r_learner(X,y,T,mu_hat):

    tau_list=[]
    var_list=[]

    for t in range(1,5):

        treat=(T==t).astype(int)

        r_y=y-mu_hat

        model=LGBMRegressor(n_estimators=200,verbosity=-1)

        model.fit(X,r_y)

        tau=model.predict(X)

        tau_list.append(tau)

        var_list.append(np.var(r_y))

    return np.column_stack(tau_list),np.array(var_list)

# ============================================================
# CAUSAL FOREST
# ============================================================

def forest_cate(X,y,T):

    tau_list=[]
    var_list=[]

    for t in range(1,5):

        T_bin=(T==t).astype(int)

        cf=CausalForestDML(
            n_estimators=400,
            min_samples_leaf=10,
            discrete_treatment=True
        )

        cf.fit(y,T_bin,X=X)

        tau=cf.effect(X)

        tau_list.append(tau)

        var_list.append(np.var(tau))

    return np.column_stack(tau_list),np.array(var_list)

# ============================================================
# VARIANCE WEIGHTED ENSEMBLE
# ============================================================

def ensemble_tau(models,variances):

    inv_var=[1/(v+1e-6) for v in variances]

    weights=np.array(inv_var)/np.sum(inv_var)

    tau=sum(w*m for w,m in zip(weights,models))

    return tau

# ============================================================
# META LEARNER (CROSS DATASET)
# ============================================================

meta_X=[]
meta_y=[]

def update_meta_features(tau,y):

    meta_X.append(tau.mean(axis=0))
    meta_y.append(y.mean())

def fit_meta_model():

    if len(meta_X)<5:
        return None

    model=Ridge()

    model.fit(np.array(meta_X),np.array(meta_y))

    return model

# ============================================================
# STABLE BEST_z
# ============================================================

def best_treatment(tau):

    tau_full=np.column_stack([np.zeros(len(tau)),tau])

    logits=tau_full/SOFTMAX_TEMP

    probs=np.exp(logits)
    probs/=probs.sum(axis=1,keepdims=True)

    idx=np.argmax(probs,axis=1)

    labels=["a","b","c","d","e"]

    return np.array(labels)[idx],tau_full

# ============================================================
# BUILD iCATE
# ============================================================

def build_iCATE(data,tau):

    rows=[]
    ids=data["ID"].values
    subgroup=data[x_col].values

    for i,t in enumerate(treatments):

        est=tau[:,i]

        se=np.std(est)/np.sqrt(len(est))

        rows.append(pd.DataFrame({
            "ID":ids,
            "z":t,
            "Estimate":est,
            "L95":est-1.96*se,
            "U95":est+1.96*se,
            "x":subgroup
        }))

    return pd.concat(rows,ignore_index=True)

# ============================================================
# SAVE ALL REQUIRED FILES
# ============================================================

def save_outputs(dataID,iCATE_df,data):

    files=[]

    # sCATE
    s_rows=[]

    for t in treatments:

        df=iCATE_df[iCATE_df["z"]==t]

        s_rows.append({
            "z":t,
            "Estimate":df.Estimate.mean(),
            "L95":df.L95.mean(),
            "U95":df.U95.mean()
        })

    sCATE=pd.DataFrame(s_rows)

    sfile=f"sCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    sCATE.to_csv(os.path.join(OUTPUT_DIR,sfile),index=False)

    files.append(sfile)

    # subCATE
    sub_rows=[]

    for t,x_val in product(treatments,subgroups):

        df=iCATE_df[
            (iCATE_df.z==t) &
            (iCATE_df.x==x_val)
        ]

        sub_rows.append({
            "z":t,
            "x":x_val,
            "Estimate":df.Estimate.mean(),
            "L95":df.L95.mean(),
            "U95":df.U95.mean()
        })

    sub=pd.DataFrame(sub_rows)

    subfile=f"subCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    sub.to_csv(os.path.join(OUTPUT_DIR,subfile),index=False)

    files.append(subfile)

    # PATE
    PATE=sCATE.copy()

    pfile=f"PATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    PATE.to_csv(os.path.join(OUTPUT_DIR,pfile),index=False)

    files.append(pfile)

    # iCATE
    icfile=f"iCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    iCATE_df[["ID","z","Estimate","L95","U95"]].to_csv(os.path.join(OUTPUT_DIR,icfile),index=False)

    files.append(icfile)

    # BEST_iCATE
    best=iCATE_df.pivot(index="ID",columns="z",values="Estimate")

    best["best_z"]=best[treatments].idxmax(axis=1)

    bestfile=f"BEST_iCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    best[["best_z"]].to_csv(os.path.join(OUTPUT_DIR,bestfile))

    files.append(bestfile)

    # BEST_sCATE
    best_s=sCATE.loc[sCATE.Estimate.idxmax(),"z"]

    best_s_file=f"BEST_sCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    pd.DataFrame({"best_z":[best_s]}).to_csv(
        os.path.join(OUTPUT_DIR,best_s_file),index=False)

    files.append(best_s_file)

    # BEST_subCATE
    rows=[]

    for x_val in subgroups:

        df=sub[sub.x==x_val]

        best=df.loc[df.Estimate.idxmax(),"z"]

        rows.append({"x":x_val,"best_z":best})

    best_sub_file=f"BEST_subCATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR,best_sub_file),index=False)

    files.append(best_sub_file)

    # BEST_PATE
    best_p=PATE.loc[PATE.Estimate.idxmax(),"z"]

    best_p_file=f"BEST_PATE_data{dataID}_team{TEAM_ID}_submission{SUBMISSION_ID}.csv"

    pd.DataFrame({"best_z":[best_p]}).to_csv(
        os.path.join(OUTPUT_DIR,best_p_file),index=False)

    files.append(best_p_file)

    return files

# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(file):

    dataID=os.path.basename(file).split("_")[1].split(".")[0]

    data=pd.read_csv(file)

    X=preprocess_covariates(data)

    y=data.y.values

    le=LabelEncoder()
    T=le.fit_transform(data.z)

    mu_hat=fit_mu(X,y)
    prop=fit_propensity(X,T)

    dr_tau,dr_var=dr_cate(X,y,T,mu_hat,prop)
    r_tau,r_var=r_learner(X,y,T,mu_hat)
    cf_tau,cf_var=forest_cate(X,y,T)

    tau=ensemble_tau(
        [dr_tau,r_tau,cf_tau],
        [dr_var.mean(),r_var.mean(),cf_var.mean()]
    )

    best_labels,tau_full=best_treatment(tau)

    update_meta_features(tau,y)

    iCATE_df=build_iCATE(data,tau)

    files=save_outputs(dataID,iCATE_df,data)

    return files

# ============================================================
# RUN PIPELINE
# ============================================================

files=sorted(glob.glob(f"{INPUT_DIR}/*.csv"))

all_files=[]

for file in tqdm(files):

    new_files=process_dataset(file)

    all_files.extend([os.path.join(OUTPUT_DIR,f) for f in new_files])

# ============================================================
# CREATE ZIP
# ============================================================

zip_filename=f"econml_ensemble_var_weighted_team{TEAM_ID}_submission{SUBMISSION_ID}.zip"

with ZipFile(zip_filename,"w") as zipf:

    for filepath in all_files:
        zipf.write(filepath,arcname=os.path.basename(filepath))

print("Submission ZIP created:",zip_filename)