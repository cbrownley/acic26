import pandas as pd
import numpy as np
from econml.dr import DRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from itertools import product
from joblib import Parallel, delayed
import os
from zipfile import ZipFile
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
teamID = "1"
submissionID = "1"
dataIDs = [f"{i:01d}" for i in range(1, 19)]
treatments = ['b', 'c', 'd', 'e']
control = 'a'
subgroups = [0, 1]
x_col = "x12"

output_dir = f"../data/outputs/econml_drlearner_team{teamID}_submission{submissionID}"
os.makedirs(output_dir, exist_ok=True)

n_bootstrap = 100
n_jobs = -1

# ----------------------------
# FAST PREPROCESSING SETUP
# ----------------------------
categorical_features = ['x1','x2','x3','x4','x5','x6','x7']
numeric_features = [f'x{i}' for i in range(1, 41) if f'x{i}' not in categorical_features]

encoder = OneHotEncoder(
    drop='first',
    sparse_output=False,
    handle_unknown='ignore'
)

feature_names = None


def fit_encoder_once(df):
    global feature_names
    encoder.fit(df[categorical_features])
    cat_names = encoder.get_feature_names_out(categorical_features)
    feature_names = list(cat_names) + numeric_features


def preprocess_covariates_fast_numpy(df):

    X_cat = encoder.transform(df[categorical_features])
    X_num = df[numeric_features].to_numpy()

    X = np.hstack((X_cat, X_num))

    return X


# ----------------------------
# BOOTSTRAP FUNCTION
# ----------------------------
def bootstrap_cate(Y, T, X, ids, t, n_bootstrap=200, random_state=123):

    T_bin = (T == t).astype(int)

    dr = DRLearner(
        model_regression=RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10
        ),
        model_propensity=RandomForestClassifier(
            n_estimators=200
        ),
        random_state=random_state
    )

    dr.fit(Y=Y, T=T_bin, X=X)
    cate_hat = dr.effect(X=X)

    n = len(Y)

    def one_bootstrap(b):

        idx = np.random.choice(n, n, replace=True)

        Xb = X[idx]
        Yb = Y[idx]
        Tb = T_bin[idx]

        dr_b = DRLearner(
            model_regression=RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10
            ),
            model_propensity=RandomForestClassifier(
                n_estimators=200
            ),
            random_state=random_state
        )

        dr_b.fit(Y=Yb, T=Tb, X=Xb)

        return dr_b.effect(X=X)

    bootstrap_results = Parallel(n_jobs=n_jobs)(
        delayed(one_bootstrap)(b)
        for b in tqdm(range(n_bootstrap), desc=f"Bootstraps T={t}", leave=False)
    )

    bootstrap_estimates = np.column_stack(bootstrap_results)

    l95 = np.percentile(bootstrap_estimates, 2.5, axis=1)
    u95 = np.percentile(bootstrap_estimates, 97.5, axis=1)

    return pd.DataFrame({
        'ID': ids,
        'z': t,
        'Estimate': cate_hat,
        'L95': l95,
        'U95': u95
    })


# ----------------------------
# ESTIMATE CATES
# ----------------------------
def estimate_cates_parallel(data, X):

    Y = data['y'].values
    T = data['z'].values
    ids = data['ID'].values

    results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_cate)(
            Y, T, X, ids, t, n_bootstrap=n_bootstrap
        )
        for t in treatments
    )

    return pd.concat(results, ignore_index=True)


# ----------------------------
# SAVE SUBMISSION FILES
# ----------------------------
def save_submission_files(dataID, iCATE_df, data):

    files = []

    sCATE_rows = []

    for t in treatments:
        df_t = iCATE_df[iCATE_df['z']==t]

        est = df_t['Estimate'].mean()
        l95 = df_t['L95'].mean()
        u95 = df_t['U95'].mean()

        sCATE_rows.append({
            'z': t,
            'Estimate': est,
            'L95': l95,
            'U95': u95
        })

    sCATE_df = pd.DataFrame(sCATE_rows)

    sCATE_file = f"sCATE_data{dataID}_team{teamID}_{submissionID}.csv"
    sCATE_df.to_csv(os.path.join(output_dir, sCATE_file), index=False)
    files.append(sCATE_file)

    subCATE_rows = []

    for t, x_val in product(treatments, subgroups):

        iCATE_sub = iCATE_df[
            (iCATE_df['z']==t) &
            (data[x_col]==x_val)
        ]

        est = iCATE_sub['Estimate'].mean()
        l95 = iCATE_sub['L95'].mean()
        u95 = iCATE_sub['U95'].mean()

        subCATE_rows.append({
            'z': t,
            'x': x_val,
            'Estimate': est,
            'L95': l95,
            'U95': u95
        })

    subCATE_df = pd.DataFrame(subCATE_rows)

    subCATE_file = f"subCATE_data{dataID}_team{teamID}_{submissionID}.csv"
    subCATE_df.to_csv(os.path.join(output_dir, subCATE_file), index=False)
    files.append(subCATE_file)

    PATE_rows = []

    for t in treatments:

        df_t = iCATE_df[iCATE_df['z']==t]

        est = df_t['Estimate'].mean()
        l95 = df_t['L95'].mean()
        u95 = df_t['U95'].mean()

        PATE_rows.append({
            'z': t,
            'Estimate': est,
            'L95': l95,
            'U95': u95
        })

    PATE_df = pd.DataFrame(PATE_rows)

    PATE_file = f"PATE_data{dataID}_team{teamID}_{submissionID}.csv"
    PATE_df.to_csv(os.path.join(output_dir, PATE_file), index=False)
    files.append(PATE_file)

    iCATE_file = f"iCATE_data{dataID}_team{teamID}_{submissionID}.csv"
    iCATE_df.to_csv(os.path.join(output_dir, iCATE_file), index=False)
    files.append(iCATE_file)

    best_iCATE = iCATE_df.pivot(index='ID', columns='z', values='Estimate')
    best_iCATE['best_z'] = best_iCATE[treatments].idxmax(axis=1)

    best_iCATE_file = f"BEST_iCATE_data{dataID}_team{teamID}_{submissionID}.csv"
    best_iCATE[['best_z']].to_csv(os.path.join(output_dir, best_iCATE_file))
    files.append(best_iCATE_file)

    best_sCATE = sCATE_df.loc[sCATE_df['Estimate'].idxmax(),'z']
    best_sCATE_file = f"BEST_sCATE_data{dataID}_team{teamID}_{submissionID}.csv"

    pd.DataFrame({'best_z':[best_sCATE]}).to_csv(
        os.path.join(output_dir, best_sCATE_file),
        index=False
    )

    files.append(best_sCATE_file)

    best_sub_rows = []

    for x_val in subgroups:

        df_sub = subCATE_df[subCATE_df['x']==x_val]

        best_z = df_sub.loc[df_sub['Estimate'].idxmax(),'z']

        best_sub_rows.append({
            'x':x_val,
            'best_z':best_z
        })

    best_sub_file = f"BEST_subCATE_data{dataID}_team{teamID}_{submissionID}.csv"

    pd.DataFrame(best_sub_rows).to_csv(
        os.path.join(output_dir, best_sub_file),
        index=False
    )

    files.append(best_sub_file)

    best_PATE = PATE_df.loc[PATE_df['Estimate'].idxmax(),'z']

    best_PATE_file = f"BEST_PATE_data{dataID}_team{teamID}_{submissionID}.csv"

    pd.DataFrame({'best_z':[best_PATE]}).to_csv(
        os.path.join(output_dir, best_PATE_file),
        index=False
    )

    files.append(best_PATE_file)

    return files


# ----------------------------
# PROCESS DATASETS
# ----------------------------
all_files = []

first_data = pd.read_csv(
    f"../data/inputs/curated_data/data_{dataIDs[0]}.csv"
)

fit_encoder_once(first_data)

for dataID in tqdm(dataIDs, desc="Processing datasets"):

    data = pd.read_csv(
        f"../data/inputs/curated_data/data_{dataID}.csv"
    )

    X = preprocess_covariates_fast_numpy(data)

    iCATE_df = estimate_cates_parallel(data, X)

    files = save_submission_files(dataID, iCATE_df, data)

    all_files.extend(
        [os.path.join(output_dir,f) for f in files]
    )


# ----------------------------
# CREATE ZIP
# ----------------------------
zip_filename = f"team{teamID}_submission{submissionID}.zip"

with ZipFile(zip_filename,'w') as zipf:

    for foldername, subfolders, filenames in os.walk(output_dir):

        for filename in filenames:

            filepath = os.path.join(foldername, filename)

            zipf.write(filepath, arcname=filename)

print(f"All files packaged into {zip_filename}")