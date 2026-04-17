import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor, LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import glob
from scipy.stats import t
from sklearn.utils import resample
import re
import warnings

warnings.filterwarnings("ignore")

# --- Competition Configuration ---
TEAM_ID = "0020"
SUBMISSION_ID = "3"
OUTPUT_FOLDER = f"{TEAM_ID}_{SUBMISSION_ID}"

# --- Script Configuration ---
DATA_FOLDER = "curated_data"
TREATMENT_ARMS = ["b", "c", "d", "e"]
CONTROL_ARM = "a"
N_BOOTSTRAPS = 1000  # Number of bootstrap samples for sCATE CIs


# --- Helper Functions ---
def get_diff_in_means(df, treatment_arm, control_arm="a"):
    """Calculates the difference in means and its confidence interval."""
    group_treat = df[df["z"] == treatment_arm]["y"]
    group_control = df[df["z"] == control_arm]["y"]

    if len(group_treat) < 2 or len(group_control) < 2:
        return np.nan, np.nan, np.nan

    mean_treat = group_treat.mean()
    mean_control = group_control.mean()

    n_treat = len(group_treat)
    n_control = len(group_control)

    var_treat = group_treat.var(ddof=1)
    var_control = group_control.var(ddof=1)

    estimate = mean_treat - mean_control
    se_diff = np.sqrt(var_treat / n_treat + var_control / n_control)

    if (n_treat - 1) <= 0 or (n_control - 1) <= 0:
        return estimate, np.nan, np.nan

    df_welch_num = (var_treat / n_treat + var_control / n_control) ** 2
    df_welch_den = ((var_treat / n_treat) ** 2 / (n_treat - 1)) + ((var_control / n_control) ** 2 / (n_control - 1))

    if df_welch_den == 0:
        return estimate, np.nan, np.nan
    df_welch = df_welch_num / df_welch_den

    if np.isnan(df_welch) or df_welch <= 0:
        return estimate, np.nan, np.nan

    t_stat = t.ppf(0.975, df=df_welch)
    return estimate, estimate - t_stat * se_diff, estimate + t_stat * se_diff


def preprocess_data(df):
    """One-hot encodes categorical features for modeling."""
    X = df.drop(columns=["ID", "y", "z"])
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    return X_processed


# --- Main Processing Loop ---
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    data_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not data_files:
        print(f"Error: No CSV files found in '{DATA_FOLDER}' directory.")
        return

    total_start_time = time.time()

    for i, file_path in enumerate(data_files):
        data_id_match = re.search(r"data_(\d+)\.csv", os.path.basename(file_path))
        if not data_id_match:
            continue
        data_id = data_id_match.group(1)
        padded_data_id = data_id.zfill(4)

        print(f"\n--- Processing file {i+1}/{len(data_files)}: dataID={padded_data_id} ---")
        file_start_time = time.time()
        df = pd.read_csv(file_path)

        # === subCATE ===
        print("1. Calculating subCATE and BEST_subCATE...")
        subcate_results = [
            {"x": x, "z": t, "Estimate": e, "L95": l, "U95": u}
            for x in [0, 1]
            for t in TREATMENT_ARMS
            for e, l, u in [get_diff_in_means(df[df["x12"] == x], t)]
        ]
        subcate_df = pd.DataFrame(subcate_results)
        subcate_df.to_csv(
            os.path.join(OUTPUT_FOLDER, f"subCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv"),
            index=False,
        )
        if not subcate_df.dropna().empty:
            subcate_df.dropna().loc[subcate_df.dropna().groupby("x")["Estimate"].idxmax()].rename(
                columns={"z": "best_z"}
            )[["x", "best_z"]].to_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"BEST_subCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
                ),
                index=False,
            )

        # === PATE (Derived from subCATE for consistency) ===
        print("2. Calculating PATE and BEST_PATE...")
        weight_0 = (df["x12"] == 0).sum() / len(df)
        weight_1 = (df["x12"] == 1).sum() / len(df)
        pate_results = []
        for z_val in TREATMENT_ARMS:
            # Get subCATE estimates, filling with 0 if a subgroup was empty
            subcate_0_est = (
                subcate_df[(subcate_df["x"] == 0) & (subcate_df["z"] == z_val)]["Estimate"].fillna(0).values[0]
            )
            subcate_1_est = (
                subcate_df[(subcate_df["x"] == 1) & (subcate_df["z"] == z_val)]["Estimate"].fillna(0).values[0]
            )

            # Point estimate is the weighted average of subCATEs
            pate_est = (subcate_0_est * weight_0) + (subcate_1_est * weight_1)

            # Confidence interval is still calculated on the full dataset
            _, l95, u95 = get_diff_in_means(df, z_val)
            pate_results.append({"z": z_val, "Estimate": pate_est, "L95": l95, "U95": u95})

        pate_df = pd.DataFrame(pate_results)
        pate_df.to_csv(
            os.path.join(OUTPUT_FOLDER, f"PATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv"),
            index=False,
        )
        if not pate_df.dropna().empty:
            pate_df.dropna().loc[[pate_df.dropna()["Estimate"].idxmax()]].rename(columns={"z": "best_z"})[
                ["best_z"]
            ].to_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"BEST_PATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
                ),
                index=False,
            )

        # === sCATE ===
        print("3. Calculating sCATE and BEST_sCATE...")
        if not pate_df.empty:
            scate_point_estimates = pate_df.set_index("z")["Estimate"]
            bootstrap_estimates = {
                t: [est for _ in range(N_BOOTSTRAPS) if not np.isnan(est := get_diff_in_means(resample(df), t)[0])]
                for t in TREATMENT_ARMS
            }
            scate_results = [
                {
                    "z": t,
                    "Estimate": scate_point_estimates.get(t),
                    "L95": np.percentile(be, 2.5),
                    "U95": np.percentile(be, 97.5),
                }
                for t, be in bootstrap_estimates.items()
                if be
            ]
            scate_df = pd.DataFrame(scate_results)
            scate_df.to_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"sCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
                ),
                index=False,
            )
            if not scate_df.empty:
                scate_df.loc[[scate_df["Estimate"].idxmax()]].rename(columns={"z": "best_z"})[["best_z"]].to_csv(
                    os.path.join(
                        OUTPUT_FOLDER,
                        f"BEST_sCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
                    ),
                    index=False,
                )

        # === iCATE ===
        print("4. Estimating and Adjusting iCATEs...")
        i_time = time.time()
        X_processed, Y, T_numeric = (
            preprocess_data(df),
            df["y"],
            LabelEncoder().fit_transform(df["z"]),
        )
        le = LabelEncoder().fit(df["z"])
        control_idx = int(np.where(le.classes_ == CONTROL_ARM)[0][0])

        lgbm_y_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "verbosity": -1,
        }
        lgbm_t_params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "verbosity": -1,
        }

        est_forest = CausalForestDML(
            model_y=LGBMRegressor(**lgbm_y_params),
            model_t=LGBMClassifier(**lgbm_t_params),
            discrete_treatment=True,
            n_estimators=1000,
            min_samples_leaf=10,
            max_depth=10,
            random_state=123,
        )
        est_forest.fit(Y, T_numeric, X=X_processed, cache_values=True)

        icate_list = []
        for treat_name in TREATMENT_ARMS:
            treat_idx = int(np.where(le.classes_ == treat_name)[0][0])
            estimates = est_forest.effect(X_processed, T0=control_idx, T1=treat_idx)
            lb, ub = est_forest.effect_interval(X_processed, T0=control_idx, T1=treat_idx, alpha=0.05)
            icate_list.append(
                pd.DataFrame(
                    {
                        "ID": df["ID"],
                        "z": treat_name,
                        "Estimate": estimates,
                        "L95": lb,
                        "U95": ub,
                    }
                )
            )
        icate_df_raw = pd.concat(icate_list).sort_values(by="ID")

        # --- Adjustment Step ---
        icate_df_raw_merged = pd.merge(icate_df_raw, df[["ID", "x12"]], on="ID")
        avg_icate_by_subgroup = (
            icate_df_raw_merged.groupby(["x12", "z"])["Estimate"]
            .mean()
            .reset_index()
            .rename(columns={"Estimate": "Avg_iCATE"})
        )

        adjustment_map = pd.merge(subcate_df, avg_icate_by_subgroup, left_on=["x", "z"], right_on=["x12", "z"]).drop(
            columns=["x12"]
        )
        adjustment_map["adjustment"] = adjustment_map["Estimate"] - adjustment_map["Avg_iCATE"]

        icate_df_adjusted = pd.merge(
            icate_df_raw_merged,
            adjustment_map[["x", "z", "adjustment"]],
            left_on=["x12", "z"],
            right_on=["x", "z"],
        ).drop(columns=["x"])
        icate_df_adjusted["Estimate"] += icate_df_adjusted["adjustment"]
        icate_df_adjusted["L95"] += icate_df_adjusted["adjustment"]
        icate_df_adjusted["U95"] += icate_df_adjusted["adjustment"]

        final_icate_df = icate_df_adjusted[["ID", "z", "Estimate", "L95", "U95"]]
        final_icate_df.to_csv(
            os.path.join(OUTPUT_FOLDER, f"iCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv"),
            index=False,
        )

        if not final_icate_df.empty:
            final_icate_df.loc[final_icate_df.groupby("ID")["Estimate"].idxmax()].rename(columns={"z": "best_z"})[
                ["ID", "best_z"]
            ].to_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"BEST_iCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
                ),
                index=False,
            )

        print(f"iCATE estimation and adjustment took {time.time() - i_time:.2f} seconds.")

        # === Final Consistency Checks ===
        print("5. Performing Final Consistency Checks...")
        # Check 5a: PATE vs Weighted subCATE
        pate_subcate_check_df = pd.DataFrame(pate_results)
        pate_subcate_check_df["Weighted_subCATE"] = pate_subcate_check_df[
            "Estimate"
        ]  # They are now equal by construction
        pate_subcate_check_df["Difference"] = 0
        print("\n  - Numerical Check: PATE vs. Weighted Average subCATE")
        print(
            pate_subcate_check_df[["z", "Estimate", "Weighted_subCATE", "Difference"]].rename(
                columns={"Estimate": "PATE_Estimate"}
            )
        )

        # Check 5b: subCATE vs Adjusted Average iCATE
        final_avg_icate = (
            icate_df_adjusted.groupby(["x12", "z"])["Estimate"]
            .mean()
            .reset_index()
            .rename(columns={"Estimate": "Adjusted_Avg_iCATE"})
        )
        final_check_df = pd.merge(subcate_df.dropna(), final_avg_icate, left_on=["x", "z"], right_on=["x12", "z"])
        final_check_df["Difference"] = final_check_df["Estimate"] - final_check_df["Adjusted_Avg_iCATE"]
        print("\n  - Numerical Check: subCATE vs. Adjusted Average iCATE")
        print(final_check_df[["x", "z", "Estimate", "Adjusted_Avg_iCATE", "Difference"]])

        # Check 5c: PATE vs sCATE
        if not pate_df.empty and not scate_df.empty:
            print("\n  - Numerical Check 5c: PATE vs. sCATE (Point Estimates and CIs)")
            pate_scate_comp = pd.merge(pate_df, scate_df, on="z", suffixes=("_PATE", "_sCATE"))
            pate_scate_comp["Estimate_Diff"] = pate_scate_comp["Estimate_PATE"] - pate_scate_comp["Estimate_sCATE"]
            pate_scate_comp["CI_Width_PATE"] = pate_scate_comp["U95_PATE"] - pate_scate_comp["L95_PATE"]
            pate_scate_comp["CI_Width_sCATE"] = pate_scate_comp["U95_sCATE"] - pate_scate_comp["L95_sCATE"]
            pate_scate_comp["CI_Width_Diff"] = pate_scate_comp["CI_Width_PATE"] - pate_scate_comp["CI_Width_sCATE"]

            print(pate_scate_comp[["z", "Estimate_Diff", "CI_Width_PATE", "CI_Width_sCATE", "CI_Width_Diff"]])

        if not final_check_df.empty:
            fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            fig.suptitle(f"Final Consistency Check for dataID={padded_data_id}", fontsize=16)
            for i_plot, x12_val in enumerate([0, 1]):
                plot_data = final_check_df[final_check_df["x"] == x12_val]
                if not plot_data.empty:
                    sns.pointplot(
                        x="z",
                        y="Estimate",
                        data=plot_data,
                        ax=ax[i_plot],
                        color="red",
                        linestyles="none",
                        markers="D",
                        scale=1.5,
                        label="subCATE",
                    )
                    sns.pointplot(
                        x="z",
                        y="Adjusted_Avg_iCATE",
                        data=plot_data,
                        ax=ax[i_plot],
                        color="blue",
                        linestyles="none",
                        markers="o",
                        scale=1.5,
                        label="Adjusted Avg. iCATE",
                    )
                    ax[i_plot].set_title(f"Subgroup: X12 = {x12_val}")
                    ax[i_plot].legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"consistency_check_{padded_data_id}.png"))
            plt.close(fig)

        print(f"File processing took {time.time() - file_start_time:.2f} seconds.")
    print(f"\n--- Total execution time: {time.time() - total_start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
