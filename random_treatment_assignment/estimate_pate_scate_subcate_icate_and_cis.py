import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

from helper_functions import (
    get_diff_in_means,
    preprocess_data,
    extract_number,
    get_scate_intervals_from_pate,
)

# --- Competition Configuration ---
TEAM_ID = "0020"
SUBMISSION_ID = "3"  # Increment this for each new submission to avoid overwriting previous results
OUTPUT_FOLDER = f"{TEAM_ID}_{SUBMISSION_ID}"

# --- Script Configuration ---
DATA_FOLDER = "curated_data"
TREATMENT_ARMS = ["b", "c", "d", "e"]
CONTROL_ARM = "a"
N_BOOTSTRAPS = 1000  # Number of bootstrap samples for sCATE CIs


# --- Main Processing Loop ---
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    data_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")), key=extract_number)
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

        # Identify categorical/nominal columns
        # This selects columns with 'object' (strings) or 'category' types
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        # Remove spaces and replace with underscores
        for col in cat_cols:
            df[col] = df[col].str.replace(" ", "_")

        # === PATE ===
        print("1. Calculating PATE and BEST_PATE...")
        print("PATE with Direct Difference in Means:")
        # Method 1. PATE is calculated directly using the get_diff_in_means function, which is robust to unequal variances and sample sizes.
        pate_results = []
        for z_val in TREATMENT_ARMS:
            pate_est_direct, l95, u95 = get_diff_in_means(df, z_val)

            # Append the directly calculated results.
            pate_results.append({"z": z_val, "Estimate": pate_est_direct, "L95": l95, "U95": u95})

        pate_df = pd.DataFrame(pate_results)
        print(pate_df)
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
        print("")

        # === subCATE ===
        print("2. Calculating subCATE and BEST_subCATE...")
        print("subCATE with Direct Difference in Means:")
        subcate_results = [
            {"x": x, "z": t, "Estimate": e, "L95": l, "U95": u}
            for x in [0, 1]
            for t in TREATMENT_ARMS
            for e, l, u in [get_diff_in_means(df[df["x12"] == x], t)]
        ]
        subcate_df = pd.DataFrame(subcate_results)
        print(subcate_df)
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
        print("")

        # === iCATE ===
        print("3. Estimating and Adjusting iCATEs...")
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

        # === sCATE ===
        print("4. Calculating sCATE and BEST_sCATE...")
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
            print("sCATE with Bootstrap CIs:")
            print(scate_df)
            # scate_df.to_csv(
            #     os.path.join(
            #         OUTPUT_FOLDER,
            #         f"sCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
            #     ),
            #     index=False,
            # )
            # if not scate_df.empty:
            #     scate_df.loc[[scate_df["Estimate"].idxmax()]].rename(
            #         columns={"z": "best_z"}
            #     )[["best_z"]].to_csv(
            #         os.path.join(
            #             OUTPUT_FOLDER,
            #             f"BEST_sCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
            #         ),
            #         index=False,
            #     )

            scate_df_final = get_scate_intervals_from_pate(
                pate_df=pate_df,
                final_icate_df=final_icate_df,
                original_df=df,
                treatment_arms=TREATMENT_ARMS,
                control_arm=CONTROL_ARM,
            )

            print("sCATE with iCATE-Adjusted SEs:")
            print(scate_df_final)

            sCATE_filename = os.path.join(
                OUTPUT_FOLDER,
                f"sCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
            )
            scate_df_final.to_csv(sCATE_filename, index=False)

            if not scate_df_final.empty:
                best_scate_df = scate_df_final.loc[[scate_df_final["Estimate"].idxmax()]].rename(
                    columns={"z": "best_z"}
                )[["best_z"]]

                BEST_sCATE_filename = os.path.join(
                    OUTPUT_FOLDER,
                    f"BEST_sCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
                )
                best_scate_df.to_csv(BEST_sCATE_filename, index=False)

        # === Final Consistency Checks ===
        print("5. Performing Final Consistency Checks...")
        # Check 5a: PATE vs Weighted subCATE
        # 1. Calculate the weights for each subgroup based on their proportion in the data
        try:
            weight_x0 = len(df[df["x12"] == 0]) / len(df)
            weight_x1 = 1.0 - weight_x0
        except ZeroDivisionError:
            weight_x0 = 0.5
            weight_x1 = 0.5

        print(f"Weight for subgroup x12=0: {weight_x0:.2f}")
        print(f"Weight for subgroup x12=1: {weight_x1:.2f}\n")

        # 2. Pivot the subCATE DataFrame for easier calculation
        subcate_pivot = subcate_df.pivot_table(index="z", columns="x", values="Estimate")

        # 3. Calculate the weighted average of subCATEs
        subcate_pivot["Weighted_Average_subCATE"] = (subcate_pivot[0] * weight_x0) + (subcate_pivot[1] * weight_x1)

        # 4. Merge with PATE estimates for comparison
        pate_df_indexed = pate_df.set_index("z")
        comparison_df = pate_df_indexed.join(subcate_pivot["Weighted_Average_subCATE"])
        comparison_df = comparison_df.rename(columns={"Estimate": "PATE_Estimate"})

        # 5. Calculate the difference
        comparison_df["Difference"] = comparison_df["PATE_Estimate"] - comparison_df["Weighted_Average_subCATE"]

        # 6. Select and reorder columns for the final output
        final_df = comparison_df[["PATE_Estimate", "Weighted_Average_subCATE", "Difference"]]

        print("Consistency Check: PATE vs. Weighted Average of subCATEs")
        print(final_df)

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
        # Now, perform the comparison with the updated DataFrame
        if not pate_df.empty and not scate_df_final.empty:
            print("\n  - Numerical Check 5c: PATE vs. sCATE (Point Estimates and CIs)")

            # Merge the PATE and the new sCATE results for comparison
            pate_scate_comp = pd.merge(pate_df, scate_df_final, on="z", suffixes=("_PATE", "_sCATE"))

            # The point estimates for PATE and sCATE are identical in this method,
            # so the difference should be zero.
            pate_scate_comp["Estimate_Diff"] = pate_scate_comp["Estimate_PATE"] - pate_scate_comp["Estimate_sCATE"]

            # Calculate the width of the confidence intervals for both
            pate_scate_comp["CI_Width_PATE"] = pate_scate_comp["U95_PATE"] - pate_scate_comp["L95_PATE"]
            pate_scate_comp["CI_Width_sCATE"] = pate_scate_comp["U95_sCATE"] - pate_scate_comp["L95_sCATE"]

            # Calculate the difference in CI widths. A positive value means the sCATE CI is narrower.
            pate_scate_comp["CI_Width_Diff"] = pate_scate_comp["CI_Width_PATE"] - pate_scate_comp["CI_Width_sCATE"]

            print("Comparison of PATE and sCATE Confidence Intervals:")
            print(
                pate_scate_comp[
                    [
                        "z",
                        "CI_Width_PATE",
                        "CI_Width_sCATE",
                        "CI_Width_Diff",
                    ]
                ].round(4)
            )

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
