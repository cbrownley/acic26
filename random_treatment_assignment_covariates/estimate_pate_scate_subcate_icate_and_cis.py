import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from econml.dml import CausalForestDML
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
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
    extract_number,
    preprocess_data,
    preprocess_data_for_bambi,
    get_pate_with_bambi,
    get_subcates_by_filtering,
    get_scate_intervals_from_pate,
    get_diff_in_means,
)

# --- Competition Configuration ---
TEAM_ID = "0020"
SUBMISSION_ID = "4"  # Increment this for each submission to avoid overwriting previous results
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

        covariate_names = [f"x{i}" for i in range(1, 41)]
        X_features = df[covariate_names]
        y_target = df["y"]

        quantitative_features = []
        binary_features = []
        categorical_features = []

        for col in X_features.columns:
            # If a column's dtype is 'object', it's treated as categorical.
            if X_features[col].dtype == "object":
                categorical_features.append(col)
            else:
                # For numeric columns, check if it's binary or quantitative.
                unique_values = X_features[col].unique()
                # Check if column has 2 unique values which are 0 and 1.
                if len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1])):
                    binary_features.append(col)
                else:
                    quantitative_features.append(col)

        print(f"Found {len(quantitative_features)} quantitative features.")
        print(f"Found {len(binary_features)} binary features.")
        print(f"Found {len(categorical_features)} categorical features.")

        # Note: Binary features do not need scaling and can be passed through.
        # The `remainder='passthrough'` handles them correctly after the main transformers.
        preprocessor = ColumnTransformer(
            transformers=[
                ("quant", StandardScaler(), quantitative_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
            ],
            remainder="passthrough",
        )

        # Create the full pipeline with preprocessing and the LASSO model
        lasso_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LassoCV(cv=5, random_state=42, max_iter=10000)),  # Using 5 folds for smaller sample
            ]
        )

        # Run the pipeline
        print("\n--- Running LASSO Variable Selection ---")
        print("Fitting model with Cross-Validation to find best alpha and select features...")
        lasso_pipeline.fit(X_features, y_target)

        # Get the fitted LASSO model from the pipeline
        lasso_model = lasso_pipeline.named_steps["regressor"]

        # Get feature names after preprocessing
        preprocessor_transformer = lasso_pipeline.named_steps["preprocessor"]

        onehot_feature_names = []
        if categorical_features:
            onehot_feature_names = preprocessor_transformer.named_transformers_["cat"].get_feature_names_out(
                categorical_features
            )

        # The passthrough features are simply the binary features, as they were not
        # transformed. This avoids the 'slice' TypeError.
        passthrough_features = binary_features

        # Combine all feature names in the correct order they appear after the transformer
        all_feature_names = quantitative_features + list(onehot_feature_names) + passthrough_features

        coefficients = lasso_model.coef_
        selected_mask = coefficients != 0
        selected_features_encoded = np.array(all_feature_names)[selected_mask]

        # Map selected features back to their original names
        selected_original_features = set()
        for feature_name in selected_features_encoded:
            is_categorical_child = False
            for cat_feat in categorical_features:
                if feature_name.startswith(cat_feat + "_"):
                    selected_original_features.add(cat_feat)
                    is_categorical_child = True
                    break
            if not is_categorical_child:
                selected_original_features.add(feature_name)

        print(f"\nTotal potential covariates (after one-hot encoding): {len(all_feature_names)}")
        print(f"Number of covariates selected by LASSO: {len(selected_original_features)}")

        # # Identify categorical/nominal columns
        # # This selects columns with 'object' (strings) or 'category' types
        # cat_cols = df.select_dtypes(include=['object', 'category']).columns

        # # Remove spaces and replace with underscores
        # for col in cat_cols:
        #     df[col] = df[col].str.replace(' ', '_')

        # === PATE ===
        print("1. Calculating PATE and BEST_PATE...")
        print("Bambi PATE results:")
        # Preprocess the data and run the analysis
        # Create the final list of columns to keep for the Bayesian analysis
        columns_for_bayesian_model = ["ID", "y", "z"] + list(selected_original_features)

        # Create a new, filtered DataFrame containing only these columns
        # It's good practice to create a copy to avoid SettingWithCopyWarning
        filtered_df = df[columns_for_bayesian_model].copy()

        preprocessed_data = preprocess_data_for_bambi(filtered_df)
        covariates = [col for col in preprocessed_data.columns if col not in ["y", "z"]]
        formula = "y ~ z + " + " + ".join(covariates)
        pate_df = get_pate_with_bambi(preprocessed_data, formula, TREATMENT_ARMS)

        print("Generated PATE Estimates:")
        print(pate_df)

        # Write the full results to the PATE output file
        pate_output_path = os.path.join(OUTPUT_FOLDER, f"PATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv")
        pate_df.to_csv(pate_output_path, index=False)

        # Identify the best treatment and write it to the BEST_PATE output file
        if not pate_df.dropna().empty:
            best_pate_df = pate_df.dropna().loc[[pate_df.dropna()["Estimate"].idxmax()]]
            best_pate_renamed = best_pate_df.rename(columns={"z": "best_z"})[["best_z"]]

            best_pate_output_path = os.path.join(
                OUTPUT_FOLDER,
                f"BEST_PATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
            )
            best_pate_renamed.to_csv(best_pate_output_path, index=False)

        # === subCATE ===
        print("2. Calculating subCATE and BEST_subCATE...")
        print("Bambi subCATE results:")
        # Preprocess the data and run the analysis
        columns_for_bayesian_model = list(dict.fromkeys(["ID", "y", "z", "x12"] + list(selected_original_features)))
        filtered_df = df[columns_for_bayesian_model].copy()
        subcate_df = get_subcates_by_filtering(preprocess_data_for_bambi(filtered_df), TREATMENT_ARMS)

        print("Generated subCATE Estimates:")
        print(subcate_df)

        subcate_output_path = os.path.join(OUTPUT_FOLDER, f"subCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv")
        subcate_df.to_csv(subcate_output_path, index=False)

        if not subcate_df.dropna().empty:
            # Find the row with the highest estimate for each x12 group
            best_indices = subcate_df.groupby("x")["Estimate"].idxmax()
            best_subcates_df = subcate_df.loc[best_indices].copy()

            # Rename 'z' to 'best_z' and select only the required columns
            best_subcates_df.rename(columns={"z": "best_z"}, inplace=True)

            best_subcate_output_path = os.path.join(
                OUTPUT_FOLDER,
                f"BEST_subCATE_{padded_data_id}_{TEAM_ID}_{SUBMISSION_ID}.csv",
            )
            best_subcates_df[["x", "best_z"]].to_csv(best_subcate_output_path, index=False)

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
        print("Generated iCATE Estimates.")
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
        final_check_df = pd.merge(
            subcate_df.dropna(),
            final_avg_icate,
            left_on=["x", "z"],
            right_on=["x12", "z"],
        )
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
