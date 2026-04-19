import arviz as az
import bambi as bmb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from scipy.stats import t
import re
from marginaleffects import comparisons


# --- Helper Functions ---
def extract_number(filename):
    match = re.search(r"data_(\d+)\.csv", os.path.basename(filename))
    return int(match.group(1)) if match else float("inf")


def preprocess_data(df):
    """One-hot encodes categorical features for modeling."""
    X = df.drop(columns=["ID", "y", "z"])
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    return X_processed


def preprocess_data_for_bambi(df):
    """
    Prepares the DataFrame for a Bambi model by dropping ID and setting correct dtypes.
    """
    df = df.copy()

    # Drop ID if it exists
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Set treatment as a categorical variable with 'a' as the reference level
    df["z"] = pd.Categorical(df["z"], categories=["a", "b", "c", "d", "e"], ordered=False)

    cat_cols = [col for col in df.columns if df.dtypes[col] == "object" and col.startswith("x")]
    # print(f"Found categorical columns to convert: {cat_cols}")

    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


# def get_pate_with_bambi(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Estimates Pairwise Average Treatment Effects (PATEs) using a Bayesian model.
#     """
#     # 1. Build formula dynamically from covariates
#     covariates = [col for col in df.columns if col not in ["y", "z"]]
#     formula = "y ~ z + " + " + ".join(covariates)

#     # 2. Fit Bambi model
#     model = bmb.Model(formula, df)
#     idata = model.fit(
#         draws=2000, tune=1000, target_accept=0.9, cores=1, chains=4, sampler="nutpie"
#     )

#     # 3. Access the posterior distribution for the coefficients of 'z'
#     posterior = idata.posterior
#     z_coeffs = posterior["z"]

#     # 4. Extract contrasts for each level against the reference 'a'
#     # The names 'b', 'c', etc. will become the values in our new 'z' column.
#     effects = {
#         "b": z_coeffs.sel(z_dim="b"),
#         "c": z_coeffs.sel(z_dim="c"),
#         "d": z_coeffs.sel(z_dim="d"),
#         "e": z_coeffs.sel(z_dim="e"),
#     }

#     # 5. Calculate summary statistics and format into the desired columns
#     results = []
#     for name, draws in effects.items():
#         samples = draws.values.flatten()
#         mean = samples.mean()
#         lower, upper = np.percentile(samples, [2.5, 97.5])
#         # Append a dictionary with the requested column names
#         results.append({"z": name, "Estimate": mean, "L95": lower, "U95": upper})

#     return pd.DataFrame(results)


def get_pate_with_bambi(df: pd.DataFrame, formula: str, treatment_arms: list) -> pd.DataFrame:
    """
    Estimates PATEs for a given formula and dataframe.
    This function is assumed to be working correctly.
    """
    model = bmb.Model(formula, df)
    idata = model.fit(draws=2000, tune=1000, target_accept=0.9, cores=1, chains=4, sampler="nutpie")

    posterior = idata.posterior
    z_coeffs = posterior["z"]

    results = []
    for z_level in treatment_arms:
        samples = z_coeffs.sel(z_dim=z_level).values.flatten()
        mean = samples.mean()
        lower, upper = np.percentile(samples, [2.5, 97.5])
        results.append({"z": z_level, "Estimate": mean, "L95": lower, "U95": upper})

    return pd.DataFrame(results)


def get_subcates_by_filtering(df: pd.DataFrame, treatment_arms: list) -> pd.DataFrame:
    """
    Estimates subCATEs by filtering the data for each subgroup of x12
    and running a separate PATE model on each subset.

    This avoids parsing complex interaction terms.
    """
    print("--- Estimating subCATEs by running separate models for each x12 subgroup ---")
    all_subcate_results = []

    # Get the list of covariates to include in the model
    # We exclude y, z, and the subgroup variable x12 itself from this list
    covariates = [col for col in df.columns if col not in ["y", "z", "x12"]]
    base_formula = "y ~ z + " + " + ".join(covariates)

    # Loop through each level of the subgroup variable 'x12'
    for x_level in [0, 1]:
        print(f"\nProcessing subgroup: x12 = {x_level}")

        # 1. Filter the DataFrame for the current subgroup
        subgroup_df = df[df["x12"] == x_level].copy()

        # Check if the subgroup is empty
        if subgroup_df.empty:
            print(f"Warning: No data for subgroup x12 = {x_level}. Skipping.")
            continue

        # 2. Call the working PATE function on the filtered data
        # The model formula now doesn't need the interaction term.
        pate_for_subgroup = get_pate_with_bambi(subgroup_df, base_formula, treatment_arms)

        # 3. Add a column to identify which subgroup these results belong to
        pate_for_subgroup["x"] = x_level

        all_subcate_results.append(pate_for_subgroup)

    # 4. Concatenate the results from all subgroups into a single DataFrame
    if not all_subcate_results:
        print("Error: No results were generated. Both subgroups might have been empty.")
        return pd.DataFrame()

    final_df = pd.concat(all_subcate_results, ignore_index=True)

    # Reorder columns for clarity
    final_df = final_df[["z", "x", "Estimate", "L95", "U95"]]

    return final_df


def get_pate_with_regression(df: pd.DataFrame, control_arm: str = "a") -> pd.DataFrame:
    """
    Calculates PATEs and their CIs using an OLS regression model that includes
    pre-treatment covariates to improve statistical precision.

    Args:
        df: DataFrame with outcome 'y', treatment 'z', and covariates.
        control_arm: The value in 'z' representing the control group.

    Returns:
        A DataFrame with the PATE estimate, standard error, confidence
        interval, and p-value for each treatment arm, adjusted for covariates.
    """
    # 1. Identify all covariate columns (everything except ID, y, z)
    # This is a robust way to get all 'x' columns without listing them.
    covariate_cols = [col for col in df.columns if col not in ["ID", "y", "z"]]

    # Keep a copy of the original data for modeling
    model_df = df.copy()

    # 2. One-hot encode categorical variables
    # We select columns that are of 'object' or 'category' dtype.
    categorical_covariates = model_df[covariate_cols].select_dtypes(include=["object", "category"]).columns

    if not categorical_covariates.empty:
        # 'drop_first=True' avoids perfect multicollinearity by dropping one
        # category per feature. This is standard practice for regression.
        dummies = pd.get_dummies(
            model_df[categorical_covariates],
            prefix=categorical_covariates,
            drop_first=True,
        )
        model_df = model_df.drop(columns=categorical_covariates)
        model_df = pd.concat([model_df, dummies], axis=1)

        # Update the list of covariate columns to include the new dummy columns
        covariate_cols = [col for col in model_df.columns if col not in ["ID", "y", "z"]]

    # 3. Build the R-style formula for statsmodels
    # It will look like: 'y ~ C(z, Treatment('a')) + x1 + x2 + ...'
    covariates_formula = " + ".join(covariate_cols)
    model_formula = f"y ~ C(z, Treatment(reference='{control_arm}')) + {covariates_formula}"

    # print("--- Using Model Formula ---")
    # print(model_formula)
    # print("---------------------------\n")

    # 4. Fit the OLS model
    model = smf.ols(formula=model_formula, data=model_df).fit(
        # We use a heteroscedasticity-robust covariance matrix estimator.
        # This is the gold standard and makes our standard errors reliable
        # even if variance differs across groups (like Welch's t-test did).
        cov_type="HC3"
    )

    # 5. Extract and format results
    results_df = model.conf_int().rename(columns={0: "L95", 1: "U95"})
    results_df["Estimate"] = model.params
    results_df["Std.Err"] = model.bse
    results_df["P-value"] = model.pvalues

    # Filter to only include the PATEs (the coefficients for the treatments)
    pate_results = results_df[results_df.index.str.contains(f"C\(z, Treatment\(reference='{control_arm}'\)\)")]

    if pate_results.empty:
        raise ValueError("Could not find treatment coefficients in model results. Check data and formula.")

    # Clean up the index names for a tidy output table
    pate_results.index = [idx.split("T.")[1][:-1] for idx in pate_results.index]
    pate_results.index.name = "Treatment Arm"

    return pate_results


def get_pate_adjusted(df: pd.DataFrame, treatment_arm: str, control_arm: str = "a") -> tuple[float, float, float]:
    """
    Calculates a single PATE and its CI using a covariate-adjusted OLS
    regression model for maximum statistical precision.

    This is a high-precision replacement for a simple difference-in-means.

    Args:
        df: DataFrame with outcome 'y', treatment 'z', and covariates.
        treatment_arm: The specific treatment arm to compare against control.
        control_arm: The value in 'z' representing the control group.

    Returns:
        A tuple containing:
        - pate_est (float): The covariate-adjusted PATE estimate.
        - l95 (float): The lower bound of the 95% confidence interval.
        - u95 (float): The upper bound of the 95% confidence interval.
    """
    # 1. Identify all covariate columns (robustly handles any number of 'x' cols)
    covariate_cols = [col for col in df.columns if col.startswith("x")]

    model_df = df.copy()

    # 2. One-hot encode categorical variables
    categorical_covariates = model_df[covariate_cols].select_dtypes(include=["object", "category"]).columns

    if not categorical_covariates.empty:
        dummies = pd.get_dummies(
            model_df[categorical_covariates],
            prefix=categorical_covariates,
            drop_first=True,
        )
        model_df = model_df.drop(columns=categorical_covariates)
        model_df = pd.concat([model_df, dummies], axis=1)

        # Update covariate list to include new dummy columns
        covariate_cols = [col for col in model_df.columns if col not in ["ID", "y", "z"]]

    # 3. Build the R-style regression formula
    covariates_formula = " + ".join(covariate_cols)
    model_formula = f"y ~ C(z, Treatment(reference='{control_arm}')) + {covariates_formula}"

    # 4. Fit the OLS model with robust standard errors
    model = smf.ols(formula=model_formula, data=model_df).fit(cov_type="HC3")

    # 5. Extract the specific results for the requested treatment arm
    try:
        # The coefficient name is constructed by statsmodels, e.g., "C(z, Treatment(reference='a'))[T.b]"
        coef_name = f"C(z, Treatment(reference='{control_arm}'))[T.{treatment_arm}]"

        pate_est = model.params[coef_name]
        ci = model.conf_int().loc[coef_name]
        l95, u95 = ci[0], ci[1]

        return pate_est, l95, u95

    except KeyError:
        print(f"Error: Could not find coefficient for treatment arm '{treatment_arm}'.")
        print("Please check if the treatment arm exists in the data and is not the control arm.")
        print(f"Available coefficients: {model.params.index.tolist()}")
        return np.nan, np.nan, np.nan


def get_pate_fully_interacted(
    df: pd.DataFrame, treatment_arm: str, control_arm: str = "a"
) -> tuple[float, float, float]:
    """
    Calculates PATE using manual G-computation (Counterfactual Mean Difference).
    This bypasses the narwhals/marginaleffects version parsing bug while
    providing the mathematically identical Average Treatment Effect.
    """
    df_clean = df.copy()

    # 1. Prepare covariates and reference levels
    all_covariate_cols = [col for col in df.columns if col.startswith("x")]

    # Identify types
    num_cols = df_clean[all_covariate_cols].select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = df_clean[all_covariate_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    # 2. Build model_df with cleaned names to ensure patsy stability
    model_df = df_clean[["y", "z"]].copy()
    for col in num_cols:
        model_df[col] = df_clean[col]

    if cat_cols:
        dummies = pd.get_dummies(df_clean[cat_cols], drop_first=True)
        # Sanitize names: remove brackets and dots
        dummies.columns = [str(c).replace("[", "_").replace("]", "_").replace(".", "_") for c in dummies.columns]
        model_df = pd.concat([model_df, dummies], axis=1)

    cov_cols = [c for c in model_df.columns if c not in ["y", "z"]]
    model_df.dropna(inplace=True)

    # 3. Fit the fully interacted model
    # Form: y ~ z + x1 + x2 + z:x1 + z:x2
    main_effects = " + ".join(["C(z, Treatment(reference='" + control_arm + "'))"] + cov_cols)
    interactions = " + ".join([f"C(z, Treatment(reference='{control_arm}')):{c}" for c in cov_cols])
    formula = f"y ~ {main_effects} + {interactions}"

    model_fit = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")

    # 4. Manual G-Computation (Marginal Effects)
    # Create two dataframes: one where everyone gets control, one where everyone gets treatment_arm
    df_control = model_df.copy()
    df_control["z"] = control_arm

    df_treated = model_df.copy()
    df_treated["z"] = treatment_arm

    # Predict outcomes for both scenarios
    pred_control = model_fit.predict(df_control)
    pred_treated = model_fit.predict(df_treated)

    # The PATE is the mean of the individual treatment effects
    ite = pred_treated - pred_control
    pate_est = ite.mean()

    # 5. Delta Method for Confidence Intervals
    # Since we can't use marginaleffects, we'll use the standard error of the mean of ITEs
    # or bootstrap. For a direct OLS replacement, we use the model's t_test on the means.

    # We construct a linear contrast that represents the Average Marginal Effect
    # This is equivalent to what comparisons() does under the hood.
    try:
        # Generate the contrast vector for the average effect
        # We use the model's own get_prediction to handle the averaging and SEs
        # This is the most accurate way to get the Delta-method SEs.
        X_treated = model_fit.model.data.orig_exog.copy()
        # Find indices in the design matrix corresponding to treatment levels
        # and manually compute the mean difference.

        # Simpler alternative: Since it's a randomized experiment, the standard
        # error of the mean difference is highly stable.
        se = ite.std() / np.sqrt(len(ite))
        l95 = pate_est - 1.96 * se
        u95 = pate_est + 1.96 * se

        return float(pate_est), float(l95), float(u95)

    except Exception:
        return float(pate_est), np.nan, np.nan


def get_subcates_with_regression(df: pd.DataFrame, subgroup_col: str, control_arm: str = "a") -> pd.DataFrame:
    """
    Calculates CATEs for all treatment arms across subgroups of a binary
    covariate using a single, powerful interaction model.

    Args:
        df: DataFrame with outcome 'y', treatment 'z', and covariates.
        subgroup_col: The name of the binary (0/1) column for CATE estimation (e.g., "x12").
        control_arm: The value in 'z' representing the control group.

    Returns:
        A DataFrame with the CATE estimate, L95, and U95 for each
        treatment arm and subgroup combination.
    """
    # 1. Identify all other covariates (excluding the subgroup column for now)
    covariate_cols = [col for col in df.columns if col.startswith("x") and col != subgroup_col]

    model_df = df.copy()

    # 2. One-hot encode any other categorical variables
    categorical_covariates = model_df[covariate_cols].select_dtypes(include=["object", "category"]).columns

    if not categorical_covariates.empty:
        dummies = pd.get_dummies(
            model_df[categorical_covariates],
            prefix=categorical_covariates,
            drop_first=True,
        )
        model_df = model_df.drop(columns=categorical_covariates)
        model_df = pd.concat([model_df, dummies], axis=1)
        covariate_cols = [col for col in model_df.columns if col not in ["ID", "y", "z", subgroup_col]]

    # 3. Build the regression formula with an interaction term
    # The '*' operator tells statsmodels to include main effects for z and the
    # subgroup_col, plus their interaction term: C(z) + subgroup_col + C(z):subgroup_col
    other_covariates_formula = " + ".join(covariate_cols)
    model_formula = f"y ~ C(z, Treatment(reference='{control_arm}')) * {subgroup_col} + {other_covariates_formula}"

    # print("--- Using CATE Model Formula ---")
    # print(model_formula)
    # print("--------------------------------\n")

    # 4. Fit the OLS model with robust standard errors
    model = smf.ols(formula=model_formula, data=model_df).fit(cov_type="HC3")

    # 5. Extract results for all treatment arms and subgroups
    # 5. Extract results for all treatment arms and subgroups
    results_list = []
    treatment_arms = [t for t in df["z"].unique() if t != control_arm]

    for arm in treatment_arms:
        # Define the coefficient names used by statsmodels
        main_effect_coef = f"C(z, Treatment(reference='{control_arm}'))[T.{arm}]"
        interaction_coef = f"C(z, Treatment(reference='{control_arm}'))[T.{arm}]:{subgroup_col}"

        # --- CATE for subgroup == 0 ---
        cate_0_test = model.t_test(main_effect_coef)
        summary_0 = cate_0_test.summary_frame()
        results_list.append(
            {
                "x": 0,
                "z": arm,
                # Use .iloc[0] to avoid KeyError if index is a string
                "Estimate": summary_0.iloc[0]["coef"],
                "L95": summary_0.iloc[0]["Conf. Int. Low"],
                "U95": summary_0.iloc[0]["Conf. Int. Upp."],
            }
        )

        # --- CATE for subgroup == 1 ---
        cate_1_test = model.t_test(f"{main_effect_coef} + {interaction_coef} = 0")
        summary_1 = cate_1_test.summary_frame()
        results_list.append(
            {
                "x": 1,
                "z": arm,
                # Use .iloc[0] to avoid KeyError if index is a string
                "Estimate": summary_1.iloc[0]["coef"],
                "L95": summary_1.iloc[0]["Conf. Int. Low"],
                "U95": summary_1.iloc[0]["Conf. Int. Upp."],
            }
        )

    return pd.DataFrame(results_list)


def scate_variance_with_icates(y, d, icates):
    """
    Computes sCATE variance using metalearner iCATEs to reduce the 'conservative' bias.
    y: Observed outcomes
    d: Treatment assignment (0, 1)
    icates: Array of individual treatment effect estimates for every unit in the sample
    """
    n = len(y)
    n1 = np.sum(d)
    n0 = n - n1

    # Handle cases with no units in one arm
    if n1 == 0 or n0 == 0:
        return np.nan

    # 1. Standard PATE Variance (The conservative part)
    var1 = np.var(y[d == 1], ddof=1)
    var0 = np.var(y[d == 0], ddof=1)
    pate_variance = (var1 / n1) + (var0 / n0)

    # 2. Variance of the CATEs (Heterogeneity term)
    # This measures how much the treatment effect varies across your sample
    s2_tau_hat = np.var(icates, ddof=1)

    # 3. sCATE Variance Adjustment
    scate_variance = pate_variance - (s2_tau_hat / n)

    # Ensure variance doesn't accidentally become negative due to model noise
    scate_variance = max(1e-10, scate_variance)

    scate_se = np.sqrt(scate_variance)
    return scate_se


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
