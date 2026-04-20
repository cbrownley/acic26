import pandas as pd
import numpy as np
import os
from scipy.stats import t
import re


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


def get_scate_intervals_from_pate(
    pate_df: pd.DataFrame,
    final_icate_df: pd.DataFrame,
    original_df: pd.DataFrame,
    treatment_arms: list,
    control_arm: str,
) -> pd.DataFrame:
    """
    Calculates sCATE confidence intervals by adjusting PATE variance with iCATE heterogeneity.

    Args:
        pate_df: DataFrame with PATE estimates (z, Estimate, L95, U95).
        final_icate_df: DataFrame with individual CATEs (ID, z, Estimate).
        original_df: The original DataFrame with raw data (z column needed for sample sizes).
        treatment_arms: A list of the names of the treatment arms.
        control_arm: The name of the control arm.

    Returns:
        A DataFrame with sCATE estimates and their new, narrower confidence intervals.
    """
    scate_results = []

    # Use the PATE point estimates as the sCATE point estimates
    scate_point_estimates = pate_df.set_index("z")["Estimate"]

    for arm in treatment_arms:
        # --- 1. Extract PATE Variance ---
        pate_row = pate_df[pate_df["z"] == arm]
        if pate_row.empty:
            continue

        pate_l95 = pate_row["L95"].iloc[0]
        pate_u95 = pate_row["U95"].iloc[0]

        # Implied PATE SEM = (Upper - Lower) / (2 * z_score)
        # For a 95% CI, the z-score is ~1.96
        pate_sem = (pate_u95 - pate_l95) / (2 * 1.96)
        pate_variance = pate_sem**2

        # --- 2. Estimate Effect Heterogeneity (S²_τ_hat) ---
        icates_for_arm = final_icate_df[final_icate_df["z"] == arm]["Estimate"]

        # Ensure there are enough iCATEs to calculate variance
        if len(icates_for_arm) < 2:
            continue

        s2_tau_hat = np.var(icates_for_arm, ddof=1)  # Use ddof=1 for sample variance

        # --- 3. Calculate sCATE Variance ---
        # Get total sample size 'n' for this specific treatment vs. control comparison
        n = original_df[original_df["z"].isin([arm, control_arm])].shape[0]

        if n == 0:
            continue

        # The core formula: V_sCATE = V_PATE - (S²_τ_hat / n)
        scate_variance = pate_variance - (s2_tau_hat / n)

        # Ensure variance is non-negative, as sampling error can sometimes make it slightly negative
        scate_variance = max(1e-12, scate_variance)

        # --- 4. Construct sCATE Confidence Intervals ---
        scate_se = np.sqrt(scate_variance)
        point_estimate = scate_point_estimates.get(arm)

        margin_of_error = 1.96 * scate_se
        scate_l95 = point_estimate - margin_of_error
        scate_u95 = point_estimate + margin_of_error

        scate_results.append(
            {
                "z": arm,
                "Estimate": point_estimate,
                "L95": scate_l95,
                "U95": scate_u95,
            }
        )

    return pd.DataFrame(scate_results)
