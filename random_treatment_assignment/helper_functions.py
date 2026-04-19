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
    X_processed = pd.get_dummies(
        X, columns=categorical_cols, drop_first=True, dtype=int
    )
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
    df_welch_den = ((var_treat / n_treat) ** 2 / (n_treat - 1)) + (
        (var_control / n_control) ** 2 / (n_control - 1)
    )

    if df_welch_den == 0:
        return estimate, np.nan, np.nan
    df_welch = df_welch_num / df_welch_den

    if np.isnan(df_welch) or df_welch <= 0:
        return estimate, np.nan, np.nan

    t_stat = t.ppf(0.975, df=df_welch)
    return estimate, estimate - t_stat * se_diff, estimate + t_stat * se_diff


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
