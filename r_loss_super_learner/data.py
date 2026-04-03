"""
data.py
Data loading and feature preprocessing.

Covariate treatment
-------------------
Per the ACIC 2026 challenge description, all string covariates (x1–x6,
which take values like "level 1", "level 2", ...) are explicitly described
as *nominal*, not ordinal.  All string columns are therefore one-hot-encoded
regardless of whether their values happen to follow a numeric naming pattern.

ColumnTransformer layout
------------------------
  • OneHotEncoder (drop="first") — all nominal string columns
  • StandardScaler                — continuous numeric columns
  • passthrough                   — binary (0/1) columns

Public API
----------
load_and_split(path)       -> (ID, Y, T, X_raw, x12)
build_preprocessor(X_raw)  -> fitted ColumnTransformer
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CONTROL


def load_and_split(path: str):
    """
    Read a competition CSV and split into components.

    Returns
    -------
    ID     : np.ndarray (n,)  — observation identifiers
    Y      : np.ndarray (n,)  — outcome (float)
    T      : np.ndarray (n,)  — treatment arm strings ('a'..'e')
    X_raw  : pd.DataFrame     — raw covariates (before preprocessing)
    x12    : np.ndarray (n,)  — binary subgroup indicator (int)
    """
    df = pd.read_csv(path)
    ID = df["ID"].values
    Y = df["y"].values.astype(float)
    T = df["z"].values
    x12 = df["x12"].values.astype(int)
    X = df.drop(columns=["ID", "y", "z"])
    return ID, Y, T, X, x12


def build_preprocessor(X_raw: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit a ColumnTransformer that:
      • One-hot-encodes all nominal (string) columns  — drop first level
      • Standard-scales continuous numeric columns
      • Passes binary (0/1) columns through unchanged

    Parameters
    ----------
    X_raw : raw covariate DataFrame (before any encoding)

    Returns
    -------
    Fitted ColumnTransformer — call .transform() to encode new data.
    """
    nominal = [c for c in X_raw.columns if X_raw[c].dtype == object]
    binary = [c for c in X_raw.columns if c not in nominal and X_raw[c].nunique() <= 2]
    numeric = [c for c in X_raw.columns if c not in nominal and c not in binary]

    steps = []
    if nominal:
        steps.append(
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                nominal,
            )
        )
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    if binary:
        steps.append(("bin", "passthrough", binary))

    ct = ColumnTransformer(steps, remainder="drop")
    ct.fit(X_raw)
    return ct
