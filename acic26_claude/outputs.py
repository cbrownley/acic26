"""
outputs.py
Derive "best treatment" DataFrames and write all CSVs to disk.

Public API
----------
build_icate_df(icates, lowers, uppers, ID) -> pd.DataFrame  (long format)
best_icate(icates, ID)                     -> pd.DataFrame
best_scalar(df, z_col, est_col)            -> pd.DataFrame
best_subcate(subcate_df)                   -> pd.DataFrame
save(df, filename, out_dir)                -> None
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from config import TREATMENTS


# ─────────────────────────────────────────────────────────────────────────────
# iCATE long-format assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_icate_df(
    icates: dict,
    lowers: dict,
    uppers: dict,
    ID: np.ndarray,
) -> pd.DataFrame:
    """
    Assemble the 4n-row long-format iCATE DataFrame.

    Columns: ID, z, Estimate, L95, U95
    """
    rows = []
    n = len(ID)
    for z in TREATMENTS:
        for i in range(n):
            rows.append({
                "ID":       int(ID[i]),
                "z":        z,
                "Estimate": float(icates[z][i]),
                "L95":      float(lowers[z][i]),
                "U95":      float(uppers[z][i]),
            })
    return pd.DataFrame(rows)[["ID", "z", "Estimate", "L95", "U95"]]


# ─────────────────────────────────────────────────────────────────────────────
# Best-treatment derivations
# ─────────────────────────────────────────────────────────────────────────────

def best_icate(icates: dict, ID: np.ndarray) -> pd.DataFrame:
    """
    For each individual, return the arm z that maximises iCATE(x_i, z).

    Returns a DataFrame with columns [ID, best_z].
    """
    matrix = np.column_stack([icates[z] for z in TREATMENTS])  # (n, 4)
    best_j = np.argmax(matrix, axis=1)
    best_z = [TREATMENTS[j] for j in best_j]
    return pd.DataFrame({"ID": ID.astype(int), "best_z": best_z})


def best_scalar(
    df: pd.DataFrame,
    z_col: str = "z",
    est_col: str = "Estimate",
) -> pd.DataFrame:
    """
    From a sCATE or PATE DataFrame return the single best arm.

    Returns a DataFrame with column [best_z].
    """
    best_z = df.loc[df[est_col].idxmax(), z_col]
    return pd.DataFrame({"best_z": [best_z]})


def best_subcate(subcate_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each subgroup value x ∈ {0, 1} return the arm that maximises
    subCATE(z, x).

    Returns a DataFrame with columns [x, best_z].
    """
    rows = []
    for x_val in [0, 1]:
        sub    = subcate_df[subcate_df["x"] == x_val]
        best_z = sub.loc[sub["Estimate"].idxmax(), "z"]
        rows.append({"x": int(x_val), "best_z": best_z})
    return pd.DataFrame(rows)[["x", "best_z"]]


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, filename: str, out_dir) -> None:
    """
    Write df to  <out_dir>/<filename>.csv  and log a one-liner summary.

    Parameters
    ----------
    df       : DataFrame to write
    filename : target filename (no directory prefix)
    out_dir  : output directory (str or Path)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"  ✓  {filename}  ({len(df)} data rows)")
