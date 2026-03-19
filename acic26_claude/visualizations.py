"""
visualizations.py
All plots for understanding and comparing causal estimates.

Design principles
-----------------
- Each function is self-contained: pass DataFrames in, get a Figure back.
- save_dir=None → return Figure without saving (notebook / interactive use).
- save_dir=Path → save PNG + return Figure.
- Small-multiples layout for anything with more than 2 panels.
- Consistent arm color palette throughout every plot.
- All figures use a single shared style context so they stay cohesive.

Public API  (single-dataset)
----------------------------
plot_icate_distributions(icate_df, data_id)
    Violin plots of iCATE distributions, one per arm.  Shows heterogeneity.

plot_estimates_with_ci(scate_df, pate_df, data_id)
    Side-by-side dot-and-whisker chart: sCATE vs PATE per arm.

plot_subcate(subcate_df, data_id)
    Grouped bar chart: arm × subgroup (x12=0, x12=1) with CI error bars.

plot_gain_curves(curves_dict, aucs_dict, data_id)
    2×2 small multiples, one panel per arm.  Relative cumulative gain curve
    with shaded area and AUC annotation.

Public API  (multi-dataset)
---------------------------
plot_auc_heatmap(auc_records, dataset_ids)
    Heatmap: datasets (rows) × arms (cols), cells = AURC value.

plot_scate_across_datasets(scate_records, dataset_ids)
    Small multiples: one panel per arm, point estimates + CI whiskers per
    dataset.  Reveals consistency and sign-variability across datasets.

plot_icate_violin_grid(icate_dfs, dataset_ids)
    Grid: rows = datasets, cols = arms.  Each cell = mini violin of iCATE
    distribution.  Best viewed for ≤ 9 datasets.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional

from config import TREATMENTS, CONTROL

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Global style & palette
# ─────────────────────────────────────────────────────────────────────────────

# One distinct color per treatment arm — order matches TREATMENTS = [b,c,d,e]
ARM_COLORS = {
    "b": "#4C8EDA",   # blue
    "c": "#E07B39",   # orange
    "d": "#3DB87A",   # green
    "e": "#C855A0",   # magenta
}
SUBGROUP_COLORS = {0: "#7FA7D4", 1: "#D47F7F"}   # x12=0 light blue, =1 rose
RANDOM_COLOR    = "#AAAAAA"

STYLE = {
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#E5E5E5",
    "grid.linewidth":     0.6,
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.frameon":     False,
    "legend.fontsize":    10,
}

def _apply_style():
    plt.rcParams.update(STYLE)


def _save_or_return(fig: plt.Figure, filename: str, save_dir) -> plt.Figure:
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [VIZ] saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. iCATE distributions  (per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_icate_distributions(
    icate_df: pd.DataFrame,
    data_id:  str,
    save_dir  = None,
) -> plt.Figure:
    """
    Violin plots of iCATE distributions by treatment arm.

    Each violin shows the full distribution of individual treatment effect
    estimates across the sample.  Wide = many individuals near that CATE
    value; an asymmetric violin suggests heterogeneous response.
    A horizontal line at y=0 separates beneficial from harmful effects.

    Parameters
    ----------
    icate_df : long-format iCATE DataFrame (cols: ID, z, Estimate, L95, U95)
    data_id  : dataset ID string (used in title)
    save_dir : optional output directory for PNG
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    positions = []
    all_vals  = []
    for i, z in enumerate(TREATMENTS):
        vals = icate_df.loc[icate_df["z"] == z, "Estimate"].values
        parts = ax.violinplot(
            vals,
            positions=[i],
            widths=0.65,
            showmedians=True,
            showextrema=False,
        )
        color = ARM_COLORS[z]
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)
        positions.append(i)
        all_vals.extend(vals)

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{z} vs {CONTROL}" for z in TREATMENTS])
    ax.set_ylabel("iCATE estimate  (Y(z) − Y(a))")
    ax.set_title(f"Dataset {data_id} — iCATE distributions by arm")
    ax.yaxis.grid(True, linewidth=0.5)
    ax.set_axisbelow(True)

    legend_patches = [
        matplotlib.patches.Patch(facecolor=ARM_COLORS[z], label=f"arm {z}", alpha=0.75)
        for z in TREATMENTS
    ]
    ax.legend(handles=legend_patches, loc="best")

    fig.tight_layout()
    return _save_or_return(fig, f"icate_dist_{data_id}.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 2. sCATE and PATE side-by-side  (per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_estimates_with_ci(
    scate_df: pd.DataFrame,
    pate_df:  pd.DataFrame,
    data_id:  str,
    save_dir  = None,
) -> plt.Figure:
    """
    Dot-and-whisker chart comparing sCATE and PATE per arm.

    Two panels share a y-axis (effect scale) so the two estimands can be
    visually compared.  Dots = point estimate; horizontal bars = 95% CI.
    A vertical line at x=0 separates positive from negative effects.

    Parameters
    ----------
    scate_df : sCATE DataFrame (cols: z, Estimate, L95, U95)
    pate_df  : PATE DataFrame  (cols: z, Estimate, L95, U95)
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    datasets = {"sCATE (sample avg)": scate_df, "PATE (population avg)": pate_df}

    y_pos = np.arange(len(TREATMENTS))

    for ax, (title, df) in zip(axes, datasets.items()):
        for i, z in enumerate(TREATMENTS):
            row  = df.loc[df["z"] == z].iloc[0]
            est  = row["Estimate"]
            xerr = np.array([[est - row["L95"]], [row["U95"] - est]])
            ax.errorbar(
                est, i,
                xerr=xerr,
                fmt="o",
                color=ARM_COLORS[z],
                markersize=8,
                capsize=5,
                linewidth=1.8,
                label=f"arm {z}",
            )
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{z} vs {CONTROL}" for z in TREATMENTS])
        ax.set_xlabel("Estimated treatment effect")
        ax.set_title(title)
        ax.grid(axis="x", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Treatment arm")
    fig.suptitle(f"Dataset {data_id} — Average treatment effects with 95% CIs",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return _save_or_return(fig, f"ate_ci_{data_id}.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 3. subCATE grouped bars  (per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_subcate(
    subcate_df: pd.DataFrame,
    data_id:    str,
    save_dir    = None,
) -> plt.Figure:
    """
    Grouped bar chart of subCATE by arm and X12 subgroup.

    Within each arm cluster, two bars show the subgroup-specific average
    treatment effect for x12=0 (light) and x12=1 (dark).  Error bars show
    95% CIs.  This makes the subgroup × arm interaction immediately visible.

    Parameters
    ----------
    subcate_df : subCATE DataFrame (cols: z, x, Estimate, L95, U95)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    n_arms     = len(TREATMENTS)
    bar_width  = 0.32
    group_gap  = 0.85
    x_centers  = np.arange(n_arms) * group_gap

    for j, x_val in enumerate([0, 1]):
        offsets = x_centers + (j - 0.5) * bar_width
        for i, z in enumerate(TREATMENTS):
            row   = subcate_df.loc[(subcate_df["z"] == z) &
                                   (subcate_df["x"] == x_val)].iloc[0]
            est   = row["Estimate"]
            yerr  = np.array([[est - row["L95"]], [row["U95"] - est]])
            color = SUBGROUP_COLORS[x_val]
            alpha = 0.85 if x_val == 1 else 0.60
            bar = ax.bar(
                offsets[i], est,
                width=bar_width - 0.04,
                color=ARM_COLORS[z],
                alpha=alpha,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.errorbar(
                offsets[i], est,
                yerr=yerr,
                fmt="none",
                color="#333333",
                capsize=4,
                linewidth=1.2,
            )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{z} vs {CONTROL}" for z in TREATMENTS])
    ax.set_ylabel("subCATE estimate  (normalised by full n)")
    ax.set_title(f"Dataset {data_id} — subCATE by arm and X₁₂ subgroup")

    legend_handles = [
        matplotlib.patches.Patch(
            facecolor="#888888", alpha=0.60, label="x₁₂ = 0"),
        matplotlib.patches.Patch(
            facecolor="#888888", alpha=0.85, label="x₁₂ = 1"),
    ]
    # Also add arm color swatches
    arm_handles = [
        matplotlib.patches.Patch(facecolor=ARM_COLORS[z], label=f"arm {z}")
        for z in TREATMENTS
    ]
    ax.legend(handles=legend_handles + arm_handles,
              ncol=3, loc="upper right", fontsize=9)

    fig.tight_layout()
    return _save_or_return(fig, f"subcate_{data_id}.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Relative cumulative gain curves  (per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gain_curves(
    curves_dict: dict,
    aucs_dict:   dict,
    data_id:     str,
    save_dir     = None,
) -> plt.Figure:
    """
    2×2 small-multiples of relative cumulative gain curves, one per arm.

    Each panel
    ----------
    - x-axis: fraction of sample targeted (sorted by descending iCATE)
    - y-axis: average treatment effect among targeted fraction, minus global ATE
    - Shaded area between curve and y=0 represents the AURC
    - Gray dashed line at y=0 is the "random targeting" baseline
    - AURC value annotated in the panel corner

    A curve that rises steeply from the left means the model correctly ranks
    high-responders above low-responders.

    Parameters
    ----------
    curves_dict : {z: fklearn curve DataFrame}  from evaluate.compute_gain_curves
    aucs_dict   : {z: float AURC}               from evaluate.evaluate_aucs
    """
    _apply_style()
    arms_present = [z for z in TREATMENTS if z in curves_dict]
    n_panels     = len(arms_present)
    if n_panels == 0:
        print("  [VIZ] No gain curves to plot.")
        return None

    ncols  = 2
    nrows  = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11, 4.5 * nrows),
                             sharex=True)
    axes_flat = np.array(axes).flatten()

    # Detect column names from fklearn output
    # fklearn returns a DataFrame with columns like:
    #   'relative_cumulative_gain', 'n_samples', 'perc_size'
    # Older versions may use 'gain' or different naming — handle both.
    def _get_xy(curve):
        """
        Extract (x_fraction, y_gain) from whatever fklearn returns.

        fklearn's relative_cumulative_gain_curve() returns either:
          - a pandas DataFrame (older versions)
          - a numpy ndarray of shape (steps, 2) where col-0=x, col-1=y
            (newer versions, e.g. 3.x)
        """
        # ── numpy array path ─────────────────────────────────────────────────
        if isinstance(curve, np.ndarray):
            if curve.ndim == 2 and curve.shape[1] >= 2:
                return curve[:, 0], curve[:, 1]
            # 1-D: treat as y only, generate x
            y = curve.ravel()
            return np.linspace(0, 1, len(y)), y

        # ── DataFrame path ───────────────────────────────────────────────────
        df = curve
        n = df["n_samples"].iloc[-1] if "n_samples" in df.columns else len(df)
        if "perc_size" in df.columns:
            x = df["perc_size"].values
        elif "n_samples" in df.columns:
            x = df["n_samples"].values / n
        else:
            x = np.linspace(0, 1, len(df))

        if "relative_cumulative_gain" in df.columns:
            y = df["relative_cumulative_gain"].values
        elif "gain" in df.columns:
            y = df["gain"].values
        else:
            skip = {"n_samples", "perc_size", "treatment", "outcome", "prediction"}
            numeric_cols = [c for c in df.select_dtypes(include=float).columns
                            if c not in skip]
            y = df[numeric_cols[0]].values if numeric_cols else np.zeros(len(x))
        return x, y

    for idx, z in enumerate(arms_present):
        ax    = axes_flat[idx]
        color = ARM_COLORS[z]
        curve = curves_dict[z]
        x, y  = _get_xy(curve)

        # Curve
        ax.plot(x, y, color=color, linewidth=2, label=f"arm {z}")
        # Shaded area between curve and baseline
        ax.fill_between(x, y, 0,
                         where=(y >= 0), color=color, alpha=0.15, interpolate=True)
        ax.fill_between(x, y, 0,
                         where=(y < 0),  color=color, alpha=0.10, interpolate=True)
        # Random baseline
        ax.axhline(0, color=RANDOM_COLOR, linewidth=1.2,
                   linestyle="--", label="random")

        # AURC annotation
        auc_val = aucs_dict.get(z, float("nan"))
        auc_txt = f"AURC = {auc_val:.4f}" if not np.isnan(auc_val) else "AURC = n/a"
        ax.text(0.97, 0.95, auc_txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor=color, linewidth=0.8))

        ax.set_title(f"Arm {z} vs {CONTROL}")
        ax.set_xlabel("Fraction of population targeted")
        ax.set_ylabel("Gain relative to ATE")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.grid(True, linewidth=0.4)
        ax.set_axisbelow(True)

    # Hide unused panels
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Dataset {data_id} — Relative cumulative gain curves\n"
        f"(curve above baseline = model ranks high-responders correctly)",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_or_return(fig, f"gain_curves_{data_id}.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUC heatmap across datasets  (multi-dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_auc_heatmap(
    auc_records:  list[dict],
    save_dir      = None,
) -> plt.Figure:
    """
    Heatmap of AURC values across datasets (rows) and treatment arms (cols).

    Call with a list of records like:
        [{"data_id": "0001", "b": 0.032, "c": 0.015, "d": 0.041, "e": 0.008}, ...]

    Color scale: white → arm-colored diverging, so it's immediately visible
    which datasets had high/low ranking quality for each arm.

    Parameters
    ----------
    auc_records : list of dicts with keys 'data_id' and one per arm in TREATMENTS
    """
    _apply_style()
    df = pd.DataFrame(auc_records).set_index("data_id")[TREATMENTS]
    df.columns = [f"{z} vs {CONTROL}" for z in TREATMENTS]

    n_datasets, n_arms = df.shape
    fig_h = max(4, 0.45 * n_datasets + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    vmax = df.abs().max().max()
    im = ax.imshow(df.values, aspect="auto",
                   cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_arms))
    ax.set_xticklabels(df.columns, rotation=0)
    ax.set_yticks(range(n_datasets))
    ax.set_yticklabels(df.index)

    # Annotate cells
    for i in range(n_datasets):
        for j in range(n_arms):
            val = df.values[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=8.5,
                    color="white" if abs(val) > vmax * 0.6 else "#333333")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("AURC", fontsize=10)

    ax.set_title("AURC (area under relative cumulative gain curve)\n"
                 "per dataset × treatment arm", fontsize=12)
    ax.set_xlabel("Treatment arm")
    ax.set_ylabel("Dataset ID")
    fig.tight_layout()
    return _save_or_return(fig, "auc_heatmap.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 6. sCATE estimates across datasets  (multi-dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scate_across_datasets(
    scate_records: list[dict],
    save_dir       = None,
) -> plt.Figure:
    """
    Small-multiples dot-and-whisker chart: sCATE per dataset, faceted by arm.

    One panel per treatment arm.  Within each panel, each dataset is a row
    with a point estimate (dot) and 95% CI (whisker).  Vertical dashed line
    at x=0 separates positive from negative effects.

    Call with a list of records:
        [{"data_id": "0001", "z": "b", "Estimate": 1.2, "L95": 0.8, "U95": 1.6}, ...]
    """
    _apply_style()
    df = pd.DataFrame(scate_records)
    dataset_ids = df["data_id"].unique()
    n_datasets  = len(dataset_ids)
    id_to_y     = {did: i for i, did in enumerate(dataset_ids)}

    ncols = min(4, len(TREATMENTS))
    nrows = int(np.ceil(len(TREATMENTS) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 1.5 + 0.45 * n_datasets),
                              sharey=True)
    axes_flat = np.array(axes).flatten()

    y_ticks = list(range(n_datasets))

    for idx, z in enumerate(TREATMENTS):
        ax      = axes_flat[idx]
        sub     = df[df["z"] == z]
        color   = ARM_COLORS[z]

        for _, row in sub.iterrows():
            y_pos = id_to_y[row["data_id"]]
            est   = row["Estimate"]
            xerr  = np.array([[est - row["L95"]], [row["U95"] - est]])
            ax.errorbar(est, y_pos,
                        xerr=xerr,
                        fmt="o",
                        color=color,
                        markersize=5,
                        capsize=3,
                        linewidth=1.2,
                        alpha=0.85)

        ax.axvline(0, color="#555555", linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(list(dataset_ids), fontsize=8)
        ax.set_title(f"arm {z} vs {CONTROL}", color=color, fontweight="bold")
        ax.set_xlabel("sCATE estimate")
        ax.grid(axis="x", linewidth=0.4)
        ax.set_axisbelow(True)

    for idx in range(len(TREATMENTS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Dataset")
    fig.suptitle("sCATE estimates across datasets (95% CIs)", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save_or_return(fig, "scate_across_datasets.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 7. iCATE violin grid across datasets  (multi-dataset, ≤ 9 datasets)
# ─────────────────────────────────────────────────────────────────────────────

def plot_icate_violin_grid(
    icate_dfs:   dict,
    save_dir     = None,
) -> plt.Figure:
    """
    Grid of violin plots: rows = datasets, columns = arms.

    Each cell shows the distribution of iCATE estimates for one (dataset, arm)
    pair.  Useful for spotting which datasets have more or less heterogeneity,
    or which arms have consistently multimodal response distributions.

    Parameters
    ----------
    icate_dfs : dict {data_id: icate_df}  — long-format iCATE DataFrames
    """
    _apply_style()
    dataset_ids = list(icate_dfs.keys())
    n_datasets  = len(dataset_ids)
    if n_datasets > 12:
        print("  [VIZ] iCATE violin grid: capping at 12 datasets for readability.")
        dataset_ids = dataset_ids[:12]
        n_datasets  = 12

    n_arms = len(TREATMENTS)
    fig, axes = plt.subplots(
        n_datasets, n_arms,
        figsize=(2.8 * n_arms, 2.2 * n_datasets),
        sharex="col",
    )
    # Ensure 2D axes array even for single dataset
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for r, did in enumerate(dataset_ids):
        df = icate_dfs[did]
        for c, z in enumerate(TREATMENTS):
            ax   = axes[r, c]
            vals = df.loc[df["z"] == z, "Estimate"].values
            parts = ax.violinplot(
                vals, positions=[0], widths=0.7,
                showmedians=True, showextrema=False,
            )
            color = ARM_COLORS[z]
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.70)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.5)

            ax.axhline(0, color="#888888", linewidth=0.6,
                       linestyle="--", alpha=0.7)
            ax.set_xticks([])
            ax.yaxis.set_tick_params(labelsize=7)
            ax.spines["bottom"].set_visible(False)

            # Row label (dataset) on left edge
            if c == 0:
                ax.set_ylabel(f"ds {did}", fontsize=8, rotation=0,
                              labelpad=30, va="center")
            # Column label (arm) on top edge
            if r == 0:
                ax.set_title(f"{z} vs {CONTROL}", color=color,
                             fontsize=10, fontweight="bold")

    fig.suptitle("iCATE distributions — datasets × arms", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save_or_return(fig, "icate_violin_grid.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PATE vs sCATE scatter  (multi-dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pate_vs_scate(
    combined_records: list[dict],
    save_dir          = None,
) -> plt.Figure:
    """
    Scatter plot: sCATE (x) vs PATE (y) per arm, across datasets.

    Points near the diagonal mean sCATE is a good proxy for PATE (expected
    when the sample is representative of the population).  Systematic
    deviations reveal covariate shift or finite-sample issues.

    Call with records containing both sCATE and PATE info:
        [{"data_id": "0001", "z": "b",
          "sCATE": 1.2, "sCATE_L95": 0.8, "sCATE_U95": 1.6,
          "PATE": 1.18, "PATE_L95": 0.75, "PATE_U95": 1.61}, ...]
    """
    _apply_style()
    df = pd.DataFrame(combined_records)

    fig, axes = plt.subplots(1, len(TREATMENTS),
                              figsize=(3.8 * len(TREATMENTS), 4),
                              sharex=False, sharey=False)
    if len(TREATMENTS) == 1:
        axes = [axes]

    for ax, z in zip(axes, TREATMENTS):
        sub   = df[df["z"] == z]
        color = ARM_COLORS[z]

        ax.scatter(sub["sCATE"], sub["PATE"],
                   c=color, alpha=0.75, s=55, zorder=3)

        # Diagonal reference line
        all_vals = pd.concat([sub["sCATE"], sub["PATE"]])
        lo, hi   = all_vals.min(), all_vals.max()
        margin   = (hi - lo) * 0.08
        diag     = np.array([lo - margin, hi + margin])
        ax.plot(diag, diag, color=RANDOM_COLOR,
                linewidth=1, linestyle="--", label="y = x", zorder=2)

        ax.set_xlabel("sCATE")
        ax.set_ylabel("PATE")
        ax.set_title(f"arm {z} vs {CONTROL}", color=color, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(diag); ax.set_ylim(diag)
        ax.grid(True, linewidth=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("sCATE vs PATE across datasets\n(points near diagonal = consistent estimation)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, "pate_vs_scate.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Convenience: generate all single-dataset plots in one call
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_single_dataset(
    data_id:    str,
    icate_df:   pd.DataFrame,
    scate_df:   pd.DataFrame,
    subcate_df: pd.DataFrame,
    pate_df:    pd.DataFrame,
    curves_dict: dict,
    aucs_dict:   dict,
    save_dir    = None,
) -> dict:
    """
    Generate all four single-dataset plots and return them as a dict.

    Parameters mirror the DataFrames produced by inference.py and the
    dicts produced by evaluate.py.

    Returns
    -------
    dict with keys: 'icate', 'estimates_ci', 'subcate', 'gain_curves'
    """
    figs = {}
    figs["icate"]        = plot_icate_distributions(icate_df, data_id, save_dir)
    figs["estimates_ci"] = plot_estimates_with_ci(scate_df, pate_df, data_id, save_dir)
    figs["subcate"]      = plot_subcate(subcate_df, data_id, save_dir)
    if curves_dict:
        figs["gain_curves"] = plot_gain_curves(curves_dict, aucs_dict, data_id, save_dir)
    return figs
