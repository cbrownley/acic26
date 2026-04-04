import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style(style="whitegrid")


def plot_comparisons(file1, file2, pair_id, out_dir=None):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1["model"] = "Model_1"
    df2["model"] = "Model_2"

    df = pd.concat([df1, df2], ignore_index=True)

    z_values = sorted(df["z"].unique())

    for z in z_values:
        d = df[df["z"] == z]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Pair {pair_id} — Group {z}", fontsize=16)

        # --- 1. KDE plot (Estimate)
        sns.kdeplot(data=d, x="Estimate", hue="model", ax=axes[0, 0], common_norm=False)
        axes[0, 0].set_title("KDE: Estimate")

        # --- 2. ECDF plot
        sns.ecdfplot(data=d, x="Estimate", hue="model", ax=axes[0, 1])
        axes[0, 1].set_title("ECDF: Estimate")

        # --- 3. Boxplot
        sns.boxplot(data=d, x="model", y="Estimate", ax=axes[0, 2])
        axes[0, 2].set_title("Boxplot: Estimate")

        # --- 4. Difference distribution (requires alignment by ID)
        d1 = df1[df1["z"] == z].sort_values("ID")
        d2 = df2[df2["z"] == z].sort_values("ID")

        delta = d1["Estimate"].values - d2["Estimate"].values

        axes[1, 0].hist(delta, bins=30)
        axes[1, 0].axvline(0, linestyle="--")
        axes[1, 0].set_title("Difference: Estimate (Model1 - Model2)")

        # --- 5. Interval width comparison
        d["width"] = d["U95"] - d["L95"]
        sns.boxplot(data=d, x="model", y="width", ax=axes[1, 1])
        axes[1, 1].set_title("CI Width (U95 - L95)")

        # --- 6. Parity scatter
        axes[1, 2].scatter(d1["Estimate"], d2["Estimate"], alpha=0.3)
        lims = [
            min(d1["Estimate"].min(), d2["Estimate"].min()),
            max(d1["Estimate"].max(), d2["Estimate"].max()),
        ]
        axes[1, 2].plot(lims, lims, "k--")
        axes[1, 2].set_title("Parity Plot")
        axes[1, 2].set_xlabel("Model 1")
        axes[1, 2].set_ylabel("Model 2")

        plt.tight_layout()

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(out_dir) / f"pair_{pair_id}_z_{z}.png")

        plt.close()


is_AA_test = True  # Set to False for 1v2 comparison

current_file = Path(__file__)
# Go up one level, then into another folder
if is_AA_test:
    folder1 = current_file.parent.parent / "inverse_variance_weighted_ensemble" / "0020_1"
    folder2 = current_file.parent.parent / "inverse_variance_weighted_ensemble" / "0020_1b"
    out_dir = current_file.parent.parent / "plots" / "1v1b"
else:
    folder1 = current_file.parent.parent / "inverse_variance_weighted_ensemble" / "0020_1"
    folder2 = current_file.parent.parent / "r_loss_super_learner" / "submissions" / "0020_2"
    out_dir = current_file.parent.parent / "plots" / "1v2"

for i in range(1, 19):
    if is_AA_test:
        f1 = folder1 / f"iCATE_{i}_0020_1.csv"
        f2 = folder2 / f"iCATE_{i}_0020_1.csv"
    else:
        f1 = folder1 / f"iCATE_{i}_0020_1.csv"
        f2 = folder2 / f"iCATE_{i}_0020_2.csv" 
    plot_comparisons(f1, f2, pair_id=i, out_dir=out_dir)
