"""
Week 2 Task - Exploratory Data Analysis (EDA) and Visualization
Dataset : Cleaned Wine Recognition Dataset (output of Week 1 cleaning pipeline)
Author  : Keshari
-------------------------------------------------------------------
This script:
  1. Loads the cleaned Wine Recognition dataset produced in Week 1.
  2. Performs initial analysis: summary statistics, shape, class balance.
  3. Creates multiple annotated visualizations (histograms, boxplots,
     correlation heatmap, scatter plots, pairplot-style grid) to uncover
     patterns, correlations and anomalies.
  4. Performs simple data transformations/aggregations (group-by summaries,
     a derived feature) and saves them for the report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_IN = "data/00_input_cleaned_wine.csv"
OUT_DATA = "data"
OUT_IMG = "images"

plt.rcParams["figure.autolayout"] = True
CLASS_COLORS = {"class_0": "#e07a30", "class_1": "#3aa66b", "class_2": "#3a6ea6"}

# ---------------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------
print("STEP 1: Loading cleaned dataset...")
df = pd.read_csv(DATA_IN)
# Use the original (unscaled) numeric columns for interpretable EDA
raw_cols = [c for c in df.columns if not c.endswith("_scaled") and c != "wine_class"]
print(f"  Shape: {df.shape}")
print(f"  Feature columns used for EDA: {raw_cols}")

# ---------------------------------------------------------------------
# STEP 2: INITIAL ANALYSIS - SUMMARY STATISTICS
# ---------------------------------------------------------------------
print("STEP 2: Computing summary statistics...")
summary = df[raw_cols].describe().T
summary.to_csv(f"{OUT_DATA}/summary_statistics.csv")

class_counts = df["wine_class"].value_counts()
class_counts.to_csv(f"{OUT_DATA}/class_distribution.csv")

# Group-wise aggregation: mean of each feature per class (a "data transformation")
class_means = df.groupby("wine_class")[raw_cols].mean().round(2)
class_means.to_csv(f"{OUT_DATA}/class_wise_mean_aggregation.csv")

# Derived feature (simple transformation): phenol-to-flavanoid ratio
df["phenol_flavanoid_ratio"] = (df["total_phenols"] / df["flavanoids"]).round(3)
df.to_csv(f"{OUT_DATA}/01_eda_dataset_with_derived_feature.csv", index=False)

with open(f"{OUT_DATA}/eda_text_summary.txt", "w") as f:
    f.write("SHAPE: {}\n\n".format(df.shape))
    f.write("CLASS DISTRIBUTION:\n{}\n\n".format(class_counts))
    f.write("SUMMARY STATISTICS:\n{}\n\n".format(summary))
    f.write("CLASS-WISE MEAN AGGREGATION:\n{}\n".format(class_means))

# ---------------------------------------------------------------------
# STEP 3: VISUALIZATION 1 - Class distribution
# ---------------------------------------------------------------------
print("STEP 3: Plotting class distribution...")
plt.figure(figsize=(6, 4.5))
plt.bar(class_counts.index, class_counts.values,
        color=[CLASS_COLORS[c] for c in class_counts.index])
plt.title("Class Distribution of Wine Cultivars")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 1, str(v), ha="center")
plt.savefig(f"{OUT_IMG}/01_class_distribution.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 4: VISUALIZATION 2 - Histograms of key features
# ---------------------------------------------------------------------
print("STEP 4: Plotting feature histograms...")
key_features = ["alcohol", "malic_acid", "flavanoids", "color_intensity", "proline", "hue"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.flat, key_features):
    ax.hist(df[col], bins=15, color="#4c72b0", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
fig.suptitle("Distribution of Key Chemical Features", fontsize=14)
plt.savefig(f"{OUT_IMG}/02_feature_histograms.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 5: VISUALIZATION 3 - Boxplots by class (spot patterns/anomalies)
# ---------------------------------------------------------------------
print("STEP 5: Plotting boxplots by class...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, col in zip(axes, ["flavanoids", "color_intensity", "proline"]):
    data_by_class = [df[df.wine_class == c][col] for c in sorted(df.wine_class.unique())]
    bp = ax.boxplot(data_by_class, labels=sorted(df.wine_class.unique()), patch_artist=True)
    for patch, c in zip(bp["boxes"], sorted(df.wine_class.unique())):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.7)
    ax.set_title(col)
    ax.set_xlabel("Wine Class")
fig.suptitle("Feature Spread by Wine Class", fontsize=14)
plt.savefig(f"{OUT_IMG}/03_boxplots_by_class.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 6: VISUALIZATION 4 - Correlation heatmap
# ---------------------------------------------------------------------
print("STEP 6: Plotting correlation heatmap...")
corr = df[raw_cols].corr()
plt.figure(figsize=(9, 7.5))
im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(raw_cols)), raw_cols, rotation=90)
plt.yticks(range(len(raw_cols)), raw_cols)
plt.title("Correlation Heatmap of Chemical Features")
plt.savefig(f"{OUT_IMG}/04_correlation_heatmap.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 7: VISUALIZATION 5 - Scatter plots highlighting strongest correlation
# ---------------------------------------------------------------------
print("STEP 7: Plotting scatter plot of the most correlated pair...")
corr_no_diag = corr.where(~np.eye(len(corr), dtype=bool))
max_pair = corr_no_diag.abs().stack().idxmax()
f1, f2 = max_pair
plt.figure(figsize=(7, 5.5))
for c in sorted(df.wine_class.unique()):
    subset = df[df.wine_class == c]
    plt.scatter(subset[f1], subset[f2], label=c, color=CLASS_COLORS[c], alpha=0.75, edgecolor="white")
plt.xlabel(f1)
plt.ylabel(f2)
plt.title(f"{f1} vs {f2} (r = {corr.loc[f1, f2]:.2f}) by Class")
plt.legend(title="Wine Class")
plt.savefig(f"{OUT_IMG}/05_scatter_top_correlation.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 8: VISUALIZATION 6 - Pairwise scatter grid (mini pairplot)
# ---------------------------------------------------------------------
print("STEP 8: Plotting pairwise scatter grid...")
pair_features = ["alcohol", "flavanoids", "color_intensity", "proline"]
fig, axes = plt.subplots(len(pair_features), len(pair_features), figsize=(11, 11))
for i, fi in enumerate(pair_features):
    for j, fj in enumerate(pair_features):
        ax = axes[i, j]
        if i == j:
            for c in sorted(df.wine_class.unique()):
                ax.hist(df[df.wine_class == c][fi], bins=12, color=CLASS_COLORS[c], alpha=0.6)
        else:
            for c in sorted(df.wine_class.unique()):
                subset = df[df.wine_class == c]
                ax.scatter(subset[fj], subset[fi], color=CLASS_COLORS[c], s=10, alpha=0.7)
        if i == len(pair_features) - 1:
            ax.set_xlabel(fj, fontsize=9)
        if j == 0:
            ax.set_ylabel(fi, fontsize=9)
        ax.tick_params(labelsize=7)
fig.suptitle("Pairwise Relationships Between Key Features (colored by class)", fontsize=13)
plt.savefig(f"{OUT_IMG}/06_pairwise_grid.png", dpi=130)
plt.close()

print("\nEDA & visualization pipeline complete. All outputs saved to data/ and images/.")
