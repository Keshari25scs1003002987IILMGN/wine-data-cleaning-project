"""
Week 1 Task - Data Acquisition, Cleaning, and Preprocessing
Dataset : Wine Recognition Dataset (UCI Machine Learning Repository)
Author  : Keshari
-------------------------------------------------------------------
This script:
  1. Acquires a publicly available dataset (UCI Wine Recognition Dataset).
  2. Injects realistic real-world data quality issues (missing values,
     duplicate rows, outliers) with a fixed random seed, since the original
     UCI dataset is already clean. This is done ONLY to demonstrate a
     genuine cleaning workflow, and is documented transparently in the
     accompanying report.
  3. Performs data exploration, cleaning (missing values, duplicates,
     outliers) and preprocessing (scaling).
  4. Saves the cleaned dataset and diagnostic plots used in the report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DATA = "data"
OUT_IMG = "images"

# ---------------------------------------------------------------------
# STEP 1: DATA ACQUISITION
# ---------------------------------------------------------------------
print("STEP 1: Acquiring dataset...")
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["target"] = df["target"].map({0: "class_0", 1: "class_1", 2: "class_2"})
df.rename(columns={"target": "wine_class"}, inplace=True)

df.to_csv(f"{OUT_DATA}/00_raw_original_uci_wine.csv", index=False)
print(f"  Raw shape: {df.shape}")

# ---------------------------------------------------------------------
# STEP 2: SIMULATE REAL-WORLD MESSINESS (documented, reproducible)
# ---------------------------------------------------------------------
print("STEP 2: Simulating missing values, duplicates and outliers...")
messy = df.copy()

# 2a. Missing values: randomly null ~6% of cells across numeric columns
numeric_cols = messy.select_dtypes(include=[np.number]).columns.tolist()
n_missing = int(0.06 * messy.shape[0] * len(numeric_cols))
rows = np.random.randint(0, messy.shape[0], n_missing)
cols = np.random.choice(numeric_cols, n_missing)
for r, c in zip(rows, cols):
    messy.loc[r, c] = np.nan

# 2b. Duplicate rows: duplicate 10 random rows
dup_rows = messy.sample(10, random_state=RANDOM_SEED)
messy = pd.concat([messy, dup_rows], ignore_index=True)

# 2c. Outliers: inject extreme values into 'magnesium' and 'proline'
outlier_idx = np.random.choice(messy.index, 6, replace=False)
messy.loc[outlier_idx[:3], "magnesium"] = messy["magnesium"].max() * 5
messy.loc[outlier_idx[3:], "proline"] = messy["proline"].max() * 4

messy.to_csv(f"{OUT_DATA}/01_messy_wine_dataset.csv", index=False)
print(f"  Messy shape: {messy.shape} | Missing cells: {messy.isna().sum().sum()} | "
      f"Duplicate rows: {messy.duplicated().sum()}")

# ---------------------------------------------------------------------
# STEP 3: INITIAL DATA EXPLORATION
# ---------------------------------------------------------------------
print("STEP 3: Initial data exploration...")
with open(f"{OUT_DATA}/exploration_summary.txt", "w") as f:
    f.write("SHAPE: {}\n\n".format(messy.shape))
    f.write("DTYPES:\n{}\n\n".format(messy.dtypes))
    f.write("MISSING VALUES PER COLUMN:\n{}\n\n".format(messy.isna().sum()))
    f.write("DUPLICATE ROWS: {}\n\n".format(messy.duplicated().sum()))
    f.write("DESCRIBE:\n{}\n".format(messy.describe().T))

# Plot: missing values per column
plt.figure(figsize=(9, 5))
missing_counts = messy.isna().sum().sort_values(ascending=False)
missing_counts = missing_counts[missing_counts > 0]
plt.bar(missing_counts.index, missing_counts.values, color="#e07a30")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Missing value count")
plt.title("Missing Values by Column (Before Cleaning)")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/01_missing_values_before.png", dpi=130)
plt.close()

# Plot: boxplots before outlier treatment (key columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].boxplot(messy["magnesium"].dropna())
axes[0].set_title("Magnesium (Before)")
axes[1].boxplot(messy["proline"].dropna())
axes[1].set_title("Proline (Before)")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/02_outliers_before.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 4: HANDLE DUPLICATES
# ---------------------------------------------------------------------
print("STEP 4: Removing duplicate rows...")
before = messy.shape[0]
clean = messy.drop_duplicates().reset_index(drop=True)
print(f"  Removed {before - clean.shape[0]} duplicate rows")

# ---------------------------------------------------------------------
# STEP 5: HANDLE MISSING VALUES (median imputation, grouped by class)
# ---------------------------------------------------------------------
print("STEP 5: Imputing missing values (median, grouped by wine_class)...")
for col in numeric_cols:
    clean[col] = clean.groupby("wine_class")[col].transform(
        lambda s: s.fillna(s.median())
    )
assert clean[numeric_cols].isna().sum().sum() == 0, "Missing values remain!"

# ---------------------------------------------------------------------
# STEP 6: HANDLE OUTLIERS (IQR capping / winsorization)
# ---------------------------------------------------------------------
print("STEP 6: Capping outliers using the IQR method...")
def cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)

outlier_report = {}
for col in numeric_cols:
    before_out = clean[col].copy()
    clean[col] = cap_outliers_iqr(clean[col])
    n_capped = (before_out != clean[col]).sum()
    if n_capped > 0:
        outlier_report[col] = int(n_capped)

with open(f"{OUT_DATA}/outlier_report.txt", "w") as f:
    for col, n in outlier_report.items():
        f.write(f"{col}: {n} value(s) capped\n")
print("  Outlier report:", outlier_report)

# Plot: boxplots after outlier treatment
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].boxplot(clean["magnesium"])
axes[0].set_title("Magnesium (After)")
axes[1].boxplot(clean["proline"])
axes[1].set_title("Proline (After)")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/03_outliers_after.png", dpi=130)
plt.close()

# Plot: missing values after cleaning (should be all zero)
plt.figure(figsize=(9, 5))
after_missing = clean.isna().sum()
plt.bar(after_missing.index, after_missing.values, color="#3aa66b")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Missing value count")
plt.title("Missing Values by Column (After Cleaning)")
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/04_missing_values_after.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 7: FEATURE PREPROCESSING (scaling)
# ---------------------------------------------------------------------
print("STEP 7: Scaling numeric features (StandardScaler)...")
scaler = StandardScaler()
scaled_values = scaler.fit_transform(clean[numeric_cols])
scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in numeric_cols])
final_df = pd.concat([clean[["wine_class"]], clean[numeric_cols], scaled_df], axis=1)

# Correlation heatmap for report
plt.figure(figsize=(9, 7))
corr = clean[numeric_cols].corr()
im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Feature Correlation Heatmap (Cleaned Data)")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/05_correlation_heatmap.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 8: SAVE FINAL CLEANED DATASET
# ---------------------------------------------------------------------
final_df.to_csv(f"{OUT_DATA}/02_cleaned_preprocessed_wine.csv", index=False)
print("STEP 8: Cleaned & preprocessed dataset saved.")
print(f"  Final shape: {final_df.shape}")
print("\nPipeline complete.")
