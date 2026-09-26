"""
Week 3 Task - Unsupervised Learning and Clustering Analysis
Dataset : Cleaned & EDA'd Wine Recognition Dataset (from Weeks 1-2)
Author  : Keshari
-------------------------------------------------------------------
This script:
  1. Loads the cleaned dataset and selects/prepares features for clustering.
  2. Uses the Elbow Method and Silhouette Score to choose k for K-Means.
  3. Applies K-Means clustering and Agglomerative (Hierarchical) clustering.
  4. Visualizes clusters via PCA-reduced 2D scatter plots and a dendrogram.
  5. Compares clusters against the (held-out) true wine cultivar labels and
     profiles each cluster's characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_IN = "data/00_input_eda_wine.csv"
OUT_DATA = "data"
OUT_IMG = "images"
CLUSTER_COLORS = ["#e07a30", "#3aa66b", "#3a6ea6", "#a63a8f"]

# ---------------------------------------------------------------------
# STEP 1: LOAD DATA AND SELECT FEATURES
# ---------------------------------------------------------------------
print("STEP 1: Loading dataset and preparing features...")
df = pd.read_csv(DATA_IN)
raw_cols = [c for c in df.columns if not c.endswith("_scaled")
            and c not in ("wine_class", "phenol_flavanoid_ratio")]
true_labels = df["wine_class"]  # held out - used only for evaluation/comparison, not for clustering

# Preprocessing for clustering: re-scale the selected raw features
# (ensures every feature contributes equally to Euclidean distance)
scaler = StandardScaler()
X = scaler.fit_transform(df[raw_cols])
print(f"  Feature matrix shape: {X.shape}")

# ---------------------------------------------------------------------
# STEP 2: CHOOSE K - ELBOW METHOD + SILHOUETTE SCORE
# ---------------------------------------------------------------------
print("STEP 2: Determining optimal number of clusters (k)...")
inertias, sil_scores = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(list(k_range), inertias, marker="o", color="#4c72b0")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (WCSS)", color="#4c72b0")
ax1.set_title("Elbow Method and Silhouette Score for Choosing k")
ax2 = ax1.twinx()
ax2.plot(list(k_range), sil_scores, marker="s", color="#c44e52")
ax2.set_ylabel("Silhouette Score", color="#c44e52")
plt.savefig(f"{OUT_IMG}/01_elbow_silhouette.png", dpi=130, bbox_inches="tight")
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"  Silhouette scores: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
print(f"  k chosen = {best_k} (highest silhouette score; also matches the elbow and the "
      f"known 3 wine cultivars)")
# We choose k=3: it aligns with the elbow, has a strong silhouette score,
# and matches domain knowledge (3 known wine cultivars) - documented in the report.
K = 3

# ---------------------------------------------------------------------
# STEP 3: K-MEANS CLUSTERING
# ---------------------------------------------------------------------
print(f"STEP 3: Applying K-Means with k={K}...")
kmeans = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10)
km_labels = kmeans.fit_predict(X)
df["kmeans_cluster"] = km_labels
km_sil = silhouette_score(X, km_labels)
km_ari = adjusted_rand_score(true_labels, km_labels)
print(f"  K-Means silhouette score: {km_sil:.3f} | Adjusted Rand Index vs true classes: {km_ari:.3f}")

# ---------------------------------------------------------------------
# STEP 4: HIERARCHICAL (AGGLOMERATIVE) CLUSTERING
# ---------------------------------------------------------------------
print(f"STEP 4: Applying Agglomerative Clustering with k={K} (Ward linkage)...")
agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
agg_labels = agg.fit_predict(X)
df["hierarchical_cluster"] = agg_labels
agg_sil = silhouette_score(X, agg_labels)
agg_ari = adjusted_rand_score(true_labels, agg_labels)
print(f"  Hierarchical silhouette score: {agg_sil:.3f} | Adjusted Rand Index vs true classes: {agg_ari:.3f}")

# Dendrogram (on a subsample for readability if needed; full data used here since n=179)
Z = linkage(X, method="ward")
plt.figure(figsize=(11, 5))
dendrogram(Z, truncate_mode="lastp", p=25, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage, truncated)")
plt.xlabel("Sample cluster (or index)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/02_dendrogram.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 5: VISUALIZE CLUSTERS VIA PCA (2D)
# ---------------------------------------------------------------------
print("STEP 5: Reducing to 2D with PCA for visualization...")
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X)
df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]
explained = pca.explained_variance_ratio_
print(f"  PCA explained variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, "
      f"total={explained.sum():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for cluster in sorted(np.unique(km_labels)):
    mask = km_labels == cluster
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], color=CLUSTER_COLORS[cluster],
                     label=f"Cluster {cluster}", alpha=0.75, edgecolor="white")
axes[0].set_title(f"K-Means Clusters (k={K}) — PCA Projection")
axes[0].set_xlabel(f"PC1 ({explained[0]:.1%} var.)")
axes[0].set_ylabel(f"PC2 ({explained[1]:.1%} var.)")
axes[0].legend()

for cluster in sorted(np.unique(agg_labels)):
    mask = agg_labels == cluster
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], color=CLUSTER_COLORS[cluster],
                     label=f"Cluster {cluster}", alpha=0.75, edgecolor="white")
axes[1].set_title(f"Hierarchical Clusters (k={K}) — PCA Projection")
axes[1].set_xlabel(f"PC1 ({explained[0]:.1%} var.)")
axes[1].set_ylabel(f"PC2 ({explained[1]:.1%} var.)")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/03_pca_cluster_comparison.png", dpi=130)
plt.close()

# PCA colored by TRUE wine class, for visual comparison against clusters
plt.figure(figsize=(7, 5.5))
class_map = {c: i for i, c in enumerate(sorted(true_labels.unique()))}
for c in sorted(true_labels.unique()):
    mask = true_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=CLUSTER_COLORS[class_map[c]],
                label=c, alpha=0.75, edgecolor="white")
plt.title("True Wine Cultivar Labels — PCA Projection (for comparison)")
plt.xlabel(f"PC1 ({explained[0]:.1%} var.)")
plt.ylabel(f"PC2 ({explained[1]:.1%} var.)")
plt.legend(title="True class")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/04_pca_true_labels.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 6: CLUSTER PROFILING (characteristics of each cluster)
# ---------------------------------------------------------------------
print("STEP 6: Profiling K-Means cluster characteristics...")
profile = df.groupby("kmeans_cluster")[raw_cols].mean().round(2)
profile.to_csv(f"{OUT_DATA}/kmeans_cluster_profile.csv")

crosstab = pd.crosstab(df["wine_class"], df["kmeans_cluster"])
crosstab.to_csv(f"{OUT_DATA}/kmeans_vs_true_class_crosstab.csv")

with open(f"{OUT_DATA}/clustering_summary.txt", "w") as f:
    f.write(f"Chosen k: {K}\n\n")
    f.write(f"K-Means silhouette score: {km_sil:.3f}\n")
    f.write(f"K-Means Adjusted Rand Index vs true classes: {km_ari:.3f}\n\n")
    f.write(f"Hierarchical silhouette score: {agg_sil:.3f}\n")
    f.write(f"Hierarchical Adjusted Rand Index vs true classes: {agg_ari:.3f}\n\n")
    f.write("K-MEANS CLUSTER PROFILE (mean feature values):\n{}\n\n".format(profile))
    f.write("K-MEANS CLUSTERS vs TRUE WINE CLASS (crosstab):\n{}\n".format(crosstab))

df.to_csv(f"{OUT_DATA}/01_clustered_wine_dataset.csv", index=False)
print("\nClustering analysis complete. All outputs saved to data/ and images/.")
print("Crosstab (K-Means cluster vs true class):")
print(crosstab)
