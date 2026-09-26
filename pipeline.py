"""
Week 6 Capstone Project — Integrative Data Science Pipeline
Problem: Can we predict a wine's cultivar (class) from its chemical analysis,
and do natural (unsupervised) groupings in the data agree with the true cultivars?

Dataset: UCI Wine Recognition Dataset (public; bundled with scikit-learn as load_wine()).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              silhouette_score, adjusted_rand_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

np.random.seed(42)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. DATA COLLECTION
# ---------------------------------------------------------------------------
raw = load_wine()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target
df["cultivar"] = df["target"].map({0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"})

# Inject a few realistic data-quality issues to demonstrate cleaning, then clean them
rng = np.random.default_rng(7)
dirty = df.copy()
dup_idx = rng.choice(dirty.index, size=5, replace=False)
dirty = pd.concat([dirty, dirty.loc[dup_idx]], ignore_index=True)
nan_rows = rng.choice(dirty.index, size=6, replace=False)
nan_cols = rng.choice(raw.feature_names, size=6)
for r, c in zip(nan_rows, nan_cols):
    dirty.loc[r, c] = np.nan
dirty.to_csv("wine_raw_dirty.csv", index=False)

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
report_lines = []
report_lines.append(f"Raw rows (with injected duplicates/nulls): {len(dirty)}")
n_dupes = dirty.duplicated().sum()
report_lines.append(f"Duplicate rows found: {n_dupes}")
n_nulls_before = dirty.isna().sum().sum()
report_lines.append(f"Missing values found: {n_nulls_before}")

clean = dirty.drop_duplicates().copy()
for col in raw.feature_names:
    if clean[col].isna().any():
        med = clean[col].median()
        clean[col] = clean[col].fillna(med)

n_nulls_after = clean.isna().sum().sum()
report_lines.append(f"Rows after de-duplication: {len(clean)}")
report_lines.append(f"Missing values after median imputation: {n_nulls_after}")

# Outlier check via IQR (report only; wine chemistry outliers are often genuine, not removed)
outlier_counts = {}
for col in raw.feature_names:
    q1, q3 = clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[col] = int(((clean[col] < lo) | (clean[col] > hi)).sum())
report_lines.append("Outliers per feature (IQR method): " + json.dumps(outlier_counts))

clean = clean.reset_index(drop=True)
clean.to_csv("wine_clean.csv", index=False)

with open("cleaning_report.txt", "w") as f:
    f.write("\n".join(report_lines))
print("\n".join(report_lines))

# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
X = clean[raw.feature_names]
y = clean["target"]

summary = X.describe().T
summary.to_csv("summary_statistics.csv")

class_counts = clean["cultivar"].value_counts()

plt.figure(figsize=(5.5, 4))
class_counts.plot(kind="bar", color=["#2E86C1", "#28B463", "#CA6F1E"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 7))
corr = X.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150)
plt.close()

top_feats = ["flavanoids", "color_intensity", "proline", "alcohol"]
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, feat in zip(axes.ravel(), top_feats):
    sns.boxplot(data=clean, x="cultivar", y=feat, ax=ax, palette="Set2")
    ax.set_title(feat)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("eda_feature_boxplots.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. MODELING — SUPERVISED (classification)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_results = {}
for name, clf in models.items():
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    model_results[name] = {
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "test_accuracy": float(test_acc),
        "y_pred": y_pred,
    }
    print(f"{name}: CV acc = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}, Test acc = {test_acc:.4f}")

best_name = max(model_results, key=lambda n: model_results[n]["test_accuracy"])
best_pred = model_results[best_name]["y_pred"]
best_clf = models[best_name]

report_txt = classification_report(y_test, best_pred, target_names=["Cultivar A", "Cultivar B", "Cultivar C"], digits=3)
print(f"\nBest model: {best_name}\n{report_txt}")

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["A", "B", "C"], yticklabels=["A", "B", "C"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix — {best_name} (acc={model_results[best_name]['test_accuracy']*100:.1f}%)")
plt.tight_layout()
plt.savefig("model_confusion_matrix.png", dpi=150)
plt.close()

if best_name == "Random Forest":
    importances = pd.Series(best_clf.feature_importances_, index=raw.feature_names).sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#1A5276")
    plt.gca().invert_yaxis()
    plt.title("Random Forest — Feature Importances")
    plt.tight_layout()
    plt.savefig("model_feature_importance.png", dpi=150)
    plt.close()
    importances.to_csv("feature_importances.csv")

with open("classification_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(report_txt)

# ---------------------------------------------------------------------------
# 5. MODELING — UNSUPERVISED (clustering)
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, cluster_labels)
ari_score = adjusted_rand_score(y, cluster_labels)
print(f"\nKMeans silhouette score: {sil_score:.4f}")
print(f"KMeans Adjusted Rand Index vs true cultivars: {ari_score:.4f}")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sc0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Set1", s=35, edgecolor="k", linewidth=0.3)
axes[0].set_title("PCA — Colored by TRUE Cultivar")
axes[0].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")

sc1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="Set2", s=35, edgecolor="k", linewidth=0.3)
axes[1].set_title(f"PCA — Colored by KMeans Cluster (ARI={ari_score:.2f})")
axes[1].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
plt.tight_layout()
plt.savefig("clustering_pca_comparison.png", dpi=150)
plt.close()

# Elbow method to justify k=3
inertias = []
k_range = range(1, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), inertias, marker="o")
plt.axvline(3, color="red", linestyle="--", alpha=0.6, label="chosen k=3")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster SSE)")
plt.title("Elbow Method for Optimal k")
plt.legend()
plt.tight_layout()
plt.savefig("clustering_elbow_method.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. SAVE ALL RESULTS FOR REPORT
# ---------------------------------------------------------------------------
results = {
    "n_samples_raw_dirty": int(len(dirty)),
    "n_duplicates_removed": int(n_dupes),
    "n_missing_imputed": int(n_nulls_before),
    "n_samples_clean": int(len(clean)),
    "class_counts": class_counts.to_dict(),
    "supervised": {name: {k: v for k, v in res.items() if k != "y_pred"} for name, res in model_results.items()},
    "best_model": best_name,
    "unsupervised": {
        "silhouette_score": float(sil_score),
        "adjusted_rand_index": float(ari_score),
        "pca_explained_variance": explained_var.tolist(),
    },
}
with open("results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll artifacts saved.")
