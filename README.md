# Unsupervised Learning & Clustering Analysis — Wine Recognition Dataset

**YuvaIntern — Virtual Data Science with Python Trainee — Week 3 Task**

## Objective
Apply unsupervised learning (clustering) to a publicly available dataset, segment it into meaningful groups, and analyze each cluster's characteristics using Scikit-learn.

## Dataset
- **Source:** Cleaned Wine Recognition Dataset from Weeks 1-2 (179 samples, 13 chemical features, 3 known cultivars used only for post-hoc validation).

## Project Structure
```
├── src/
│   └── clustering_pipeline.py      # Full clustering pipeline
├── data/
│   ├── 00_input_eda_wine.csv               # Input (from Week 2)
│   ├── 01_clustered_wine_dataset.csv       # Output with cluster labels + PCA coords
│   ├── kmeans_cluster_profile.csv
│   ├── kmeans_vs_true_class_crosstab.csv
│   └── clustering_summary.txt
├── images/                         # Elbow/silhouette plot, dendrogram, PCA cluster plots
├── requirements.txt
└── README.md
```

## Steps Performed
1. **Preprocessing** — re-standardized features with StandardScaler for distance-based clustering.
2. **Choosing k** — Elbow Method + Silhouette Score across k = 2 to 8; k = 3 selected.
3. **K-Means clustering** (k=3, n_init=10, random_state=42).
4. **Hierarchical (Agglomerative) clustering** with Ward linkage, plus a dendrogram.
5. **Visualization** — PCA-reduced 2D scatter plots for both methods and for the true labels (for comparison only).
6. **Evaluation** — Silhouette Score and Adjusted Rand Index (vs. true cultivar labels, held out from clustering).
7. **Cluster profiling** — mean feature values per cluster, and a crosstab against true classes.

## How to Run
```bash
pip install -r requirements.txt
python src/clustering_pipeline.py
```

## Key Findings
- k = 3 is optimal by both the Elbow Method and Silhouette Score, matching the 3 known wine cultivars.
- K-Means: Silhouette = 0.292, Adjusted Rand Index vs. true classes = 0.916.
- Hierarchical (Ward): Silhouette = 0.284, Adjusted Rand Index vs. true classes = 0.863.
- Clusters are clearly distinguished by flavanoids, color intensity, alcohol and proline.

## Author
Keshari — YuvaIntern Virtual Data Science with Python Trainee Program
