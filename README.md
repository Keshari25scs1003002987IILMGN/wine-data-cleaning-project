# Week 6 Capstone — Integrative Data Science Project

**Predicting wine cultivar from chemical analysis, and testing whether unsupervised
clustering rediscovers those cultivars without labels.**

Built as the final capstone for the *Virtual Data Science with Python* internship (YuvaIntern).

## Problem Statement
Given the chemical analysis of wines grown in the same region of Italy but derived
from three different cultivars, can we (a) **predict** the cultivar from its
chemistry (supervised classification), and (b) does **unsupervised clustering**
of the same chemistry naturally separate the wines into groups that agree with
the true cultivars — i.e., is the cultivar signal strong enough to be discovered
without labels at all?

## Dataset
- **Name:** UCI Wine Recognition Dataset
- **Access:** Public; bundled with scikit-learn (`sklearn.datasets.load_wine`),
  original source: https://archive.ics.uci.edu/dataset/109/wine
- **Size:** 178 samples, 13 numeric chemical features, 3 classes (cultivars)

## Pipeline (`pipeline.py`)
1. **Data collection** — load the public dataset (deliberately re-injected 5
   duplicate rows + 6 missing values to demonstrate a realistic cleaning step).
2. **Data cleaning** — de-duplication, median imputation, IQR outlier audit.
3. **EDA** — class balance, correlation heatmap, feature boxplots by cultivar.
4. **Modeling — supervised** — Logistic Regression and Random Forest, 5-fold
   stratified cross-validation, held-out test evaluation.
5. **Modeling — unsupervised** — KMeans clustering (k chosen via elbow method),
   PCA for 2D visualization, Adjusted Rand Index against true labels.
6. **Evaluation** — accuracy, precision/recall/F1, confusion matrix, feature
   importance, silhouette score, ARI.

## Results
| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 98.3% ± 1.4% | 97.8% |
| **Random Forest (best)** | 97.8% ± 2.1% | **100%** |

- **KMeans (k=3) vs true cultivars:** Adjusted Rand Index = **0.90**, silhouette
  score = 0.28 — the chemistry alone almost perfectly recovers the true cultivar
  groupings without ever seeing the labels.
- **Top predictive features:** color intensity, flavanoids, proline, alcohol.

## Files
| File | Description |
|---|---|
| `pipeline.py` | End-to-end pipeline: cleaning → EDA → supervised + unsupervised modeling → evaluation |
| `wine_raw_dirty.csv` | Raw data with injected duplicates/nulls (before cleaning) |
| `wine_clean.csv` | Cleaned dataset used for modeling |
| `cleaning_report.txt`, `results_summary.json`, `classification_report.txt`, `feature_importances.csv` | Generated reports/metrics |
| `eda_*.png`, `model_*.png`, `clustering_*.png` | All figures used in the report |
| `Week6_Capstone_Report.docx` | Full capstone report |

## How to run
```bash
pip install scikit-learn pandas matplotlib seaborn
python pipeline.py
```

## Key Challenges
- Balancing realistic data-cleaning steps against a dataset that is already
  fairly clean — resolved by deliberately injecting a controlled amount of
  duplication/missingness so the cleaning stage is genuine rather than a no-op.
- Choosing k for KMeans without label leakage — resolved with the elbow method
  on inertia, confirming k=3 independently of the known number of cultivars.
- Comparing "supervised-perfect" accuracy against a fair unsupervised baseline
  — used Adjusted Rand Index (chance-corrected) rather than raw label overlap.

## Author
Keshari — YuvaIntern Virtual Data Science with Python Trainee
