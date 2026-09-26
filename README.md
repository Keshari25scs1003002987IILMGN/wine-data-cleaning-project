# Data Acquisition, Cleaning & Preprocessing — Wine Recognition Dataset

**YuvaIntern — Virtual Data Science with Python Trainee — Week 1 Task**

## Objective
Acquire a publicly available dataset, perform data cleaning, and preprocess it using Python, while documenting every decision made.

## Dataset
- **Source:** [UCI Machine Learning Repository — Wine Recognition Dataset](https://archive.ics.uci.edu/dataset/109/wine), accessed locally via `sklearn.datasets.load_wine()`.
- **Description:** Results of a chemical analysis of wines grown in the same region of Italy, derived from three different cultivars (classes). 13 numeric chemical features (alcohol, malic acid, ash, magnesium, flavanoids, proline, etc.) plus a class label.
- **Note:** The original UCI dataset is already clean, so to genuinely demonstrate a data-cleaning workflow this project deterministically injects missing values, duplicate rows and outliers (fixed random seed = 42) before cleaning them back out. This is fully documented and reproducible — see `src/data_cleaning_pipeline.py`, Step 2.

## Project Structure
```
├── src/
│   └── data_cleaning_pipeline.py   # Full pipeline: acquire → explore → clean → preprocess
├── data/
│   ├── 00_raw_original_uci_wine.csv
│   ├── 01_messy_wine_dataset.csv
│   ├── 02_cleaned_preprocessed_wine.csv
│   ├── exploration_summary.txt
│   └── outlier_report.txt
├── images/                         # Diagnostic plots used in the report
├── requirements.txt
└── README.md
```

## Steps Performed
1. **Data Acquisition** — loaded the Wine Recognition dataset via scikit-learn.
2. **Initial Exploration** — shape, dtypes, missing-value counts, summary statistics.
3. **Duplicate Handling** — identified and removed exact duplicate rows.
4. **Missing Value Handling** — imputed numeric columns using the median, computed per wine class to preserve class-specific distributions.
5. **Outlier Handling** — detected outliers using the IQR method (1.5×IQR rule) and capped (winsorized) them rather than dropping rows, to preserve sample size.
6. **Preprocessing** — standardized numeric features with `StandardScaler` for downstream modeling readiness.
7. **Output** — saved the cleaned + scaled dataset and all diagnostic plots.

## How to Run
```bash
pip install -r requirements.txt
python src/data_cleaning_pipeline.py
```

## Key Findings
- 137 missing cells and 9 duplicate rows were introduced/detected across 178→188 rows.
- Median imputation grouped by class was chosen over global-mean imputation because the three wine cultivars have distinct chemical profiles (e.g. very different average `magnesium` and `proline`), so a single global fill value would blur class separability.
- IQR-based capping was chosen over row deletion so no observations were lost — important given the dataset's already-small size (178 rows).
- Post-cleaning, all missing values were resolved (0 remaining) and extreme outlier values in `magnesium` and `proline` were brought within 1.5×IQR bounds.

## Author
Keshari — YuvaIntern Virtual Data Science with Python Trainee Program
