# Exploratory Data Analysis & Visualization — Wine Recognition Dataset

**YuvaIntern — Virtual Data Science with Python Trainee — Week 2 Task**

## Objective
Perform exploratory data analysis (EDA) and visualization on a public dataset, extracting meaningful insights using Pandas, Matplotlib and NumPy, building on the cleaned dataset from Week 1.

## Dataset
- **Source:** Cleaned output of the Week 1 task — the UCI Wine Recognition Dataset (`00_input_cleaned_wine.csv`), 179 samples across 3 cultivars (`class_0`, `class_1`, `class_2`), 13 chemical features.

## Project Structure
```
├── src/
│   └── eda_visualization_pipeline.py   # Full EDA + visualization pipeline
├── data/
│   ├── 00_input_cleaned_wine.csv               # Input (from Week 1)
│   ├── 01_eda_dataset_with_derived_feature.csv # Output with derived feature
│   ├── summary_statistics.csv
│   ├── class_distribution.csv
│   ├── class_wise_mean_aggregation.csv
│   └── eda_text_summary.txt
├── images/                             # All visualizations used in the report
├── requirements.txt
└── README.md
```

## Steps Performed
1. **Initial Analysis** — summary statistics (`describe()`), class distribution check.
2. **Transformations/Aggregations** — group-by mean per class; derived `phenol_flavanoid_ratio` feature.
3. **Visualizations** — class distribution bar chart, feature histograms, boxplots by class, correlation heatmap, scatter plot of the strongest correlated pair, and a manual pairwise scatter/histogram grid.
4. **Interpretation** — patterns and class-separating features documented in the report.

## How to Run
```bash
pip install -r requirements.txt
python src/eda_visualization_pipeline.py
```

## Key Findings
- Strongest correlation: `total_phenols` vs `flavanoids` (r = 0.83).
- `class_2` wines: lowest flavanoids/phenols, highest color intensity.
- `class_0` wines: highest proline and alcohol — the most full-bodied cultivar.
- No extreme anomalies remain, confirming the Week 1 cleaning pipeline was effective.

## Author
Keshari — YuvaIntern Virtual Data Science with Python Trainee Program
