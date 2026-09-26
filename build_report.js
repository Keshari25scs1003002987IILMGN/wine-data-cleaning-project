const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  BorderStyle, PageBreak
} = require("docx");

const img = (name) => fs.readFileSync(`${__dirname}/${name}`);
const dim = (w, h, maxW = 560) => {
  const scale = Math.min(1, maxW / w);
  return { width: Math.round(w * scale), height: Math.round(h * scale) };
};

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ text, heading: level, spacing: { before: 300, after: 150 } });
}
function para(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 150 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function image(name, w, h, caption) {
  const d = dim(w, h);
  const children = [
    new Paragraph({
      children: [new ImageRun({ data: img(name), transformation: d, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 150, after: 60 },
    }),
  ];
  if (caption) {
    children.push(new Paragraph({
      children: [new TextRun({ text: caption, italics: true, size: 20, color: "555555" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }));
  }
  return children;
}
function codeBlock(lines) {
  const arr = lines.split("\n");
  return new Paragraph({
    children: arr.flatMap((l, i) => [
      new TextRun({ text: l, font: "Consolas", size: 18 }),
      ...(i < arr.length - 1 ? [new TextRun({ break: 1 })] : []),
    ]),
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    spacing: { before: 100, after: 200 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
    },
  });
}
function simpleTable(headerRow, dataRows, colWidths) {
  const rows = [headerRow, ...dataRows];
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((c, ci) => new TableCell({
        width: { size: colWidths[ci], type: WidthType.DXA },
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "2E4053" } : undefined,
        children: [new Paragraph({
          children: [new TextRun({ text: String(c), bold: ri === 0, color: ri === 0 ? "FFFFFF" : "000000" })],
          alignment: AlignmentType.CENTER,
        })],
      })),
    })),
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      // Title page
      new Paragraph({ text: "", spacing: { before: 600 } }),
      new Paragraph({
        children: [new TextRun({ text: "Week 6 Task", bold: true, size: 32, color: "1A5276" })],
        alignment: AlignmentType.CENTER,
      }),
      new Paragraph({
        children: [new TextRun({ text: "Integrative Capstone Project and Evaluation", bold: true, size: 40 })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 100, after: 100 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Predicting Wine Cultivar from Chemical Analysis — A Supervised and Unsupervised Learning Pipeline", italics: true, size: 24, color: "555555" })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Virtual Data Science with Python Trainee Internship — YuvaIntern", size: 22 })],
        alignment: AlignmentType.CENTER,
      }),
      new Paragraph({
        children: [new TextRun({ text: "Prepared by: Keshari", size: 22 })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 100 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Date: 26 September 2026", size: 22 })],
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // 1. Problem Statement
      heading("1. Problem Statement"),
      para("Wine producers and quality-control labs often want to verify a wine's declared cultivar (grape variety/origin) using only its chemical fingerprint, without relying on manual tasting or paperwork. This project asks two questions using the same chemical dataset:"),
      bullet("Supervised: Can a model predict which of three cultivars a wine belongs to, given 13 chemical measurements (alcohol, flavanoids, color intensity, etc.)?"),
      bullet("Unsupervised: If we ignore the cultivar labels entirely, does clustering the chemistry alone naturally rediscover the same three groups? This tests whether the cultivar signal is intrinsic to the chemistry, useful for scenarios with no labels at all."),

      // 2. Methodological Approach
      heading("2. Methodological Approach"),
      para("A standard end-to-end data science pipeline was followed: data collection, cleaning, exploratory analysis, supervised modeling, unsupervised modeling, and evaluation — implemented in a single reproducible Python script (pipeline.py, see GitHub repository)."),
      bullet("Supervised models: Logistic Regression (linear baseline) and Random Forest (non-linear ensemble), compared via 5-fold stratified cross-validation and a held-out test set."),
      bullet("Unsupervised model: KMeans clustering (k chosen via the elbow method), visualized with Principal Component Analysis (PCA), and scored against the true labels using the Adjusted Rand Index (ARI) purely for validation — never for training."),

      // 3. Data Collection
      heading("3. Data Collection"),
      para("Dataset: UCI Wine Recognition Dataset, publicly available and bundled with scikit-learn (load_wine()), originally sourced from https://archive.ics.uci.edu/dataset/109/wine. It contains 178 wine samples from three cultivars grown in the same region of Italy, each described by 13 chemical attributes (alcohol, malic acid, ash, alkalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, and proline)."),
      para("To make the cleaning stage of this report genuine rather than a formality, 5 duplicate rows and 6 missing values were deliberately injected into a working copy of the data (wine_raw_dirty.csv) before cleaning."),

      // 4. Data Cleaning
      heading("4. Data Cleaning"),
      bullet("Raw working rows (with injected issues): 183"),
      bullet("Duplicate rows detected and removed: 5"),
      bullet("Missing values detected: 6, imputed using per-feature median (robust to outliers, appropriate for skewed chemical measurements)"),
      bullet("Final clean dataset: 178 rows x 13 features, 0 missing values"),
      bullet("Outlier audit (IQR method): a small number of statistical outliers were found in malic_acid, ash, alcalinity_of_ash, magnesium and color_intensity (2-4 points each) — these were retained rather than removed, since they reflect genuine chemical variation between cultivars rather than data errors."),

      // 5. EDA
      heading("5. Exploratory Data Analysis"),
      para("Class balance: Cultivar A = 59, Cultivar B = 71, Cultivar C = 48 samples — reasonably balanced, so accuracy is a fair primary metric without needing to correct for class imbalance."),
      ...image("eda_class_distribution.png", 550, 400, "Figure 1: Class distribution across the three cultivars"),
      para("The correlation heatmap below reveals strong positive correlation between total_phenols, flavanoids and od280/od315_of_diluted_wines — all related to phenolic compounds — suggesting some redundancy that ensemble/tree models can naturally handle via feature selection during training."),
      ...image("eda_correlation_heatmap.png", 900, 700, "Figure 2: Feature correlation heatmap"),
      para("Boxplots of the four features that turned out to be most predictive (see Section 7) show visibly distinct distributions across cultivars — an early signal that classification would be highly feasible."),
      ...image("eda_feature_boxplots.png", 900, 700, "Figure 3: Distribution of top predictive features by cultivar"),

      // 6. Modeling
      heading("6. Modeling — Supervised and Unsupervised"),
      para("Supervised (classification):", { bold: true }),
      para("All 13 features were standardized (zero mean, unit variance) before modeling, then split 75/25 into train/test sets, stratified by class."),
      codeBlock(
`X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
for name, clf in models.items():
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)`
      ),
      simpleTable(
        ["Model", "5-Fold CV Accuracy", "Held-out Test Accuracy"],
        [
          ["Logistic Regression", "98.3% (+/- 1.4%)", "97.8%"],
          ["Random Forest (best)", "97.8% (+/- 2.1%)", "100.0%"],
        ],
        [3600, 2800, 2800]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      para("Unsupervised (clustering):", { bold: true }),
      para("KMeans was applied to the same standardized features, without using the cultivar labels at all. The number of clusters (k=3) was chosen independently via the elbow method on inertia, then compared post-hoc to the true labels only for evaluation."),
      ...image("clustering_elbow_method.png", 700, 480, "Figure 4: Elbow method confirming k=3 as a natural choice"),
      codeBlock(
`kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
ari_score = adjusted_rand_score(y, cluster_labels)   # 0.90
sil_score = silhouette_score(X_scaled, cluster_labels)  # 0.28`
      ),

      // 7. Evaluation
      heading("7. Evaluation of Model Performance"),
      para("The Random Forest classifier was selected as the best supervised model, achieving perfect accuracy on the held-out test set:"),
      ...image("model_confusion_matrix.png", 550, 500, "Figure 5: Confusion matrix — Random Forest on test set (45 samples)"),
      para("Precision, recall and F1-score were all 1.000 for every cultivar on the test set, and 5-fold cross-validation accuracy of 97.8% confirms this is not a lucky test split."),
      para("Feature importance shows which chemical measurements drove the predictions most:"),
      ...image("model_feature_importance.png", 800, 570, "Figure 6: Random Forest feature importances"),
      para("For the unsupervised side, the Adjusted Rand Index of 0.90 (1.0 = perfect agreement, 0.0 = random) shows that KMeans clustering — with no access to labels — recovered groupings that align very strongly with the true cultivars. The silhouette score of 0.28 is moderate, reflecting that while clusters are distinguishable, they are not perfectly separated in the full 13-dimensional feature space (some overlap is visible in the PCA projection)."),
      ...image("clustering_pca_comparison.png", 1500, 500, "Figure 7: PCA projection — true cultivar labels (left) vs. KMeans clusters (right)"),
      para("Implications: the strong agreement between unsupervised clusters and true cultivars implies the chemical fingerprint carries an almost sufficient signal on its own — useful in settings where labeling wines is expensive or delayed, since a first-pass clustering could pre-group unlabeled samples with high reliability before any manual confirmation."),

      // 8. Conclusion
      heading("8. Conclusion, Recommendations, and Further Work"),
      para("Both the supervised and unsupervised approaches strongly support the hypothesis that a wine's cultivar is well-predicted, and even naturally discoverable, from its chemistry alone. The Random Forest classifier achieved 100% test accuracy and 97.8% cross-validated accuracy, while unsupervised KMeans clustering matched the true cultivar groupings with an Adjusted Rand Index of 0.90."),
      para("Recommendations:", { bold: true }),
      bullet("Deploy the Random Forest model as a quick chemical-fingerprint verification check, using color_intensity, flavanoids, proline and alcohol as the primary signals."),
      bullet("Use clustering as an exploratory pre-screening tool on new, unlabeled batches before formal cultivar confirmation."),
      bullet("Given the small dataset (178 samples), validate on a larger, independently collected batch before any production use, to guard against overfitting to this specific vineyard/region."),
      para("Further work:", { bold: true }),
      bullet("Test additional models (SVM, gradient boosting) and hyperparameter tuning via grid search."),
      bullet("Explore hierarchical clustering and DBSCAN to see if they resolve the moderate silhouette score more cleanly than KMeans."),
      bullet("Investigate feature reduction (e.g., dropping the weakest 4-5 features) to build a cheaper, faster-to-measure model for field use."),

      // 9. Reference
      heading("9. Dataset and Code Reference"),
      para("Dataset: UCI Wine Recognition Dataset — https://archive.ics.uci.edu/dataset/109/wine"),
      para("GitHub repository: [insert your repository URL here after pushing, e.g. https://github.com/<your-username>/week6-capstone-wine-analysis]"),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(`${__dirname}/Week6_Capstone_Report.docx`, buf);
  console.log("Report written.");
});
