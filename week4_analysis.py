import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

d = load_wine(as_frame=True); X, y = d.data, d.target
names = list(d.target_names)
info = {"shape": list(X.shape), "missing": int(X.isnull().sum().sum()), "class_counts": {names[k]: int(v) for k, v in y.value_counts().sort_index().items()}}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
 "Logistic Regression": Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=1000))]),
 "KNN": Pipeline([("s",StandardScaler()),("m",KNeighborsClassifier())]),
 "SVM (RBF)": Pipeline([("s",StandardScaler()),("m",SVC())]),
 "Random Forest": Pipeline([("m",RandomForestClassifier(n_estimators=200,random_state=42))]),
}
res = {}
for n, m in models.items():
    r = cross_validate(m, Xtr, ytr, cv=cv, scoring=["accuracy","f1_macro"])
    m.fit(Xtr, ytr); p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro")
    res[n] = dict(cv_acc=float(r["test_accuracy"].mean()), cv_std=float(r["test_accuracy"].std()),
                  cv_f1=float(r["test_f1_macro"].mean()), folds=[float(x) for x in r["test_accuracy"]],
                  test_acc=float(accuracy_score(yte,p)), precision=float(pr), recall=float(rc), f1=float(f1))
# tuning
grid = GridSearchCV(Pipeline([("m",RandomForestClassifier(random_state=42))]),
  {"m__n_estimators":[100,200,300],"m__max_depth":[None,3,5,8],"m__min_samples_split":[2,4]}, cv=cv, scoring="accuracy")
grid.fit(Xtr, ytr)
best = grid.best_estimator_; p = best.predict(Xte)
pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro")
tuned = dict(params={k.replace("m__",""):v for k,v in grid.best_params_.items()}, cv_acc=float(grid.best_score_),
             test_acc=float(accuracy_score(yte,p)), precision=float(pr), recall=float(rc), f1=float(f1))
tuned["params"]={k:(None if v is None else int(v)) for k,v in tuned["params"].items()}
cm = confusion_matrix(yte, p).tolist()
report = classification_report(yte, p, target_names=names, output_dict=True)
rep = {n:{k:float(v) for k,v in report[n].items()} for n in names}
# importance
rf = best.named_steps["m"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
pi = permutation_importance(best, Xte, yte, n_repeats=30, random_state=42)
pim = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
out = dict(info=info, res=res, tuned=tuned, cm=cm, report=rep, imp=imp.round(4).to_dict(), pimp=pim.round(4).to_dict(), names=names,
           split=[len(Xtr),len(Xte)], desc=X.describe().T[["mean","std","min","max"]].round(2).to_dict("index"))
json.dump(out, open("results.json","w"), indent=1)

plt.rcParams.update({"font.size":10})
# charts
fig, ax = plt.subplots(figsize=(6,4)); ConfusionMatrixDisplay(np.array(cm), display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - Tuned Random Forest (Test Set)"); plt.tight_layout(); plt.savefig("cm.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(6.5,4.5)); top=imp.head(10)[::-1]; ax.barh(top.index, top.values, color="#2E5C8A")
ax.set_title("Top 10 Feature Importances (Random Forest)"); ax.set_xlabel("Gini importance"); plt.tight_layout(); plt.savefig("fi.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(6.5,4)); ns=list(res); ax.bar(ns,[res[n]["cv_acc"] for n in ns], yerr=[res[n]["cv_std"] for n in ns], color="#2E5C8A", capsize=4)
ax.set_ylim(0.85,1.01); ax.set_ylabel("5-fold CV accuracy"); ax.set_title("Cross-Validation Accuracy by Model"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig("cv.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(6.5,4)); c=y.value_counts().sort_index(); ax.bar(names,c.values,color=["#2E5C8A","#5B9BD5","#A5C8E4"]); ax.set_title("Class Distribution"); ax.set_ylabel("Samples"); plt.tight_layout(); plt.savefig("cls.png", dpi=160); plt.close()
print(json.dumps(out, indent=1))
