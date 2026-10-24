import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "readmitted_30_days"

# ---------------------------------------------------------------- 1. data
df = pd.read_csv("ml_dataset.csv")
y = df[TARGET]
X = df.drop(columns=[TARGET, "patient_id"])      # ID is not a feature

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)} "
      f"| readmission rate: {y.mean():.1%}")

# ---------------------------------------------------------------- 2. models
def make_pipe(model, scale=False):
    """Encoding + scaling + model in one object, so the backend loads one file."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler() if scale else "passthrough", num_cols),
    ])
    return Pipeline([("prep", pre), ("model", model)])

models = {
    "Logistic Regression": make_pipe(
        LogisticRegression(max_iter=1000, class_weight="balanced"), scale=True),
    "Random Forest": make_pipe(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                               class_weight="balanced", random_state=SEED)),
}

# ---------------------------------------------------------------- 3. compare
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

print("\n=== Test-set results (threshold 0.5) ===")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"\n{name}")
    print(f"  accuracy {(pred == y_test).mean():.3f} | "
          f"precision {precision_score(y_test, pred):.3f} | "
          f"recall {recall_score(y_test, pred):.3f} | "
          f"f1 {f1_score(y_test, pred):.3f} | "
          f"test AUC {roc_auc_score(y_test, proba):.3f} | "
          f"5-fold CV AUC {cv_auc:.3f}")
    print("  confusion matrix [[TN FP], [FN TP]]:")
    print(confusion_matrix(y_test, pred))

# ---------------------------------------------------------------- 4. final model = Logistic Regression
final = models["Logistic Regression"]

# Tune the threshold on TRAIN data only (cross-validated probabilities).
# Rule: keep recall >= 0.70, then pick the threshold with the best F1.
TARGET_RECALL = 0.70
train_proba = cross_val_predict(final, X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]
rows = []
for t in np.arange(0.20, 0.80, 0.01):
    p = (train_proba >= t).astype(int)
    rows.append((t, recall_score(y_train, p), f1_score(y_train, p)))
ok = [r for r in rows if r[1] >= TARGET_RECALL]
threshold = float(max(ok, key=lambda r: r[2])[0]) if ok else 0.5
print(f"\nTuned threshold: {threshold:.2f}")

proba = final.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
tuned = {
    "model": "Logistic Regression",
    "threshold": round(threshold, 2),
    "accuracy": round(float((pred == y_test).mean()), 3),
    "precision": round(precision_score(y_test, pred), 3),
    "recall": round(recall_score(y_test, pred), 3),
    "f1": round(f1_score(y_test, pred), 3),
    "roc_auc": round(roc_auc_score(y_test, proba), 3),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
}
print("\n=== Final model on test set (tuned threshold) ===")
print(json.dumps(tuned, indent=2))

# ---------------------------------------------------------------- 5. explainability
names = final.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(final.named_steps["model"].coef_[0], index=names)
print("\n=== Logistic Regression coefficients (+ raises risk, - lowers) ===")
print(coefs.sort_values(key=abs, ascending=False).round(3).to_string())

rf = models["Random Forest"]
imp = pd.Series(rf.named_steps["model"].feature_importances_,
                index=rf.named_steps["prep"].get_feature_names_out())
print("\n=== Random Forest feature importance (top 8) ===")
print(imp.sort_values(ascending=False).head(8).round(3).to_string())

# ---------------------------------------------------------------- 6. save for the backend
final.fit(X, y)                       # refit on ALL data before saving
Path("models").mkdir(exist_ok=True)
joblib.dump({"pipeline": final,
             "threshold": threshold,
             "model_name": "Logistic Regression",
             "feature_columns": list(X.columns)},
            "models/readmission_model.joblib")
Path("models/final_metrics.json").write_text(json.dumps(tuned, indent=2))
print("\nSaved models/readmission_model.joblib and models/final_metrics.json")