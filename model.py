"""
model.py — Flight Connection Risk Classifier
=============================================
Reads : preprocessed_dataset.csv   (saved by eda.ipynb Step 12)
Target: risk_label  {0=Risky, 1=Tight, 2=Safe}
Model : RandomForestClassifier  (class_weight='balanced', random_state=42)

Run   : python model.py
Output: rf_connection_risk.pkl  — trained model
        confusion_matrix.png    — evaluation plot
        feature_importance.png  — top-feature bar chart
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20
DATA_PATH    = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")
MODEL_OUT    = os.path.join(os.path.dirname(__file__), "rf_connection_risk.pkl")
SEP          = "=" * 65
LABEL_MAP    = {0: "Risky", 1: "Tight", 2: "Safe"}
RISK_COLORS  = ["#e74c3c", "#f39c12", "#2ecc71"]

# ─────────────────────────────────────────────────────────────
# Columns to DROP before training (with justification)
# ─────────────────────────────────────────────────────────────
DROP_COLS = {
    # ── Leakage ──────────────────────────────────────────────
    "time_margin_min":     "LEAKAGE   | directly encodes connection_time - delay - required; ~1:1 with label",
    # ── Redundant / derived from kept features ────────────────
    "required_time_min":   "REDUNDANT | = walking + security + immigration + congestion + baggage (all kept)",
    "walking_time_min":    "REDUNDANT | exact duplicate of terminal_walk_time_min",
    "baggage_time_min":    "CONSTANT  | always 10 — zero variance",
    # ── Raw datetime integers (encoded timestamps, not useful) ─
    "dep_sched":           "RAW TS    | already decomposed into departure_hour / day_of_week / scheduled_flight_min",
    "arr_sched":           "RAW TS    | already decomposed into scheduled_flight_min",
    "departure.scheduled": "RAW TS    | datetime string encoded as int — dep_sched covers it",
    "arrival.scheduled":   "RAW TS    | datetime string encoded as int — arr_sched covers it",
    # ── Identifiers / near-zero signal ───────────────────────
    "flight_date":         "IDENTIFIER| only 2 unique dates — no generalisation value",
    "departure.gate":      "IDENTIFIER| ~437 unique values; high-cardinality, no generalisation",
    "has_delay_info":      "NEAR-CONST| arrival_delay_min already removed in eda.ipynb; flag loses meaning",
}

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 1 — LOAD DATA")
print(SEP)

df = pd.read_csv(DATA_PATH)
print(f"Loaded  : {DATA_PATH}")
print(f"Shape   : {df.shape}")
print(f"Columns : {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# 2. DROP LEAKAGE / REDUNDANT / IDENTIFIER COLUMNS
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2 — COLUMN SELECTION (DROP DECISIONS)")
print(SEP)

print(f"\n{'Column':<35} {'Decision'}")
print("-" * 70)

cols_to_drop = []
for col, reason in DROP_COLS.items():
    if col in df.columns:
        cols_to_drop.append(col)
        print(f"  DROP  {col:<30} {reason}")
    else:
        print(f"  SKIP  {col:<30} (not present in dataset)")

df.drop(columns=cols_to_drop, inplace=True)

print(f"\nShape after drops : {df.shape}")
print(f"Remaining columns : {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3 — FEATURE / TARGET SPLIT")
print(SEP)

TARGET = "risk_label"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"Features ({len(FEATURES)}) : {FEATURES}")
print(f"\nTarget distribution:\n{y.value_counts().rename(LABEL_MAP).sort_index()}")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 4 — TRAIN / TEST SPLIT")
print(SEP)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED,
)

print(f"Train : {X_train.shape}   Test : {X_test.shape}")
print(f"Train label dist : {dict(y_train.value_counts().sort_index())}")
print(f"Test  label dist : {dict(y_test.value_counts().sort_index())}")

# ─────────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 5 — RANDOM FOREST TRAINING")
print(SEP)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=5,
    max_features="sqrt",
    class_weight="balanced",   # handles Safe(999) > Risky/Tight(750) imbalance
    random_state=SEED,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
print("Model trained successfully.")
print(f"  n_estimators : {rf.n_estimators}")
print(f"  max_features : {rf.max_features}")
print(f"  class_weight : {rf.class_weight}")

# ─────────────────────────────────────────────────────────────
# 6. EVALUATION — TEST SET
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 6 — EVALUATION (TEST SET)")
print(SEP)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

print("\nClassification Report:")
target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
print(classification_report(y_test, y_pred, target_names=target_names))

# OvR ROC-AUC
y_bin = label_binarize(y_test, classes=[0, 1, 2])
auc_ovr = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
print(f"Macro OvR ROC-AUC : {auc_ovr:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 7 — 5-FOLD CROSS-VALIDATION (full dataset)")
print(SEP)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Per-fold    : {[round(s, 4) for s in cv_scores]}")

# ─────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 8 — GENERATING PLOTS")
print(SEP)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.titleweight": "bold"})

# ── 8a. Confusion Matrix ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=0)
ax.set_title(
    f"Confusion Matrix — Random Forest\nTest Accuracy: {acc:.2%}  |  ROC-AUC: {auc_ovr:.4f}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved -> confusion_matrix.png")
plt.show()

# ── 8b. Feature Importance ────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

n_top = min(20, len(importances))
top_imp = importances.head(n_top)

palette = [
    "#2ecc71" if v >= top_imp.iloc[0] * 0.5 else
    "#f39c12" if v >= top_imp.iloc[0] * 0.2 else
    "#95a5a6"
    for v in top_imp.values
]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top_imp.index[::-1], top_imp.values[::-1],
               color=palette[::-1], edgecolor="white", height=0.7)
for bar, val in zip(bars, top_imp.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=11)
ax.set_title(f"Top {n_top} Feature Importances — Random Forest", fontsize=13)
ax.set_xlim(0, top_imp.iloc[0] * 1.15)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved -> feature_importance.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 9 — SAVE MODEL")
print(SEP)

model_payload = {
    "model":    rf,
    "features": FEATURES,
    "label_map": LABEL_MAP,
    "test_accuracy": round(acc, 4),
    "roc_auc_macro": round(auc_ovr, 4),
    "cv_mean": round(cv_scores.mean(), 4),
    "cv_std":  round(cv_scores.std(),  4),
}

joblib.dump(model_payload, MODEL_OUT)
print(f"Saved -> {MODEL_OUT}")

# ─────────────────────────────────────────────────────────────
# 10. QUICK INFERENCE EXAMPLE
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 10 — QUICK INFERENCE EXAMPLE (first 5 test rows)")
print(SEP)

sample = X_test.head(5).copy()
preds  = rf.predict(sample)
probs  = rf.predict_proba(sample)

for i, (idx, row) in enumerate(sample.iterrows()):
    actual    = LABEL_MAP[y_test.loc[idx]]
    predicted = LABEL_MAP[preds[i]]
    prob_str  = "  ".join(f"{LABEL_MAP[j]}={probs[i][j]:.2f}" for j in range(3))
    match     = "✔" if actual == predicted else "✘"
    print(f"  Row {idx:>4} | Actual={actual:<6} Pred={predicted:<6} {match}  | {prob_str}")

print(f"\n{SEP}")
print("DONE")
print(SEP)
print(f"  Model saved   -> {MODEL_OUT}")
print(f"  Test Accuracy : {acc:.2%}")
print(f"  ROC-AUC (OvR) : {auc_ovr:.4f}")
print(f"  CV Accuracy   : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
