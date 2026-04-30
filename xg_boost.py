import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("preprocessing1.csv")

# Split
X = df.drop("risk_label", axis=1)
y = df["risk_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Note: StandardScaler removed — XGBoost is tree-based and scale-invariant

# Model
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    verbosity=0,
)

xgb.fit(X_train, y_train)

# Prediction
y_pred = xgb.predict(X_test)

# Evaluation
print("=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importances (gain — average gain per split, most informative for XGB)
importances = pd.Series(
    xgb.get_booster().get_score(importance_type="gain"),
    name="gain"
).reindex(X.columns, fill_value=0.0).sort_values(ascending=False)

print("\n=== Feature Importances (XGBoost — Gain) ===")
for feat, imp in importances.items():
    bar = "#" * int(imp / importances.iloc[0] * 40)   # scaled bar
    print(f"  {feat:<35} {imp:>9.2f}  {bar}")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c" if i < 3 else "#9b59b6" for i in range(len(importances))]
ax.barh(importances.index[::-1], importances.values[::-1],
        color=colors[::-1], edgecolor="white", height=0.7)
for i, val in enumerate(importances.values[::-1]):
    ax.text(val + importances.iloc[0] * 0.01, i, f"{val:.1f}", va="center", fontsize=9)
ax.set_xlabel("Average Gain per Split", fontsize=11)
ax.set_title("Feature Importances — XGBoost (Gain)", fontsize=13, fontweight="bold")
ax.set_xlim(0, importances.iloc[0] * 1.18)
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150, bbox_inches="tight")
print("\nSaved -> xgb_feature_importance.png")
plt.show()
