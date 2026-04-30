import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("preprocessing1.csv")

# Split
X = df.drop("risk_label", axis=1)
y = df["risk_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Evaluation
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n=== Feature Importances (Random Forest) ===")
for feat, imp in importances.items():
    bar = "#" * int(imp * 100)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(importances))]
ax.barh(importances.index[::-1], importances.values[::-1],
        color=colors[::-1], edgecolor="white", height=0.7)
for i, (val, name) in enumerate(zip(importances.values[::-1], importances.index[::-1])):
    ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=11)
ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlim(0, importances.iloc[0] * 1.18)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150, bbox_inches="tight")
print("\nSaved -> rf_feature_importance.png")
plt.show()
