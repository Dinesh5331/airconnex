import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "preprocessing1.csv")
DEFAULT_PLOT_PATH = os.path.join(BASE_DIR, "rf_feature_importance.png")


def train_random_forest(data_path=DEFAULT_DATA_PATH, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)

    X = df.drop("risk_label", axis=1)
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    return {
        "dataframe": df,
        "features": X.columns.tolist(),
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred),
        "feature_importances": importances,
    }


def plot_feature_importances(importances, output_path=DEFAULT_PLOT_PATH):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(importances))]
    ax.barh(
        importances.index[::-1],
        importances.values[::-1],
        color=colors[::-1],
        edgecolor="white",
        height=0.7,
    )
    for i, (value, name) in enumerate(
        zip(importances.values[::-1], importances.index[::-1])
    ):
        ax.text(value + 0.002, i, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=11)
    ax.set_title("Feature Importances - Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlim(0, importances.iloc[0] * 1.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {output_path}")
    plt.show()


def main():
    result = train_random_forest()

    print("=== Random Forest ===")
    print("Accuracy:", result["accuracy"])
    print(result["classification_report"])

    print("\n=== Feature Importances (Random Forest) ===")
    for feature, importance in result["feature_importances"].items():
        bar = "#" * int(importance * 100)
        print(f"  {feature:<35} {importance:.4f}  {bar}")

    plot_feature_importances(result["feature_importances"])


if __name__ == "__main__":
    main()
