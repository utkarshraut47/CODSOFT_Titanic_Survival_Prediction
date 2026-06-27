import warnings
warnings.filterwarnings("ignore")

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filename="Titanic-Dataset.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find '{filename}' in {script_dir}")
    return pd.read_csv(csv_path)


def summarize_dataset(df):
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")
    print("Shape of the dataset:", df.shape, "\n")
    print("Information about the dataset:")
    df.info()
    print("\nMissing values percentage per column:")
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print(missing_pct)


def plot_survival_distribution(df):
    plt.figure(figsize=(6, 4))
    counts = df["Survived"].value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values, color=["#d62728", "#1f77b4"])
    plt.title("Distribution of Survival (0 = No, 1 = Yes)")
    plt.xlabel("Survived")
    plt.ylabel("Count")
    plt.xticks([0, 1])
    plt.tight_layout()
    plt.show()


def preprocess_dataset(df):
    df = df.copy()
    df["Age"].fillna(df["Age"].median(), inplace=True)
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    drop_cols = [col for col in ["PassengerId", "Name", "Ticket"] if col in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    categorical_columns = [col for col in ["Embarked", "Pclass"] if col in df.columns]
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return df


def train_and_evaluate(df):
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError as error:
        print("scikit-learn is required to train the model. Install it with 'pip install scikit-learn'.")
        raise error

    if "Survived" not in df.columns:
        raise ValueError("The dataset must include a 'Survived' column.")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining data shape:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    return model, X_test, y_test, y_pred, cm


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0, 1], ["Did Not Survive", "Survived"], rotation=45, ha="right")
    plt.yticks([0, 1], ["Did Not Survive", "Survived"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(y_true, y_pred):
    plt.figure(figsize=(10, 4))
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    plt.subplot(1, 2, 1)
    plt.bar(true_counts.index.astype(str), true_counts.values, color="#1f77b4")
    plt.title("True Survival Distribution")
    plt.xlabel("Survived")
    plt.ylabel("Count")
    plt.xticks([0, 1])

    plt.subplot(1, 2, 2)
    plt.bar(pred_counts.index.astype(str), pred_counts.values, color="#ff7f0e")
    plt.title("Predicted Survival Distribution")
    plt.xlabel("Survived")
    plt.ylabel("Count")
    plt.xticks([0, 1])

    plt.tight_layout()
    plt.show()


def main():
    try:
        df = load_dataset()
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    summarize_dataset(df)
    plot_survival_distribution(df)

    cleaned_df = preprocess_dataset(df)
    print("\nDataFrame after cleaning and preprocessing:")
    print(cleaned_df.head(), "\n")
    print("Information about the preprocessed DataFrame:")
    cleaned_df.info()
    print("\nMissing values percentage after preprocessing:")
    print((cleaned_df.isnull().sum() / len(cleaned_df) * 100).round(2))

    model, X_test, y_test, y_pred, cm = train_and_evaluate(cleaned_df)
    plot_confusion_matrix(cm)
    plot_prediction_distribution(y_test, y_pred)

    importances = model.feature_importances_
    feature_names = cleaned_df.drop(columns=["Survived"]).columns
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#2ca02c")
    plt.title("Feature Importances for Titanic Survival Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

