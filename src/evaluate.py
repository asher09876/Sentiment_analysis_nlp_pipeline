import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(features_path: Path, model_path: Path, reports_dir: Path):
    # Load data & model
    X_train, y_train, X_test, y_test = joblib.load(features_path)
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write(report)

    print(f" Evaluation complete. Accuracy = {acc:.3f}")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"])
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(reports_dir / "confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    features_file = Path("data/features/tfidf_data.pkl")
    model_file = Path("models/log_reg_model.pkl")
    reports_dir = Path("reports")
    evaluate_model(features_file, model_file, reports_dir)
