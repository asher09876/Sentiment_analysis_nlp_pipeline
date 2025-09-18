import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(features_path: Path, models_dir: Path):
    X_train, y_train, X_test, y_test = joblib.load(features_path)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Training complete. Accuracy = {acc:.2f}")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "log_reg_model.pkl"
    joblib.dump(model, model_path)

    return model_path, acc
    
if __name__ == "__main__":
    features_file = Path("data/features/tfidf_data.pkl")
    models_dir = Path("models")
    train_model(features_file, models_dir)
