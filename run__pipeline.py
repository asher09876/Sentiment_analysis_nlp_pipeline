from pathlib import Path
from src.ingestion import fetch_news
from src.validation import validate_and_clean
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline():
    print(" Starting pipeline...")

    raw_dir = Path("data/raw")
    raw_file = fetch_news("artificial intelligence", raw_dir)

    processed_file = validate_and_clean(raw_file, Path("data/processed"))

    preprocess(processed_file, Path("data/features"))

    features_file = Path("data/features/tfidf_data.pkl")
    model_path, acc = train_model(features_file, Path("models"))

    evaluate_model(features_file, model_path, Path("reports"))

    print(" Pipeline finished successfully!")

if __name__ == "__main__":
    run_pipeline()
