import pandas as pd
from pathlib import Path
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
        
    # Remove non letters
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = text.lower().strip()

    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.is_alpha]
    return " ".join(tokens)

def preprocess(input_path: Path, out_dir: Path):
    df = pd.read_csv(input_path)
    
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()

    df["clean_text"] = df["text"].apply(clean_text)

    # Adding fake labels for now 
    df["label"] = (df.index % 2).map({0: 0, 1: 1})  # 0=negative, 1=positive

    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42)

    # TF-IDF Vectorization Term Frequency Inverse Document Frequency
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train_tfidf, y_train, X_test_tfidf, y_test), out_dir / "tfidf_data.pkl")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")

    print(f" Preprocessing complete. Data saved to {out_dir}")

if __name__ == "__main__":
    processed_file = Path("data/processed/clean_news.csv")
    features_dir = Path("data/features")
    preprocess(processed_file, features_dir)
