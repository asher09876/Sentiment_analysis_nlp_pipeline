import pandas as pd
from pathlib import Path

def validate_and_clean(input_path: Path, out_dir: Path) -> Path:
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["title", "content"])
    df = df[df["title"].str.strip() != ""] 
    df = df[df["content"].str.strip() != ""]

    df = df.drop_duplicates(subset=["title", "content"])

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce") #coerce- invalid dt are treated as Not a time
    df = df.dropna(subset=["publishedAt"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clean_news.csv"
    df.to_csv(out_path, index=False)

    print(f" Validation complete. {len(df)} rows saved to {out_path}")
    return out_path


if __name__ == "__main__":
    raw_file = Path("data/raw/news_artificial_intelligence.csv")
    processed_dir = Path("data/processed")
    validate_and_clean(raw_file, processed_dir)
