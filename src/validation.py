import pandas as pd
from pathlib import Path

def validate_and_clean(input_path: Path, out_dir: Path) -> Path:
    """
    Validate and clean ingested news data.
    - Drop missing/empty text rows
    - Drop duplicates
    - Ensure datetime type
    - Save cleaned data
    """
    df = pd.read_csv(input_path)

    # Drop rows with no text
    df = df.dropna(subset=["title", "content"])
    df = df[df["title"].str.strip() != ""]
    df = df[df["content"].str.strip() != ""]

    # Drop duplicates
    df = df.drop_duplicates(subset=["title", "content"])

    # Ensure publishedAt is datetime
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["publishedAt"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clean_news.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Validation complete. {len(df)} rows saved to {out_path}")
    return out_path


if __name__ == "__main__":
    raw_file = Path("data/raw/news_artificial_intelligence.csv")
    processed_dir = Path("data/processed")
    validate_and_clean(raw_file, processed_dir)
