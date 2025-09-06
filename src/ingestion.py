import requests
import pandas as pd
from pathlib import Path
import os


def fetch_news(keyword: str, out_dir: Path) -> Path:
    """
    Fetch recent news articles about a keyword from NewsAPI
    and save them as a CSV in data/raw/.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY not found. Did you set it correctly?")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,   # free tier linit 100 req/day
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # Extract useful fields
    articles = [
        {
            "title": art.get("title"),
            "description": art.get("description"),
            "content": art.get("content"),
            "publishedAt": art.get("publishedAt"),
            "source": art.get("source", {}).get("name"),
        }
        for art in data.get("articles", [])
    ]

    df = pd.DataFrame(articles)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"news_{keyword.replace(' ', '_')}.csv"
    df.to_csv(out_path, index=False)

    print(f" Saved {len(df)} articles to {out_path}")
    return out_path


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    fetch_news("artificial intelligence", raw_dir)
