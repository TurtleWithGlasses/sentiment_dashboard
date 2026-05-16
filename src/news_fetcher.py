import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_BASE = "https://newsapi.org/v2/everything"

# English-language outlets based in Turkey or covering Turkey closely
TURKISH_DOMAINS = [
    "dailysabah.com",
    "hurriyetdailynews.com",
    "trtworld.com",
    "aa.com.tr",
    "bianet.org",
    "ahvalnews.com",
    "turkishminute.com",
    "duvarenglish.com",
    "turkeyagenda.com",
    "yenisafak.com",
    "sabah.com.tr",
]


def fetch_headlines(
    keyword: str,
    days_back: int = 7,
    page_size: int = 100,
    turkey_only: bool = False,
) -> pd.DataFrame:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY not set. Copy .env.example to .env and add your key.")

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": keyword,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    if turkey_only:
        params["domains"] = ",".join(TURKISH_DOMAINS)

    response = requests.get(NEWS_API_BASE, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'Unknown error')}")

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()

    rows = []
    for article in articles:
        published = article.get("publishedAt", "")
        try:
            published_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            published_dt = None

        rows.append({
            "title": article.get("title") or "",
            "description": article.get("description") or "",
            "source": (article.get("source") or {}).get("name", "Unknown"),
            "url": article.get("url") or "",
            "published_at": published_dt,
            "published_date": published_dt.date() if published_dt else None,
        })

    df = pd.DataFrame(rows)
    df = df[df["title"].str.strip() != ""]
    df = df[df["title"] != "[Removed]"]
    return df.reset_index(drop=True)
