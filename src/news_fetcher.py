import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_BASE = "https://newsapi.org/v2/everything"

# Turkish news domains — covers all language modes (tr + en).
# The language filter in fetch_headlines handles which articles are returned.
TURKISH_DOMAINS = [
    # ── English-language Turkish outlets ──────────────────────────────────────
    "dailysabah.com",
    "hurriyetdailynews.com",
    "trtworld.com",
    "ahvalnews.com",
    "turkishminute.com",
    "duvarenglish.com",
    "turkeyagenda.com",
    "indyturk.com",
    "voaturkce.com",
    "yetkinreport.com",

    # ── Major print-origin news sites ─────────────────────────────────────────
    "sozcu.com.tr",
    "hurriyet.com.tr",
    "sabah.com.tr",
    "milliyet.com.tr",
    "cumhuriyet.com.tr",
    "yenisafak.com",
    "aksam.com.tr",
    "turkiyegazetesi.com.tr",
    "birgun.net",
    "evrensel.net",
    "yeniakit.com.tr",
    "karar.com",
    "korkusuz.com.tr",
    "milligazete.com.tr",
    "anayurtgazetesi.com",
    "yenicaggazetesi.com.tr",

    # ── TV channel web portals ─────────────────────────────────────────────────
    "ntv.com.tr",
    "haberturk.com",
    "cnnturk.com",
    "halktv.com.tr",
    "szctv.com.tr",
    "tele1.com.tr",
    "ekoltv.com.tr",
    "tgrthaber.com.tr",
    "ahaber.com.tr",
    "krttv.com.tr",
    "haberglobal.com.tr",
    "nowhaber.com.tr",

    # ── Public broadcasters and news agencies ─────────────────────────────────
    "trthaber.com",
    "aa.com.tr",
    "ankahaber.net",
    "iha.com.tr",
    "dha.com.tr",

    # ── Digital-native and independent outlets ─────────────────────────────────
    "t24.com.tr",
    "gazeteduvar.com.tr",
    "medyascope.tv",
    "diken.com.tr",
    "odatv.com",
    "gercekgundem.com",
    "kisadalga.net",
    "sol.org.tr",
    "kronos34.news",
    "gazetepencere.com",
    "bianet.org",
    "artigercek.com",
    "politikyol.com",
    "dokuz8haber.com",
    "serbestiyet.com",
    "fatihaltayli.com.tr",

    # ── International Turkish-language editions ────────────────────────────────
    "bbc.com",          # bbc.com/turkce
    "dw.com",           # dw.com/tr
    "tr.euronews.com",
    "tr.sputniknews.com",
    "qha.com.tr",

    # ── High-traffic news portals and aggregators ──────────────────────────────
    "ensonhaber.com",
    "haberler.com",
    "haber7.com",
    "internethaber.com",
    "sondakika.com",
    "mynet.com",        # mynet.com/haber

    # ── Economy and finance ────────────────────────────────────────────────────
    "ekonomim.com",
    "bloomberght.com",
    "paraanaliz.com",
    "fortuneturkey.com",
    "capital.com.tr",

    # ── Sports ────────────────────────────────────────────────────────────────
    "sporx.com",
    "fanatik.com.tr",
    "fotomac.com.tr",
    "ajansspor.com",
    "aspor.com.tr",
    "beinsports.com.tr",
    "mackolik.com",     # mackolik.com/haberler

    # ── Technology and science ─────────────────────────────────────────────────
    "donanimhaber.com",
    "webtekno.com",
    "shiftdelete.net",
    "webrazzi.com",
    "log.com.tr",
    "evrimagaci.org",

    # ── Media industry ────────────────────────────────────────────────────────
    "medyatava.com",
    "medyaradar.com",
    "medyakoridoru.com",

    # ── Regional / local ──────────────────────────────────────────────────────
    "yeniasir.com.tr",
    "egedesonsoz.com",
    "bursahakimiyet.com.tr",
    "kocaeligazetesi.com.tr",
    "guneydoguekspres.com",
    "pusulagazetesi.com.tr",
    "gaziantepolay.com",
]


def _parse_domains(raw: str) -> list[str]:
    """Split a comma/newline/space-separated domain string into a clean list."""
    import re
    parts = re.split(r"[,\n\s]+", raw.strip())
    return [p.strip().lower() for p in parts if p.strip()]


def _get_api_key() -> str | None:
    """Return the NewsAPI key from environment or Streamlit secrets."""
    key = os.getenv("NEWS_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("NEWS_API_KEY")
        except Exception:
            pass
    return key


def fetch_headlines(
    keyword: str,
    days_back: int = 7,
    page_size: int = 100,
    turkey_only: bool = False,
    language: str = "en",
    custom_domains: list[str] | None = None,
) -> pd.DataFrame:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("NEWS_API_KEY not set. Copy .env.example to .env and add your key.")

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": keyword,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": language,
        "pageSize": page_size,
        "apiKey": api_key,
    }

    domains = list(TURKISH_DOMAINS) if turkey_only else []
    if custom_domains:
        domains.extend(d for d in custom_domains if d not in domains)
    if domains:
        params["domains"] = ",".join(domains)

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
