# Sentiment Dashboard

A real-time news sentiment analysis dashboard built with Streamlit. Type a keyword and instantly see how the media is covering it — with trend lines, sentiment scores, and a word cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![VADER](https://img.shields.io/badge/NLP-VADER%20Sentiment-green)

## Features

- **Live headline feed** — fetches the past 7 days of news via NewsAPI
- **Per-article sentiment** — each headline scored Positive / Neutral / Negative using VADER
- **Trend line** — daily average sentiment over time (Plotly)
- **Sentiment distribution** — donut chart breakdown
- **Source comparison** — which outlets are most positive or negative on your topic
- **Word cloud** — most frequent terms in the headlines (colored green → red)
- **Filterable feed** — click any sentiment label to filter the article list

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Charts | Plotly |
| NLP | VADER (vaderSentiment / NLTK) |
| News data | NewsAPI |
| Word cloud | wordcloud + matplotlib |
| Config | python-dotenv |

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/TurtleWithGlasses/sentiment_dashboard.git
cd sentiment_dashboard
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free NewsAPI key

Sign up at [newsapi.org](https://newsapi.org) — free tier gives 100 requests/day.

### 3. Configure

```bash
cp .env.example .env
# Open .env and paste your key:
# NEWS_API_KEY=your_key_here
```

### 4. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
sentiment_dashboard/
├── app.py                  # Streamlit UI — layout, metrics, charts, article feed
├── requirements.txt
├── .env.example
└── src/
    ├── news_fetcher.py     # NewsAPI wrapper → returns a clean pandas DataFrame
    ├── sentiment.py        # VADER scorer — compound score + Positive/Neutral/Negative label
    └── visualizations.py  # Plotly charts + matplotlib word cloud
```

## How It Works

1. User enters a keyword (e.g. `"earthquake"`, `"economy"`, `"Türkiye"`)
2. `news_fetcher.py` calls NewsAPI and returns up to 100 recent headlines as a DataFrame
3. `sentiment.py` runs VADER on `title + description` for each article
4. `visualizations.py` builds the trend line, donut, source bar chart, and word cloud
5. `app.py` renders everything in Streamlit with KPI metrics at the top

## Notes

- VADER runs locally — no GPU, no extra API cost
- NewsAPI free tier: 100 requests/day, headlines only (no full article text)
- `.env` is git-ignored — never commit your API key

## License

MIT
