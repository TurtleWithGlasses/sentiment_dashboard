import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download VADER lexicon on first run
nltk.download("vader_lexicon", quiet=True)

_analyzer = SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_text(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": _label(compound),
    }


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment columns to a headlines DataFrame."""
    if df.empty:
        return df

    # Combine title + description for better signal
    texts = (df["title"] + ". " + df["description"].fillna("")).tolist()
    results = [score_text(t) for t in texts]
    sentiment_df = pd.DataFrame(results)

    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
