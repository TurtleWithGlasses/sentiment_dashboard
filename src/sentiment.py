import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_text(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": _label(compound),
    }


def analyze_dataframe(df: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    if df.empty:
        return df

    texts = (df["title"] + ". " + df["description"].fillna("")).tolist()
    results = [score_text(t) for t in texts]

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
