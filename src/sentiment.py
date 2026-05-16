import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()
_multilingual_pipe = None


def _get_multilingual_pipe():
    global _multilingual_pipe
    if _multilingual_pipe is None:
        from transformers import pipeline
        _multilingual_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            truncation=True,
            max_length=512,
            top_k=None,
        )
    return _multilingual_pipe


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_text_en(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": _label(compound),
    }


def score_text_multilingual(text: str) -> dict:
    pipe = _get_multilingual_pipe()
    # top_k=None returns all labels; result is a list of {label, score} dicts
    result = pipe(text)[0]
    score_map = {item["label"]: item["score"] for item in result}

    positive = score_map.get("Positive", 0.0)
    negative = score_map.get("Negative", 0.0)
    neutral = score_map.get("Neutral", 0.0)
    # compound mirrors VADER's range: positive pull vs negative pull
    compound = round(positive - negative, 4)

    return {
        "compound": compound,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "label": max(score_map, key=score_map.get),
    }


def analyze_dataframe(df: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    if df.empty:
        return df

    texts = (df["title"] + ". " + df["description"].fillna("")).tolist()
    scorer = score_text_multilingual if language == "tr" else score_text_en
    results = [scorer(t) for t in texts]

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
