import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()
_translator = GoogleTranslator(source="tr", target="en")


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def _translate(text: str) -> str:
    try:
        # Google Translate accepts up to 5000 chars; truncate to be safe
        return _translator.translate(text[:4000]) or text
    except Exception:
        return text  # fall back to original if translation fails


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


def score_text_tr(text: str) -> dict:
    translated = _translate(text)
    return score_text_en(translated)


def analyze_dataframe(df: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    if df.empty:
        return df

    texts = (df["title"] + ". " + df["description"].fillna("")).tolist()
    scorer = score_text_tr if language == "tr" else score_text_en
    results = [scorer(t) for t in texts]

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
