import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral": "#95a5a6",
    "Negative": "#e74c3c",
}

EXTRA_STOPWORDS = {
    "says", "said", "new", "one", "will", "also", "us", "get",
    "year", "years", "like", "just", "make", "time", "could",
    "would", "first", "two", "three", "four", "five",
}


def trend_chart(df: pd.DataFrame) -> go.Figure:
    """Daily average compound sentiment over time."""
    daily = (
        df.groupby("published_date")["compound"]
        .mean()
        .reset_index()
        .rename(columns={"compound": "avg_sentiment"})
        .sort_values("published_date")
    )

    fig = go.Figure()

    # Zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.5)", line_width=1)

    # Filled area
    fig.add_trace(go.Scatter(
        x=daily["published_date"],
        y=daily["avg_sentiment"],
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#3498db", width=2.5),
        fillcolor="rgba(52,152,219,0.15)",
        marker=dict(size=7, color="#3498db"),
        hovertemplate="<b>%{x}</b><br>Avg sentiment: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="Sentiment Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Average Compound Score",
        yaxis=dict(range=[-1, 1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of positive / neutral / negative counts."""
    counts = df["label"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)

    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=[SENTIMENT_COLORS[l] for l in counts.index]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} articles<extra></extra>",
    ))
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def source_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart: avg sentiment per news source."""
    source_data = (
        df.groupby("source")["compound"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_sentiment", "count": "articles"})
        .nlargest(top_n, "articles")
        .sort_values("avg_sentiment")
    )

    colors = [
        SENTIMENT_COLORS["Positive"] if v >= 0.05
        else SENTIMENT_COLORS["Negative"] if v <= -0.05
        else SENTIMENT_COLORS["Neutral"]
        for v in source_data["avg_sentiment"]
    ]

    fig = go.Figure(go.Bar(
        x=source_data["avg_sentiment"],
        y=source_data["source"],
        orientation="h",
        marker_color=colors,
        customdata=source_data["articles"],
        hovertemplate="<b>%{y}</b><br>Avg sentiment: %{x:.3f}<br>Articles: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Avg Sentiment by Source (top {top_n})",
        xaxis=dict(range=[-1, 1], title="Avg Compound Score"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=max(250, top_n * 30),
    )
    return fig


def wordcloud_figure(df: pd.DataFrame, keyword: str) -> plt.Figure:
    """Return a matplotlib Figure with the word cloud."""
    all_text = " ".join(df["title"].tolist() + df["description"].fillna("").tolist())
    # Strip URLs and punctuation
    all_text = re.sub(r"http\S+|www\S+", "", all_text)
    all_text = re.sub(r"[^a-zA-Z\s]", " ", all_text)

    stopwords = STOPWORDS | EXTRA_STOPWORDS | {keyword.lower()}

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        max_words=80,
        colormap="RdYlGn",
        prefer_horizontal=0.8,
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig
