import streamlit as st
import pandas as pd

from src.news_fetcher import fetch_headlines, _parse_domains
from src.sentiment import analyze_dataframe
from src.visualizations import (
    trend_chart,
    distribution_chart,
    source_chart,
    wordcloud_figure,
)

st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="📰",
    layout="wide",
)

# Persist data across reruns so filter interactions don't reset the page
if "df" not in st.session_state:
    st.session_state.df = None
if "keyword_used" not in st.session_state:
    st.session_state.keyword_used = ""
if "region_used" not in st.session_state:
    st.session_state.region_used = "Worldwide"
if "language_used" not in st.session_state:
    st.session_state.language_used = "English"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 Sentiment Dashboard")
    st.markdown("Monitor news sentiment in real time.")

    keyword = st.text_input(
        "Keyword / Topic",
        value="economy",
        placeholder="e.g. earthquake, Türkiye, AI",
        help=(
            "Supports NewsAPI query syntax:\n"
            "• **Single word** — `economy`\n"
            "• **Phrase** — `\"world economy\"` (use quotes)\n"
            "• **AND** — `USA AND China`\n"
            "• **OR** — `Europe OR economy`\n"
            "• **Exclude** — `economy NOT crypto`"
        ),
    )
    source_region = st.radio(
        "Source Region",
        options=["Worldwide", "Turkey"],
        horizontal=True,
        help="'Turkey' limits results to Turkish news outlets.",
    )

    article_language = st.radio(
        "Article Language",
        options=["English", "Turkish"],
        horizontal=True,
        help="'Turkish' fetches Turkish-language articles. Each article is translated to English via Google Translate before sentiment scoring.",
    )

    st.markdown("**Custom Domains** *(optional)*")
    custom_domains_raw = st.text_area(
        "custom_domains_input",
        placeholder="bbc.com\nreuters.com\nnytimes.com",
        height=100,
        label_visibility="collapsed",
        help=(
            "Restrict results to specific sites. One domain per line (or comma-separated).\n"
            "Leave empty to fetch from all available sources.\n"
            "Added on top of the Turkey list when 'Turkey' region is selected."
        ),
    )

    days_back = st.slider("Look-back period (days)", min_value=1, max_value=7, value=7)
    max_articles = st.slider("Max articles", min_value=10, max_value=100, value=50, step=10)

    run = st.button("Analyze", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Data: [NewsAPI](https://newsapi.org) · "
        "Model: VADER (NLTK) · "
        "Built with Streamlit"
    )

# ── Fetch & analyze (only when button is clicked) ─────────────────────────────
if run:
    lang_code = "tr" if article_language == "Turkish" else "en"

    with st.spinner(f"Fetching headlines for **{keyword}**…"):
        try:
            df_raw = fetch_headlines(
                keyword,
                days_back=days_back,
                page_size=max_articles,
                turkey_only=(source_region == "Turkey"),
                language=lang_code,
                custom_domains=_parse_domains(custom_domains_raw) or None,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not fetch news: {e}")
            st.stop()

    if df_raw.empty:
        st.warning("No articles found. Try a different keyword or increase the look-back period.")
        st.stop()

    with st.spinner("Analyzing sentiment…"):
        st.session_state.df = analyze_dataframe(df_raw, language=lang_code)

    st.session_state.keyword_used = keyword
    st.session_state.region_used = source_region
    st.session_state.language_used = article_language

# ── Main area ──────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown(
        """
        ## Welcome
        Enter a keyword in the sidebar and click **Analyze** to see:
        - Live headlines from the past week
        - Per-article sentiment scores (Positive / Neutral / Negative)
        - Sentiment trend over time
        - Word cloud of the most frequent terms

        **Query examples**
        | Goal | What to type |
        |---|---|
        | Single topic | `earthquake` |
        | Exact phrase | `"world economy"` |
        | Both words required | `USA AND China` |
        | Either word | `Europe OR economy` |
        | Exclude a term | `economy NOT crypto` |
        | Combined | `"interest rates" AND (Fed OR ECB)` |
        """
    )
    st.stop()

df = st.session_state.df
keyword = st.session_state.keyword_used
region = st.session_state.region_used
language = st.session_state.language_used

# ── KPI metrics ───────────────────────────────────────────────────────────────
total = len(df)
avg_score = df["compound"].mean()
pos_pct = (df["label"] == "Positive").mean() * 100
neg_pct = (df["label"] == "Negative").mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Articles", total)
col2.metric("Avg Sentiment", f"{avg_score:+.3f}")
col3.metric("Positive", f"{pos_pct:.0f}%")
col4.metric("Negative", f"{neg_pct:.0f}%")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col, dist_col = st.columns([3, 1])
with chart_col:
    st.plotly_chart(trend_chart(df), use_container_width=True)
with dist_col:
    st.plotly_chart(distribution_chart(df), use_container_width=True)

src_col, wc_col = st.columns([1, 2])
with src_col:
    st.plotly_chart(source_chart(df), use_container_width=True)
with wc_col:
    st.subheader("Word Cloud")
    try:
        fig = wordcloud_figure(df, keyword)
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Word cloud unavailable: {e}")

st.divider()

# ── Article feed ──────────────────────────────────────────────────────────────
region_badge = "🇹🇷 Turkey" if region == "Turkey" else "🌍 Worldwide"
lang_badge = "🇹🇷 Turkish" if language == "Turkish" else "🇬🇧 English"
st.subheader(f"Headlines — {keyword!r}  ·  {region_badge}  ·  {lang_badge}")

LABEL_COLOR = {
    "Positive": "green",
    "Neutral": "gray",
    "Negative": "red",
}

sentiment_filter = st.multiselect(
    "Filter by sentiment",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"],
)

filtered = df[df["label"].isin(sentiment_filter)]

for _, row in filtered.iterrows():
    color = LABEL_COLOR.get(row["label"], "gray")
    score_str = f"{row['compound']:+.3f}"
    date_str = row["published_at"].strftime("%b %d, %H:%M") if pd.notna(row["published_at"]) else "—"

    with st.container():
        title_col, badge_col = st.columns([8, 1])
        with title_col:
            st.markdown(f"**[{row['title']}]({row['url']})**")
            desc = row.get("description", "")
            if desc:
                st.caption(desc[:180] + ("…" if len(desc) > 180 else ""))
            st.caption(f"🗞 {row['source']} · 🕒 {date_str}")
        with badge_col:
            st.markdown(
                f"<div style='text-align:center;padding:8px;border-radius:8px;"
                f"background:{'rgba(46,204,113,0.15)' if color=='green' else 'rgba(231,76,60,0.15)' if color=='red' else 'rgba(149,165,166,0.15)'};"
                f"color:{color};font-weight:bold;font-size:0.85rem'>"
                f"{row['label']}<br><span style='font-size:1.1rem'>{score_str}</span></div>",
                unsafe_allow_html=True,
            )
        st.divider()
