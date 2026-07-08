import re
from collections import Counter

import pandas as pd
import streamlit as st


POSITIVE_WORDS = {
    "amazing", "awesome", "best", "better", "brilliant", "calm", "confident",
    "delight", "easy", "excellent", "excited", "fast", "favorite", "friendly",
    "glad", "good", "great", "happy", "helpful", "impressive", "improved",
    "joy", "like", "love", "loved", "nice", "perfect", "positive", "recommend",
    "smooth", "strong", "success", "thank", "thanks", "useful", "win", "wonderful",
}

NEGATIVE_WORDS = {
    "angry", "annoying", "awful", "bad", "broken", "bug", "confusing", "delay",
    "disappointed", "disaster", "fail", "failed", "frustrated", "hate", "hated",
    "hard", "issue", "late", "loss", "negative", "problem", "poor", "sad",
    "slow", "terrible", "unhappy", "useless", "weak", "worse", "worst", "wrong",
}

EXAMPLES = {
    "Positive product feedback": "The new dashboard is fast, helpful, and much easier to use.",
    "Negative support comment": "I am frustrated because the app is slow and the upload failed again.",
    "Neutral announcement": "The company released version 2.1 today with three interface updates.",
    "Mixed social post": "The design is great, but the checkout delay is annoying.",
}


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def analyze_sentiment(text):
    tokens = tokenize(text)
    counts = Counter(tokens)
    positive_hits = sum(counts[word] for word in POSITIVE_WORDS)
    negative_hits = sum(counts[word] for word in NEGATIVE_WORDS)
    total_signal = positive_hits + negative_hits

    if total_signal == 0:
        label = "neutral"
        confidence = 0.5
    elif positive_hits > negative_hits:
        label = "positive"
        confidence = positive_hits / total_signal
    elif negative_hits > positive_hits:
        label = "negative"
        confidence = negative_hits / total_signal
    else:
        label = "neutral"
        confidence = 0.5

    return {
        "sentiment": label,
        "confidence": round(confidence, 3),
        "positive_words": positive_hits,
        "negative_words": negative_hits,
        "tokens": len(tokens),
        "matched_positive": sorted(set(tokens) & POSITIVE_WORDS),
        "matched_negative": sorted(set(tokens) & NEGATIVE_WORDS),
    }


def sentiment_color(label):
    return {
        "positive": "#16a34a",
        "negative": "#dc2626",
        "neutral": "#64748b",
    }.get(label, "#64748b")


st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")

st.title("Twitter Sentiment Dashboard")
st.caption(
    "An interactive NLP portfolio page for a BERT fine-tuning project on "
    "positive, negative, and neutral social-media sentiment."
)

st.info(
    "This hosted version uses a transparent lexical analyzer because the fine-tuned "
    "BERT model artifact is not included in the repository. The notebook documents "
    "the BERT training workflow."
)

overview_tab, analyzer_tab, batch_tab, model_tab, learning_tab = st.tabs(
    ["Overview", "Analyzer", "Batch View", "Model Workflow", "Learning"]
)

with overview_tab:
    st.subheader("Project Idea")
    st.markdown(
        """
This project explores sentiment analysis for Twitter-style text. The notebook
fine-tunes a BERT sequence-classification model to classify posts as positive,
negative, or neutral.

The online app is designed as a portfolio-friendly dashboard: it lets reviewers
try example text, inspect scoring behavior, compare multiple posts, and
understand how the original BERT workflow was built.
"""
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Task", "Sentiment")
    col2.metric("Classes", "3")
    col3.metric("Model studied", "BERT")
    col4.metric("App", "Streamlit")

    st.subheader("Use Cases")
    st.markdown(
        """
- Monitor customer feedback themes
- Review social-media reaction to a product or event
- Compare positive, negative, and neutral language patterns
- Demonstrate NLP deployment and model explainability concepts
"""
    )

with analyzer_tab:
    st.subheader("Single Text Analyzer")
    selected_example = st.selectbox("Start with an example", ["Custom text"] + list(EXAMPLES.keys()))

    default_text = "" if selected_example == "Custom text" else EXAMPLES[selected_example]
    text = st.text_area(
        "Text to analyze",
        value=default_text,
        height=150,
        placeholder="Paste a tweet, review, or short customer comment...",
    )

    if text.strip():
        result = analyze_sentiment(text)
        color = sentiment_color(result["sentiment"])

        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"""
<div style="border-left: 8px solid {color}; padding: 1rem; background: rgba(148,163,184,.12); border-radius: 8px;">
  <h3 style="margin:0; text-transform: capitalize;">{result['sentiment']}</h3>
  <p style="margin:.25rem 0 0 0;">Confidence-style score: {result['confidence']:.0%}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            st.progress(result["confidence"])

        with right:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Tokens": result["tokens"],
                            "Positive signal": result["positive_words"],
                            "Negative signal": result["negative_words"],
                            "Matched positive words": ", ".join(result["matched_positive"]) or "-",
                            "Matched negative words": ", ".join(result["matched_negative"]) or "-",
                        }
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning("Enter text to analyze sentiment.")

with batch_tab:
    st.subheader("Batch Sentiment View")
    st.markdown("Paste one post per line to compare sentiment patterns across multiple examples.")
    batch_text = st.text_area(
        "Batch input",
        value="\n".join(EXAMPLES.values()),
        height=180,
    )

    rows = []
    for line in [item.strip() for item in batch_text.splitlines() if item.strip()]:
        result = analyze_sentiment(line)
        rows.append(
            {
                "Text": line,
                "Sentiment": result["sentiment"],
                "Confidence": result["confidence"],
                "Positive signal": result["positive_words"],
                "Negative signal": result["negative_words"],
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df["Sentiment"].value_counts())

with model_tab:
    st.subheader("BERT Fine-Tuning Workflow")
    workflow = pd.DataFrame(
        [
            {
                "Step": "1. Dataset",
                "Description": "Twitter sentiment dataset with positive, negative, and neutral labels.",
            },
            {
                "Step": "2. Preprocessing",
                "Description": "Clean text, tokenize, normalize casing, and prepare labels.",
            },
            {
                "Step": "3. Tokenization",
                "Description": "Use BERT tokenizer with padding, truncation, and attention masks.",
            },
            {
                "Step": "4. Fine-tuning",
                "Description": "Train TFBertForSequenceClassification for three sentiment classes.",
            },
            {
                "Step": "5. Deployment",
                "Description": "Serve predictions through a Streamlit interface once model artifacts are available.",
            },
        ]
    )
    st.dataframe(workflow, hide_index=True, use_container_width=True)

    st.warning(
        "The original app references a local Windows model path, so the trained BERT "
        "weights are not available to Streamlit Cloud from this repository."
    )

    st.markdown(
        """
#### To enable real BERT inference online

- Commit or externally host the trained tokenizer and model files.
- Replace the lexical analyzer with cached Hugging Face model loading.
- Add dependency pins for `tensorflow`, `transformers`, and `tokenizers`.
- Show model-backed confidence scores and evaluation metrics.
"""
    )

with learning_tab:
    st.subheader("What This Project Demonstrates")
    st.markdown(
        """
This project demonstrates an end-to-end NLP workflow: preparing social-media
text, fine-tuning a transformer classifier, and presenting results in an
interactive web app.

The hosted version also shows an important deployment lesson: model artifacts
must be portable. A notebook can train a strong model, but a public app needs
stable model files, reproducible dependencies, and clear fallback behavior when
the model is not available.
"""
    )

    st.markdown(
        """
#### Technical Concepts

- Text cleaning and normalization
- BERT tokenization
- Transformer sequence classification
- Sentiment dashboard design
- Cloud deployment readiness
"""
    )
