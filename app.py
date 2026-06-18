"""
Streamlit frontend for Fake News Detection.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io
    3. Select repo, branch, and set main file path to `app.py`
"""

import streamlit as st
from predict import predict_article

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# ------------------------------------------------------------------
# Custom CSS for a cleaner look
# ------------------------------------------------------------------

st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
    }
    .fake-box {
        background-color: #ffe0e0;
        color: #b00020;
        border: 2px solid #b00020;
    }
    .true-box {
        background-color: #e0f7e0;
        color: #006600;
        border: 2px solid #006600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------

st.markdown('<div class="big-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Paste a news article headline and body below. '
    'The model will classify it as Fake or True news.</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    title_input = st.text_input("Article Title", placeholder="Enter the headline...")
    text_input = st.text_area(
        "Article Body",
        placeholder="Paste the full article text here...",
        height=220,
    )
    submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

# ------------------------------------------------------------------
# Prediction logic
# ------------------------------------------------------------------

if submitted:
    if not title_input.strip() and not text_input.strip():
        st.warning("Please enter at least a title or some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_article(title_input, text_input)

        label = result["label"]
        confidence = result["confidence"]
        is_fake = result["is_fake"]

        if is_fake:
            box_class = "fake-box"
            icon = "⚠️"
            verdict = "FAKE NEWS"
        else:
            box_class = "true-box"
            icon = "✅"
            verdict = "TRUE NEWS"

        confidence_text = ""
        if confidence is not None:
            confidence_text = f"<br><span style='font-size:1rem; font-weight:400;'>Confidence: {confidence*100:.1f}%</span>"

        st.markdown(
            f'<div class="result-box {box_class}">'
            f"{icon} {verdict}"
            f"{confidence_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Show raw JSON response"):
            st.json(result)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Built with scikit-learn (TF-IDF + Linear SVM) • FastAPI backend • Streamlit frontend"
)
