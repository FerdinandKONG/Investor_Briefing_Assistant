import os
import time
from io import StringIO

import pandas as pd
import streamlit as st
from transformers import pipeline

from utils import (
    build_beginner_explanation,
    coerce_prediction,
    detect_risk_signals,
    evaluate_correctness,
    format_entities,
    normalize_sentiment_label,
)


DEFAULT_SENTIMENT_MODEL = "ProsusAI/finbert"
DEFAULT_NER_MODEL = "dslim/bert-base-NER"


def read_config(name: str, default: str) -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return str(os.getenv(name, default))


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_id: str):
    return pipeline(
        task="text-classification",
        model=model_id,
        tokenizer=model_id,
    )


@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_id: str):
    return pipeline(
        task="token-classification",
        model=model_id,
        tokenizer=model_id,
        aggregation_strategy="simple",
    )


def analyze_news(text: str, sentiment_pipe, ner_pipe) -> dict:
    clean_text = " ".join(str(text or "").split())
    if not clean_text:
        raise ValueError("Please enter a financial news sentence or paragraph.")

    started = time.perf_counter()
    raw_sentiment = sentiment_pipe(clean_text, truncation=True, max_length=512)
    sentiment_result = coerce_prediction(raw_sentiment)

    raw_entities = ner_pipe(clean_text, truncation=True, max_length=512)
    entities = format_entities(raw_entities)
    risks = detect_risk_signals(clean_text)
    elapsed = time.perf_counter() - started

    explanation = build_beginner_explanation(
        sentiment=sentiment_result["label"],
        confidence=sentiment_result["score"],
        risk_signals=risks,
        entities=entities,
    )

    return {
        "text": clean_text,
        "sentiment": sentiment_result["label"],
        "confidence": sentiment_result["score"],
        "entities": entities,
        "risk_signals": risks,
        "explanation": explanation,
        "runtime_sec": elapsed,
    }


def render_metric_card(label: str, value: str, helper: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(result: dict):
    confidence_pct = f"{result['confidence'] * 100:.1f}%"
    risk_count = len(result["risk_signals"])
    entity_count = len(result["entities"])

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Sentiment", result["sentiment"], "Financial news tone")
    with col2:
        render_metric_card("Confidence", confidence_pct, "Model probability")
    with col3:
        render_metric_card("Runtime", f"{result['runtime_sec']:.2f}s", "Single input inference")

    st.markdown(
        f"""
        <div class="brief-card">
            <div class="section-label">Beginner briefing</div>
            <p>{result["explanation"]}</p>
            <p class="small-note">Detected {risk_count} risk theme(s) and {entity_count} named entity/entities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_batch(df: pd.DataFrame, text_col: str, expected_col: str | None, sentiment_pipe, ner_pipe) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        text = row.get(text_col, "")
        if not str(text).strip():
            continue
        result = analyze_news(text, sentiment_pipe, ner_pipe)
        expected = normalize_sentiment_label(row.get(expected_col, "")) if expected_col else ""
        correctness = evaluate_correctness(expected, result["sentiment"]) if expected_col else ""
        rows.append(
            {
                "test_id": idx + 1,
                "input_news": result["text"],
                "expected_label": expected,
                "app_output": result["sentiment"],
                "confidence": round(result["confidence"], 4),
                "runtime_sec": round(result["runtime_sec"], 3),
                "correct_or_not": correctness,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Financial News Sentiment & Risk Briefing Assistant",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 4rem; }
    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: #ffffff;
        border-radius: 18px;
        padding: 24px 24px 20px 24px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 30px rgba(15,23,42,0.16);
        margin-bottom: 16px;
    }
    .hero-kicker {
        font-size: 12px;
        color: #cbd5e1;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.22;
        margin-bottom: 10px;
    }
    .hero-sub {
        font-size: 15px;
        line-height: 1.65;
        color: #e5e7eb;
    }
    .brief-card, .metric-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.04);
        margin-bottom: 14px;
    }
    .metric-label, .section-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-helper, .small-note {
        color: #64748b;
        font-size: 13px;
        margin-top: 8px;
    }
    .disclaimer {
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        background: rgba(245, 158, 11, 0.10);
        border-radius: 10px;
        font-size: 13px;
        margin-top: 16px;
    }
    @media (prefers-color-scheme: dark) {
        .metric-label, .section-label, .metric-helper, .small-note { color: #94a3b8; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

sentiment_model_id = read_config("SENTIMENT_MODEL_ID", DEFAULT_SENTIMENT_MODEL)
ner_model_id = read_config("NER_MODEL_ID", DEFAULT_NER_MODEL)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Hugging Face business application</div>
        <div class="hero-title">Financial News Sentiment & Risk Briefing Assistant</div>
        <div class="hero-sub">
            Paste a company news sentence or paragraph. The app classifies financial sentiment,
            extracts key entities, and turns the result into a beginner-friendly risk briefing.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model configuration")
    st.caption("Use Streamlit secrets or environment variables to switch to your fine-tuned model.")
    st.code(f"SENTIMENT_MODEL_ID={sentiment_model_id}\nNER_MODEL_ID={ner_model_id}", language="text")
    st.markdown(
        """
        The final project should replace the default sentiment model with the model your group fine-tuned
        and uploaded to Hugging Face.
        """
    )

try:
    with st.spinner("Loading Hugging Face pipelines..."):
        sentiment_pipe = load_sentiment_pipeline(sentiment_model_id)
        ner_pipe = load_ner_pipeline(ner_model_id)
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

examples = [
    "Tesla shares fell after the company reported lower-than-expected quarterly deliveries amid rising competition in China.",
    "Apple reported stronger-than-expected revenue as services growth continued to support margins.",
    "The company announced that its board will meet next week to review the quarterly financial report.",
]

if "news_text" not in st.session_state:
    st.session_state.news_text = examples[0]

st.subheader("Single News Briefing")
sample_choice = st.selectbox("Try a sample news sentence", examples, index=0)
if st.button("Use selected sample", use_container_width=True):
    st.session_state.news_text = sample_choice

with st.form("news_form", clear_on_submit=False):
    news_text = st.text_area(
        "Financial news input",
        key="news_text",
        height=150,
        placeholder="Paste a financial news headline, company announcement, or short market commentary.",
    )
    submitted = st.form_submit_button("Run briefing", type="primary", use_container_width=True)

if submitted:
    try:
        result = analyze_news(news_text, sentiment_pipe, ner_pipe)
        render_result_cards(result)

        tab_briefing, tab_entities, tab_risks, tab_download = st.tabs(
            ["Briefing", "Entities", "Risk Signals", "Export"]
        )

        with tab_briefing:
            st.markdown("**Input news**")
            st.write(result["text"])
            st.markdown("**Plain-English explanation**")
            st.write(result["explanation"])

        with tab_entities:
            if result["entities"]:
                st.dataframe(pd.DataFrame(result["entities"]), use_container_width=True, hide_index=True)
            else:
                st.info("No named entities were detected by the NER pipeline.")

        with tab_risks:
            if result["risk_signals"]:
                risk_df = pd.DataFrame(
                    [{"Risk Signal": item.name, "Keyword Evidence": item.evidence} for item in result["risk_signals"]]
                )
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            else:
                st.info("No rule-based risk signal was detected. Review the original news context before drawing conclusions.")

        with tab_download:
            export_df = pd.DataFrame(
                [
                    {
                        "input_news": result["text"],
                        "sentiment": result["sentiment"],
                        "confidence": round(result["confidence"], 4),
                        "runtime_sec": round(result["runtime_sec"], 3),
                        "risk_signals": "; ".join(item.name for item in result["risk_signals"]),
                        "entities": "; ".join(item["Entity"] for item in result["entities"]),
                        "explanation": result["explanation"],
                    }
                ]
            )
            st.download_button(
                label="Download result CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="single_news_briefing_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Batch Testing")
st.caption("Upload a CSV with a text column. If it also has expected labels, the app calculates testing accuracy.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    if batch_df.empty:
        st.warning("The uploaded CSV is empty.")
    else:
        columns = list(batch_df.columns)
        text_col = st.selectbox("Text column", columns, index=columns.index("text") if "text" in columns else 0)
        expected_options = ["None"] + columns
        expected_default = expected_options.index("expected_label") if "expected_label" in columns else 0
        expected_col_choice = st.selectbox("Expected label column", expected_options, index=expected_default)
        max_rows = st.slider("Maximum rows to test", min_value=1, max_value=min(200, len(batch_df)), value=min(50, len(batch_df)))

        if st.button("Run batch test", type="primary", use_container_width=True):
            expected_col = None if expected_col_choice == "None" else expected_col_choice
            with st.spinner("Running batch predictions..."):
                result_df = run_batch(batch_df.head(max_rows), text_col, expected_col, sentiment_pipe, ner_pipe)

            if expected_col and not result_df.empty:
                correct = (result_df["correct_or_not"] == "Correct").sum()
                total = result_df["correct_or_not"].isin(["Correct", "Incorrect"]).sum()
                accuracy = correct / total if total else 0.0
                render_metric_card("App Accuracy", f"{accuracy:.1%}", f"{correct} correct out of {total} testing samples")

            st.dataframe(result_df, use_container_width=True, hide_index=True)
            buffer = StringIO()
            result_df.to_csv(buffer, index=False)
            st.download_button(
                label="Download batch results CSV",
                data=buffer.getvalue().encode("utf-8"),
                file_name="batch_testing_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.markdown(
    """
    <div class="disclaimer">
        Disclaimer: This app is for educational and information briefing purposes only.
        It does not provide buy, sell, hold, or other investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)
