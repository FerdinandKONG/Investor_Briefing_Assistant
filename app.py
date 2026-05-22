 import os
import re
import time
import html
from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st
from transformers import pipeline


# ============================================================
# Model Configuration
# ============================================================

DEFAULT_SENTIMENT_MODEL = "ProsusAI/finbert"
DEFAULT_NER_MODEL = "dslim/bert-base-NER"


# ============================================================
# Helper Data Structure
# ============================================================

@dataclass
class RiskSignal:
    name: str
    evidence: str


# ============================================================
# Utility Functions
# ============================================================

def read_config(name: str, default: str) -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except Exception:
        value = os.getenv(name, default)
    return str(value).strip() if value is not None else default


def normalize_sentiment_label(label: Any) -> str:
    if label is None:
        return ""

    text = str(label).strip().lower()

    positive_terms = {"positive", "pos", "bullish", "increase", "up", "label_2", "2"}
    negative_terms = {"negative", "neg", "bearish", "decrease", "down", "label_0", "0"}
    neutral_terms = {"neutral", "neu", "label_1", "1"}

    if text in positive_terms or "positive" in text or "bullish" in text:
        return "Positive"
    if text in negative_terms or "negative" in text or "bearish" in text:
        return "Negative"
    if text in neutral_terms or "neutral" in text:
        return "Neutral"

    return str(label).strip().title()


def coerce_prediction(raw_prediction: Any) -> dict:
    if isinstance(raw_prediction, list):
        if len(raw_prediction) == 0:
            return {"label": "Unknown", "score": 0.0}

        first = raw_prediction[0]

        if isinstance(first, list) and len(first) > 0:
            first = first[0]

        if isinstance(first, dict):
            return {
                "label": normalize_sentiment_label(first.get("label", "Unknown")),
                "score": float(first.get("score", 0.0)),
            }

    if isinstance(raw_prediction, dict):
        return {
            "label": normalize_sentiment_label(raw_prediction.get("label", "Unknown")),
            "score": float(raw_prediction.get("score", 0.0)),
        }

    return {"label": "Unknown", "score": 0.0}


def format_entities(raw_entities: list[dict]) -> list[dict]:
    formatted = []

    for item in raw_entities:
        entity = item.get("word") or item.get("entity") or ""
        entity_group = item.get("entity_group") or item.get("entity") or "ENTITY"
        score = float(item.get("score", 0.0))

        entity = str(entity).replace("##", "").strip()

        if not entity:
            continue

        formatted.append(
            {
                "Entity": entity,
                "Type": str(entity_group),
                "Confidence": round(score, 4),
            }
        )

    seen = set()
    unique_entities = []

    for row in formatted:
        key = (row["Entity"].lower(), row["Type"])
        if key not in seen:
            unique_entities.append(row)
            seen.add(key)

    return unique_entities


def detect_risk_signals(text: str) -> list[RiskSignal]:
    risk_dictionary = {
        "Revenue or earnings pressure": [
            "missed expectations", "lower-than-expected", "revenue fell", "profit fell",
            "loss widened", "weak earnings", "margin pressure", "decline in sales"
        ],
        "Market competition": [
            "competition", "rival", "market share", "price war", "competitive pressure"
        ],
        "Regulatory or legal risk": [
            "regulator", "regulatory", "lawsuit", "legal", "investigation", "fine",
            "penalty", "compliance"
        ],
        "Demand or delivery weakness": [
            "weak demand", "lower demand", "delivery fell", "shipments fell",
            "slowdown", "inventory buildup"
        ],
        "Debt or liquidity risk": [
            "debt", "liquidity", "cash flow", "bankruptcy", "default", "credit risk"
        ],
        "Operational disruption": [
            "shutdown", "supply chain", "production halt", "strike", "delay",
            "shortage", "recall"
        ],
        "Cost pressure": [
            "costs rose", "higher costs", "inflation", "layoff", "job cuts",
            "restructuring"
        ],
    }

    lowered = text.lower()
    results = []

    for risk_name, keywords in risk_dictionary.items():
        matched = []
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, lowered):
                matched.append(kw)

        if matched:
            results.append(RiskSignal(name=risk_name, evidence=", ".join(matched)))

    return results


def evaluate_correctness(expected_label: str, predicted_label: str) -> str:
    expected = normalize_sentiment_label(expected_label)
    predicted = normalize_sentiment_label(predicted_label)

    if not expected:
        return ""

    return "Correct" if expected == predicted else "Incorrect"


def build_beginner_explanation(
    sentiment: str,
    confidence: float,
    risk_signals: list[RiskSignal],
    entities: list[dict],
) -> str:
    sentiment = normalize_sentiment_label(sentiment)
    confidence_pct = confidence * 100

    if sentiment == "Positive":
        tone_explanation = (
            "The news appears positive, which may indicate favorable business momentum, "
            "stronger financial performance, or improved market expectations."
        )
    elif sentiment == "Negative":
        tone_explanation = (
            "The news appears negative, which may indicate business pressure, weaker market expectations, "
            "or potential downside risk."
        )
    elif sentiment == "Neutral":
        tone_explanation = (
            "The news appears neutral, meaning it does not strongly suggest either positive or negative "
            "business impact based on the model output."
        )
    else:
        tone_explanation = (
            "The model output is not clearly mapped to a standard financial sentiment class."
        )

    if risk_signals:
        risk_text = " Key risk themes detected: " + "; ".join(
            f"{item.name} ({item.evidence})" for item in risk_signals
        ) + "."
    else:
        risk_text = " No obvious rule-based risk theme was detected, but the original context should still be reviewed."

    if entities:
        entity_names = [row["Entity"] for row in entities[:5]]
        entity_text = " Main entities detected: " + ", ".join(entity_names) + "."
    else:
        entity_text = " No major named entities were detected by the NER model."

    explanation = (
        f"{tone_explanation} The model confidence is {confidence_pct:.1f}%."
        f"{risk_text}{entity_text}"
    )

    return explanation


# ============================================================
# Hugging Face Pipeline Loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_id: str, hf_token: str | None = None):
    kwargs = {
        "task": "text-classification",
        "model": model_id,
        "tokenizer": model_id,
    }
    if hf_token:
        kwargs["token"] = hf_token
    return pipeline(**kwargs)


@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_id: str, hf_token: str | None = None):
    kwargs = {
        "task": "token-classification",
        "model": model_id,
        "tokenizer": model_id,
        "aggregation_strategy": "simple",
    }
    if hf_token:
        kwargs["token"] = hf_token
    return pipeline(**kwargs)


# ============================================================
# Core Analysis Functions
# ============================================================

def analyze_news(text: str, sentiment_pipe, ner_pipe) -> dict:
    clean_text = " ".join(str(text or "").split())

    if not clean_text:
        raise ValueError("Please enter a financial news headline, sentence, or paragraph.")

    started = time.perf_counter()

    raw_sentiment = sentiment_pipe(clean_text, truncation=True, max_length=512)
    sentiment_result = coerce_prediction(raw_sentiment)

    ner_text = clean_text
    try:
        tokenizer = ner_pipe.tokenizer
        token_ids = tokenizer.encode(
            clean_text,
            add_special_tokens=False,
            truncation=True,
            max_length=510,
        )
        ner_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        ner_text = clean_text[:1500]

    raw_entities = ner_pipe(ner_text)
    entities = format_entities(raw_entities)

    risk_signals = detect_risk_signals(clean_text)

    elapsed = time.perf_counter() - started

    explanation = build_beginner_explanation(
        sentiment=sentiment_result["label"],
        confidence=sentiment_result["score"],
        risk_signals=risk_signals,
        entities=entities,
    )

    return {
        "text": clean_text,
        "sentiment": sentiment_result["label"],
        "confidence": sentiment_result["score"],
        "entities": entities,
        "risk_signals": risk_signals,
        "explanation": explanation,
        "runtime_sec": elapsed,
    }


def run_batch(
    df: pd.DataFrame,
    text_col: str,
    expected_col: str | None,
    sentiment_pipe,
    ner_pipe,
) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        text = row.get(text_col, "")

        if not str(text).strip():
            continue

        try:
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
                    "risk_signals": "; ".join(item.name for item in result["risk_signals"]),
                    "entities": "; ".join(item["Entity"] for item in result["entities"]),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "test_id": idx + 1,
                    "input_news": str(text),
                    "expected_label": row.get(expected_col, "") if expected_col else "",
                    "app_output": "Error",
                    "confidence": 0,
                    "runtime_sec": 0,
                    "correct_or_not": "Incorrect" if expected_col else "",
                    "risk_signals": "",
                    "entities": "",
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# UI Rendering Functions
# ============================================================

def render_metric_card(label: str, value: str, helper: str = ""):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_helper = html.escape(str(helper))

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
            <div class="metric-helper">{safe_helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(result: dict):
    confidence_pct = f"{result['confidence'] * 100:.1f}%"
    risk_count = len(result["risk_signals"])
    entity_count = len(result["entities"])
    safe_explanation = html.escape(result["explanation"])

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
            <p>{safe_explanation}</p>
            <p class="small-note">
                Detected {risk_count} risk theme(s) and {entity_count} named entity/entities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_template_csv() -> bytes:
    template = pd.DataFrame(
        {
            "text": [
                "Tesla shares fell after the company reported lower-than-expected quarterly deliveries amid rising competition in China.",
                "Apple reported stronger-than-expected revenue as services growth continued to support margins.",
                "The company announced that its board will meet next week to review the quarterly financial report.",
            ],
            "expected_label": [
                "Negative",
                "Positive",
                "Neutral",
            ],
        }
    )

    return template.to_csv(index=False).encode("utf-8")


# ============================================================
# Streamlit Page Setup
# ============================================================

st.set_page_config(
    page_title="Financial News Sentiment & Risk Briefing Assistant",
    page_icon="📈",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: #ffffff;
        border-radius: 18px;
        padding: 28px 28px 22px 28px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 30px rgba(15,23,42,0.16);
        margin-bottom: 18px;
    }

    .hero-kicker {
        font-size: 12px;
        color: #cbd5e1;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.22;
        margin-bottom: 10px;
    }

    .hero-sub {
        font-size: 15px;
        line-height: 1.7;
        color: #e5e7eb;
    }

    .section-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 18px;
        padding: 20px 20px 12px 20px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.04);
    }

    .brief-card,
    .metric-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.04);
        margin-bottom: 14px;
    }

    .metric-label,
    .section-label {
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

    .metric-helper,
    .small-note {
        color: #64748b;
        font-size: 13px;
        margin-top: 8px;
    }

    .info-box {
        border-left: 4px solid #2563eb;
        padding: 12px 14px;
        background: rgba(37, 99, 235, 0.08);
        border-radius: 10px;
        font-size: 13px;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    .disclaimer {
        border-left: 4px solid #f59e0b;
        padding: 12px 14px;
        background: rgba(245, 158, 11, 0.10);
        border-radius: 10px;
        font-size: 13px;
        margin-top: 20px;
    }

    .subtle-text {
        color: #94a3b8;
        font-size: 13px;
    }

    @media (prefers-color-scheme: dark) {
        .metric-label,
        .section-label,
        .metric-helper,
        .small-note {
            color: #94a3b8;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load Models
# ============================================================

sentiment_model_id = read_config("SENTIMENT_MODEL_ID", DEFAULT_SENTIMENT_MODEL)
ner_model_id = read_config("NER_MODEL_ID", DEFAULT_NER_MODEL)
hf_token = read_config("HF_TOKEN", "")

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Deep Learning Business Application</div>
        <div class="hero-title">Financial News Sentiment & Risk Briefing Assistant</div>
        <div class="hero-sub">
            This app helps beginner investors quickly understand financial news.
            It classifies sentiment, extracts key entities, detects possible risk themes,
            and generates a plain-English briefing.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("View model configuration and pipeline structure"):
    st.code(
        f"SENTIMENT_MODEL_ID = {sentiment_model_id}\nNER_MODEL_ID = {ner_model_id}",
        language="text",
    )
    st.markdown(
        """
        **Project pipeline structure**

        1. Hugging Face Pipeline 1: Financial sentiment classification  
        2. Hugging Face Pipeline 2: Named entity recognition  
        3. Additional business logic: Risk signal detection  
        4. Output: Beginner-friendly briefing
        """
    )

try:
    with st.spinner("Loading Hugging Face pipelines..."):
        sentiment_pipe = load_sentiment_pipeline(
            model_id=sentiment_model_id,
            hf_token=hf_token if hf_token else None,
        )
        ner_pipe = load_ner_pipeline(
            model_id=ner_model_id,
            hf_token=hf_token if hf_token else None,
        )
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.info(
        "Please check your model ID, requirements.txt, Hugging Face token, "
        "and Streamlit Cloud secrets."
    )
    st.stop()


# ============================================================
# Single News Analysis
# ============================================================

examples = [
    "Tesla shares fell after the company reported lower-than-expected quarterly deliveries amid rising competition in China.",
    "Apple reported stronger-than-expected revenue as services growth continued to support margins.",
    "The company announced that its board will meet next week to review the quarterly financial report.",
    "Nvidia shares rose after analysts highlighted strong demand for AI chips and data center products.",
    "Boeing faced renewed regulatory scrutiny after another production quality issue was reported.",
]

if "news_text" not in st.session_state:
    st.session_state.news_text = examples[0]

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Single News Briefing")

st.markdown(
    """
    <div class="info-box">
        Enter one financial news headline or short paragraph. The app will return sentiment,
        confidence, risk signals, named entities, and a plain-English explanation.
    </div>
    """,
    unsafe_allow_html=True,
)

sample_col1, sample_col2 = st.columns([3, 1])

with sample_col1:
    sample_choice = st.selectbox(
        "Try a sample news sentence",
        examples,
        index=0,
    )

with sample_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Use sample", use_container_width=True):
        st.session_state.news_text = sample_choice

with st.form("single_news_form", clear_on_submit=False):
    news_text = st.text_area(
        "Financial news input",
        key="news_text",
        height=150,
        placeholder="Paste a financial news headline, company announcement, or short market commentary.",
    )

    submitted = st.form_submit_button(
        "Run briefing",
        type="primary",
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    try:
        result = analyze_news(news_text, sentiment_pipe, ner_pipe)
        render_result_cards(result)

        tab_briefing, tab_entities, tab_risks, tab_export = st.tabs(
            ["Briefing", "Entities", "Risk Signals", "Export"]
        )

        with tab_briefing:
            st.markdown("**Input news**")
            st.write(result["text"])

            st.markdown("**Plain-English explanation**")
            st.write(result["explanation"])

        with tab_entities:
            if result["entities"]:
                st.dataframe(
                    pd.DataFrame(result["entities"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No named entities were detected by the NER pipeline.")

        with tab_risks:
            if result["risk_signals"]:
                risk_df = pd.DataFrame(
                    [
                        {
                            "Risk Signal": item.name,
                            "Keyword Evidence": item.evidence,
                        }
                        for item in result["risk_signals"]
                    ]
                )

                st.dataframe(
                    risk_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No rule-based risk signal was detected. "
                    "Please still review the original news before making any decision."
                )

        with tab_export:
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

            st.dataframe(export_df, use_container_width=True, hide_index=True)

            st.download_button(
                label="Download single result CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="single_news_briefing_result.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as exc:
        st.error(str(exc))


# ============================================================
# Batch Testing Section
# ============================================================

with st.expander("Experimental Testing Panel", expanded=False):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Batch Testing for Experimental Results")

    st.markdown(
        """
        <div class="info-box">
            This section is mainly used for project evaluation. Upload a labelled testing CSV
            to calculate app accuracy and generate records for the Experimental Results Excel file.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        st.download_button(
            label="Download batch testing CSV template",
            data=create_template_csv(),
            file_name="batch_testing_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with top_col2:
        uploaded_file = st.file_uploader(
            "Upload testing CSV",
            type=["csv"],
            label_visibility="visible",
        )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV file: {exc}")
            st.stop()

        if batch_df.empty:
            st.warning("The uploaded CSV is empty.")
        else:
            st.write("Preview of uploaded data")
            st.dataframe(batch_df.head(10), use_container_width=True)

            columns = list(batch_df.columns)

            config_col1, config_col2, config_col3 = st.columns([2, 2, 1])

            with config_col1:
                default_text_index = columns.index("text") if "text" in columns else 0
                text_col = st.selectbox(
                    "Select text column",
                    columns,
                    index=default_text_index,
                )

            with config_col2:
                expected_options = ["None"] + columns
                expected_default = (
                    expected_options.index("expected_label")
                    if "expected_label" in columns
                    else 0
                )

                expected_col_choice = st.selectbox(
                    "Select expected label column",
                    expected_options,
                    index=expected_default,
                )

            with config_col3:
                max_rows = st.number_input(
                    "Max rows",
                    min_value=1,
                    max_value=min(200, len(batch_df)),
                    value=min(50, len(batch_df)),
                    step=1,
                )

            if st.button("Run batch test", type="primary", use_container_width=True):
                expected_col = None if expected_col_choice == "None" else expected_col_choice

                with st.spinner("Running batch predictions..."):
                    result_df = run_batch(
                        df=batch_df.head(int(max_rows)),
                        text_col=text_col,
                        expected_col=expected_col,
                        sentiment_pipe=sentiment_pipe,
                        ner_pipe=ner_pipe,
                    )

                if result_df.empty:
                    st.warning("No valid text rows were processed.")
                else:
                    metric_cols = st.columns(2)

                    if expected_col:
                        valid_rows = result_df["correct_or_not"].isin(["Correct", "Incorrect"])
                        correct = (result_df["correct_or_not"] == "Correct").sum()
                        total = valid_rows.sum()
                        accuracy = correct / total if total else 0.0

                        with metric_cols[0]:
                            render_metric_card(
                                "App Accuracy",
                                f"{accuracy:.1%}",
                                f"{correct} correct out of {total} testing samples",
                            )

                    avg_runtime = result_df["runtime_sec"].mean()
                    with metric_cols[1 if expected_col else 0]:
                        render_metric_card(
                            "Average Runtime",
                            f"{avg_runtime:.2f}s",
                            "Average runtime per testing sample",
                        )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    buffer = StringIO()
                    result_df.to_csv(buffer, index=False)

                    st.download_button(
                        label="Download batch results CSV",
                        data=buffer.getvalue().encode("utf-8"),
                        file_name="batch_testing_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)