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

# Stable sentiment models for model selection experiments.
# Important: The app only loads and uses ONE selected sentiment model at a time.
STABLE_SENTIMENT_MODELS = {
    "DistilRoBERTa Financial News": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "FinancialBERT Sentiment": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    "ProsusAI FinBERT": "ProsusAI/finbert",
}

DEFAULT_SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Second Hugging Face pipeline: Named Entity Recognition
STABLE_NER_MODELS = {
    "BERT Base NER": "dslim/bert-base-NER",
    "RoBERTa Large NER": "Jean-Baptiste/roberta-large-ner-english",
    "BERT Large CoNLL03 NER": "dbmdz/bert-large-cased-finetuned-conll03-english",
}

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
    """
    Read values from Streamlit secrets or environment variables.
    """
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except Exception:
        value = os.getenv(name, default)

    return str(value).strip() if value is not None else default


def normalize_sentiment_label(label: Any, model_id: str = "") -> str:
    """
    Normalize different Hugging Face model outputs into:
    Positive, Negative, Neutral.

    Different models may use different LABEL_0 / LABEL_1 / LABEL_2 mappings.
    This function prevents label mapping errors during model testing.
    """
    if label is None:
        return ""

    raw = str(label).strip()
    text = raw.lower()
    model_id = str(model_id or "").lower()

    # HKUST FinBERT Tone mapping:
    # LABEL_0 = Neutral, LABEL_1 = Positive, LABEL_2 = Negative
    if "yiyanghkust/finbert-tone" in model_id:
        mapping = {
            "label_0": "Neutral",
            "label_1": "Positive",
            "label_2": "Negative",
            "0": "Neutral",
            "1": "Positive",
            "2": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
            "negative": "Negative",
        }
        if text in mapping:
            return mapping[text]

    # FinancialBERT common mapping:
    # LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
    if "ahmedrachid/financialbert-sentiment-analysis" in model_id:
        mapping = {
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
        }
        if text in mapping:
            return mapping[text]

    # General text labels
    if "positive" in text or "bullish" in text:
        return "Positive"
    if "negative" in text or "bearish" in text:
        return "Negative"
    if "neutral" in text:
        return "Neutral"

    # General fallback mapping used by many 3-class sentiment models:
    # LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
    fallback_mapping = {
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
        "0": "Negative",
        "1": "Neutral",
        "2": "Positive",
        "pos": "Positive",
        "neg": "Negative",
        "neu": "Neutral",
    }

    if text in fallback_mapping:
        return fallback_mapping[text]

    return raw.title()


def coerce_prediction(raw_prediction: Any, model_id: str = "") -> dict:
    """
    Convert Hugging Face pipeline output into a standard format:
    {"label": ..., "score": ...}
    """
    if isinstance(raw_prediction, list):
        if len(raw_prediction) == 0:
            return {"label": "Unknown", "score": 0.0}

        first = raw_prediction[0]

        if isinstance(first, list) and len(first) > 0:
            first = first[0]

        if isinstance(first, dict):
            return {
                "label": normalize_sentiment_label(first.get("label", "Unknown"), model_id),
                "score": float(first.get("score", 0.0)),
                "raw_label": str(first.get("label", "Unknown")),
            }

    if isinstance(raw_prediction, dict):
        return {
            "label": normalize_sentiment_label(raw_prediction.get("label", "Unknown"), model_id),
            "score": float(raw_prediction.get("score", 0.0)),
            "raw_label": str(raw_prediction.get("label", "Unknown")),
        }

    return {"label": "Unknown", "score": 0.0, "raw_label": "Unknown"}


def format_entities(raw_entities: list[dict]) -> list[dict]:
    """
    Format NER output into a clean table.
    """
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

def normalize_entity_text(entity: Any) -> str:
    """
    Normalize entity text for fair matching.
    """
    text = str(entity or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def split_expected_entities(value: Any) -> list[str]:
    """
    Split expected entities from a semicolon-separated string.
    Example: Tesla; China
    """
    if value is None:
        return []

    items = str(value).split(";")
    entities = []

    for item in items:
        clean = normalize_entity_text(item)
        if clean:
            entities.append(clean)

    return entities


def calculate_entity_match_score(expected_entities: list[str], predicted_entities: list[dict]) -> tuple[float, str, str]:
    """
    Calculate NER entity match score.

    Score = number of matched expected entities / number of expected entities

    This is a simple and explainable metric for project experiments.
    """
    if not expected_entities:
        return 0.0, "", ""

    predicted_clean = [
        normalize_entity_text(item.get("Entity", ""))
        for item in predicted_entities
    ]

    matched = []
    missing = []

    for expected in expected_entities:
        found = False

        for predicted in predicted_clean:
            if expected == predicted or expected in predicted or predicted in expected:
                found = True
                break

        if found:
            matched.append(expected)
        else:
            missing.append(expected)

    score = len(matched) / len(expected_entities)

    return score, "; ".join(matched), "; ".join(missing)


def run_ner_batch(
    df: pd.DataFrame,
    text_col: str,
    expected_entities_col: str,
    ner_pipe,
    ner_model_id: str,
) -> pd.DataFrame:
    """
    Test Pipeline 2 only.

    Important:
    This function only runs the NER pipeline.
    It does not run sentiment classification.
    """
    rows = []

    for idx, row in df.iterrows():
        text = str(row.get(text_col, "")).strip()

        if not text:
            continue

        expected_entities = split_expected_entities(row.get(expected_entities_col, ""))

        try:
            started = time.perf_counter()

            # Pipeline 2 only: NER
            raw_entities = ner_pipe(text)
            predicted_entities = format_entities(raw_entities)

            elapsed = time.perf_counter() - started

            score, matched, missing = calculate_entity_match_score(
                expected_entities=expected_entities,
                predicted_entities=predicted_entities,
            )

            rows.append(
                {
                    "test_id": idx + 1,
                    "ner_model": ner_model_id,
                    "input_news": text,
                    "expected_entities": "; ".join(expected_entities),
                    "predicted_entities": "; ".join(item["Entity"] for item in predicted_entities),
                    "entity_match_score": round(score, 4),
                    "matched_entities": matched,
                    "missing_entities": missing,
                    "runtime_sec": round(elapsed, 3),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "test_id": idx + 1,
                    "ner_model": ner_model_id,
                    "input_news": text,
                    "expected_entities": "; ".join(expected_entities),
                    "predicted_entities": "Error",
                    "entity_match_score": 0,
                    "matched_entities": "",
                    "missing_entities": "; ".join(expected_entities),
                    "runtime_sec": 0,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)
def detect_risk_signals(text: str) -> list[RiskSignal]:
    """
    Rule-based business logic for identifying risk themes.
    This is not counted as a Hugging Face pipeline.
    """
    risk_dictionary = {
        "Revenue or earnings pressure": [
            "missed expectations", "lower-than-expected", "weaker-than-expected",
            "revenue fell", "profit fell", "loss widened", "weak earnings",
            "margin pressure", "decline in sales", "sales dropped"
        ],
        "Market competition": [
            "competition", "rival", "market share", "price war", "competitive pressure"
        ],
        "Regulatory or legal risk": [
            "regulator", "regulatory", "lawsuit", "legal", "investigation",
            "fine", "penalty", "compliance", "scrutiny", "export restrictions"
        ],
        "Demand or delivery weakness": [
            "weak demand", "lower demand", "delivery fell", "deliveries fell",
            "shipments fell", "slowdown", "inventory buildup"
        ],
        "Debt or liquidity risk": [
            "debt", "liquidity", "cash flow", "bankruptcy", "default", "credit risk"
        ],
        "Operational disruption": [
            "shutdown", "supply chain", "production halt", "strike", "delay",
            "shortage", "recall", "production quality", "wiring flaws"
        ],
        "Cost pressure": [
            "costs rose", "higher costs", "inflation", "layoff", "job cuts",
            "workforce cuts", "restructuring", "tariffs"
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
    """
    Compare expected label and app output.
    """
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
    """
    Build a short plain-English explanation for beginner investors.
    """
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

    return (
        f"{tone_explanation} The model confidence is {confidence_pct:.1f}%."
        f"{risk_text}{entity_text}"
    )


# ============================================================
# Hugging Face Pipeline Loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_id: str, hf_token: str | None = None):
    """
    Load only the selected sentiment model.
    Streamlit caches by model_id, so different models will not be mixed.
    """
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
    """
    Load the NER pipeline.
    """
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

def analyze_news(text: str, sentiment_pipe, ner_pipe, sentiment_model_id: str) -> dict:
    """
    Run the full app process using ONLY the currently selected sentiment model:
    1. Sentiment classification
    2. Named entity recognition
    3. Rule-based risk detection
    4. Beginner-friendly explanation
    """
    clean_text = " ".join(str(text or "").split())

    if not clean_text:
        raise ValueError("Please enter a financial news headline, sentence, or paragraph.")

    started = time.perf_counter()

    # Pipeline 1: selected sentiment model only
    raw_sentiment = sentiment_pipe(clean_text, truncation=True, max_length=512)
    sentiment_result = coerce_prediction(raw_sentiment, sentiment_model_id)

    # Pipeline 2: NER
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
        "raw_sentiment_label": sentiment_result.get("raw_label", ""),
        "confidence": sentiment_result["score"],
        "entities": entities,
        "risk_signals": risk_signals,
        "explanation": explanation,
        "runtime_sec": elapsed,
        "sentiment_model_id": sentiment_model_id,
    }


def run_batch(
    df: pd.DataFrame,
    text_col: str,
    expected_col: str | None,
    sentiment_pipe,
    ner_pipe,
    sentiment_model_id: str,
) -> pd.DataFrame:
    """
    Run batch testing using ONE selected sentiment model.
    This prevents different sentiment pipelines from being mixed in one test.
    """
    rows = []

    for idx, row in df.iterrows():
        text = row.get(text_col, "")

        if not str(text).strip():
            continue

        try:
            result = analyze_news(
                text=text,
                sentiment_pipe=sentiment_pipe,
                ner_pipe=ner_pipe,
                sentiment_model_id=sentiment_model_id,
            )

            expected = normalize_sentiment_label(row.get(expected_col, "")) if expected_col else ""
            correctness = evaluate_correctness(expected, result["sentiment"]) if expected_col else ""

            rows.append(
                {
                    "test_id": idx + 1,
                    "sentiment_model": sentiment_model_id,
                    "input_news": result["text"],
                    "expected_label": expected,
                    "raw_model_label": result["raw_sentiment_label"],
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
                    "sentiment_model": sentiment_model_id,
                    "input_news": str(text),
                    "expected_label": row.get(expected_col, "") if expected_col else "",
                    "raw_model_label": "Error",
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

    .warning-box {
        border-left: 4px solid #f59e0b;
        padding: 12px 14px;
        background: rgba(245, 158, 11, 0.10);
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
# Model Selection
# ============================================================

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

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Model Selection")

model_col1, model_col2 = st.columns(2)

with model_col1:
    selected_model_name = st.selectbox(
        "Select Pipeline 1 sentiment model",
        list(STABLE_SENTIMENT_MODELS.keys()),
        index=0,
    )

with model_col2:
    selected_ner_model_name = st.selectbox(
        "Select Pipeline 2 NER model",
        list(STABLE_NER_MODELS.keys()),
        index=0,
    )

sentiment_model_id = STABLE_SENTIMENT_MODELS[selected_model_name]
ner_model_id = STABLE_NER_MODELS[selected_ner_model_name]

sentiment_model_id = STABLE_SENTIMENT_MODELS[selected_model_name]
ner_model_id = read_config("NER_MODEL_ID", DEFAULT_NER_MODEL)

st.markdown(
    f"""
    <div class="info-box">
        Current sentiment model: <b>{html.escape(sentiment_model_id)}</b><br>
        Current NER model: <b>{html.escape(ner_model_id)}</b><br>
        Only the selected sentiment model will be used in single testing and batch testing.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("View pipeline structure"):
    st.markdown(
        """
        **Project pipeline structure**

        1. Hugging Face Pipeline 1: Financial sentiment classification  
        2. Hugging Face Pipeline 2: Named entity recognition  
        3. Additional business logic: Risk signal detection  
        4. Output: Beginner-friendly briefing  

        For experimental results, select one sentiment model, run the same testing CSV,
        download the result, and then repeat the process for another model.
        """
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Load Selected Pipelines
# ============================================================

try:
    with st.spinner(f"Loading selected sentiment pipeline: {selected_model_name}"):
        sentiment_pipe = load_sentiment_pipeline(
            model_id=sentiment_model_id,
            hf_token=hf_token if hf_token else None,
        )

    with st.spinner("Loading NER pipeline..."):
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
        result = analyze_news(
            text=news_text,
            sentiment_pipe=sentiment_pipe,
            ner_pipe=ner_pipe,
            sentiment_model_id=sentiment_model_id,
        )

        render_result_cards(result)

        tab_briefing, tab_entities, tab_risks, tab_export = st.tabs(
            ["Briefing", "Entities", "Risk Signals", "Export"]
        )

        with tab_briefing:
            st.markdown("**Model used**")
            st.write(result["sentiment_model_id"])

            st.markdown("**Input news**")
            st.write(result["text"])

            st.markdown("**Raw model label**")
            st.write(result["raw_sentiment_label"])

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
                        "sentiment_model": result["sentiment_model_id"],
                        "input_news": result["text"],
                        "raw_model_label": result["raw_sentiment_label"],
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
# Experimental Testing Panel
# ============================================================

with st.expander("Experimental Testing Panel", expanded=False):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Batch Testing for Experimental Results")

    st.markdown(
        f"""
        <div class="warning-box">
            This section is for project evaluation. It uses only the currently selected sentiment model:
            <b>{html.escape(sentiment_model_id)}</b>.
            To compare models, run this batch test once for each selected model and record the results separately.
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

            if st.button("Run batch test for selected model", type="primary", use_container_width=True):
                expected_col = None if expected_col_choice == "None" else expected_col_choice

                with st.spinner(f"Running batch predictions using {selected_model_name} only..."):
                    result_df = run_batch(
                        df=batch_df.head(int(max_rows)),
                        text_col=text_col,
                        expected_col=expected_col,
                        sentiment_pipe=sentiment_pipe,
                        ner_pipe=ner_pipe,
                        sentiment_model_id=sentiment_model_id,
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

                    st.markdown("**Batch testing results**")
                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    buffer = StringIO()
                    result_df.to_csv(buffer, index=False)

                    safe_model_name = selected_model_name.lower().replace(" ", "_").replace("/", "_")
                    file_name = f"batch_results_{safe_model_name}.csv"

                    st.download_button(
                        label="Download batch results CSV",
                        data=buffer.getvalue().encode("utf-8"),
                        file_name=file_name,
                        mime="text/csv",
                        use_container_width=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Pipeline 2 NER Testing Panel
# ============================================================

with st.expander("Pipeline 2 NER Testing Panel", expanded=False):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Pipeline 2: Named Entity Recognition Testing")

    st.markdown(
        f"""
        <div class="warning-box">
            This section tests Pipeline 2 only. It uses the currently selected NER model:
            <b>{html.escape(ner_model_id)}</b>.
            It does not run the sentiment classification pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Your CSV file should contain at least two columns:

        - `text`: financial news sentence  
        - `expected_entities`: manually labelled entities separated by semicolons  

        Example: `Tesla; China`
        """
    )

    ner_uploaded_file = st.file_uploader(
        "Upload NER testing CSV",
        type=["csv"],
        key="ner_testing_csv",
    )

    if ner_uploaded_file is not None:
        try:
            ner_df = pd.read_csv(ner_uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV file: {exc}")
            st.stop()

        if ner_df.empty:
            st.warning("The uploaded CSV is empty.")
        else:
            st.write("Preview of uploaded NER testing data")
            st.dataframe(ner_df.head(10), use_container_width=True)

            ner_columns = list(ner_df.columns)

            ner_config_col1, ner_config_col2, ner_config_col3 = st.columns([2, 2, 1])

            with ner_config_col1:
                ner_text_col = st.selectbox(
                    "Select text column for NER",
                    ner_columns,
                    index=ner_columns.index("text") if "text" in ner_columns else 0,
                    key="ner_text_col",
                )

            with ner_config_col2:
                expected_entities_col = st.selectbox(
                    "Select expected entities column",
                    ner_columns,
                    index=ner_columns.index("expected_entities") if "expected_entities" in ner_columns else 0,
                    key="expected_entities_col",
                )

            with ner_config_col3:
                ner_max_rows = st.number_input(
                    "NER max rows",
                    min_value=1,
                    max_value=min(200, len(ner_df)),
                    value=min(30, len(ner_df)),
                    step=1,
                    key="ner_max_rows",
                )

            if st.button("Run Pipeline 2 NER test", type="primary", use_container_width=True):
                with st.spinner(f"Running NER test using {selected_ner_model_name} only..."):
                    ner_result_df = run_ner_batch(
                        df=ner_df.head(int(ner_max_rows)),
                        text_col=ner_text_col,
                        expected_entities_col=expected_entities_col,
                        ner_pipe=ner_pipe,
                        ner_model_id=ner_model_id,
                    )

                if ner_result_df.empty:
                    st.warning("No valid text rows were processed.")
                else:
                    avg_score = ner_result_df["entity_match_score"].mean()
                    avg_runtime = ner_result_df["runtime_sec"].mean()

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        render_metric_card(
                            "Average Entity Match Score",
                            f"{avg_score:.1%}",
                            "Matched expected entities / total expected entities",
                        )

                    with metric_col2:
                        render_metric_card(
                            "Average NER Runtime",
                            f"{avg_runtime:.2f}s",
                            "Average runtime per NER sample",
                        )

                    st.dataframe(
                        ner_result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    ner_buffer = StringIO()
                    ner_result_df.to_csv(ner_buffer, index=False)

                    safe_ner_model_name = selected_ner_model_name.lower().replace(" ", "_").replace("/", "_")

                    st.download_button(
                        label="Download Pipeline 2 NER results CSV",
                        data=ner_buffer.getvalue().encode("utf-8"),
                        file_name=f"pipeline2_ner_results_{safe_ner_model_name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)
    
# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        Disclaimer: This application is developed for ISOM5240 educational project purposes.
        It provides information briefing only and does not provide buy, sell, hold, or other investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)