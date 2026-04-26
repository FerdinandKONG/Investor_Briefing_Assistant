import re
from dataclasses import dataclass
from typing import Iterable


SENTIMENT_LABELS = {
    "positive": "Positive",
    "pos": "Positive",
    "label_2": "Positive",
    "neutral": "Neutral",
    "neu": "Neutral",
    "label_1": "Neutral",
    "negative": "Negative",
    "neg": "Negative",
    "label_0": "Negative",
}


RISK_RULES = {
    "Earnings risk": [
        "earnings miss",
        "missed expectations",
        "lower-than-expected",
        "profit warning",
        "margin pressure",
        "revenue decline",
        "weak guidance",
        "loss widened",
    ],
    "Competition risk": [
        "competition",
        "rival",
        "market share",
        "price war",
        "competitive pressure",
    ],
    "Regulatory risk": [
        "regulator",
        "regulatory",
        "investigation",
        "lawsuit",
        "fine",
        "antitrust",
        "compliance",
    ],
    "Macroeconomic risk": [
        "inflation",
        "interest rate",
        "rates",
        "recession",
        "slowdown",
        "tariff",
        "currency pressure",
    ],
    "Liquidity or financing risk": [
        "debt",
        "liquidity",
        "cash flow",
        "refinancing",
        "downgrade",
        "credit rating",
    ],
    "Operational risk": [
        "supply chain",
        "production delay",
        "delivery delay",
        "recall",
        "shutdown",
        "labor dispute",
    ],
    "Market volatility": [
        "shares fell",
        "stock fell",
        "selloff",
        "plunged",
        "tumbled",
        "volatility",
        "sharp decline",
    ],
}


@dataclass(frozen=True)
class RiskSignal:
    name: str
    evidence: str


def normalize_sentiment_label(label: str) -> str:
    key = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    return SENTIMENT_LABELS.get(key, str(label or "Unknown").strip().title())


def coerce_prediction(raw_prediction):
    """Return a single top prediction from common Transformers pipeline shapes."""
    pred = raw_prediction
    if isinstance(pred, list) and pred and isinstance(pred[0], list):
        pred = pred[0]
    if isinstance(pred, list):
        if not pred:
            return {"label": "Unknown", "score": 0.0}
        pred = max(pred, key=lambda item: float(item.get("score", 0.0)))
    if not isinstance(pred, dict):
        return {"label": "Unknown", "score": 0.0}
    return {
        "label": normalize_sentiment_label(pred.get("label")),
        "score": float(pred.get("score", 0.0)),
    }


def detect_risk_signals(text: str) -> list[RiskSignal]:
    lowered = str(text or "").lower()
    signals: list[RiskSignal] = []
    for name, keywords in RISK_RULES.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, lowered):
                signals.append(RiskSignal(name=name, evidence=keyword))
                break
    return signals


def format_entities(raw_entities: Iterable[dict]) -> list[dict]:
    seen = set()
    formatted = []
    for ent in raw_entities or []:
        word = str(ent.get("word") or "").replace(" ##", "").replace("##", "").strip()
        group = str(ent.get("entity_group") or ent.get("entity") or "ENTITY").upper()
        if not word:
            continue
        key = (word.lower(), group)
        if key in seen:
            continue
        seen.add(key)
        formatted.append(
            {
                "Entity": word,
                "Type": group,
                "Confidence": round(float(ent.get("score", 0.0)), 3),
            }
        )
    return formatted


def build_beginner_explanation(
    sentiment: str,
    confidence: float,
    risk_signals: list[RiskSignal],
    entities: list[dict],
) -> str:
    confidence_pct = round(confidence * 100, 1)
    if sentiment == "Positive":
        base = (
            f"The model reads this news as positive with {confidence_pct}% confidence. "
            "For a beginner investor, this usually means the wording points to favorable business news, "
            "such as stronger demand, better results, or improving expectations."
        )
    elif sentiment == "Negative":
        base = (
            f"The model reads this news as negative with {confidence_pct}% confidence. "
            "For a beginner investor, this usually means the wording points to pressure, uncertainty, "
            "or weaker business expectations."
        )
    elif sentiment == "Neutral":
        base = (
            f"The model reads this news as neutral with {confidence_pct}% confidence. "
            "For a beginner investor, this usually means the sentence is more informational than clearly good or bad."
        )
    else:
        base = "The model could not confidently classify the sentiment of this text."

    if risk_signals:
        risks = ", ".join(signal.name.lower() for signal in risk_signals[:3])
        base += f" The main risk themes detected are {risks}."
    if entities:
        names = ", ".join(ent["Entity"] for ent in entities[:4])
        base += f" Key names to check next include {names}."
    return base


def evaluate_correctness(expected: str, predicted: str) -> str:
    if not expected:
        return ""
    expected_norm = normalize_sentiment_label(expected)
    predicted_norm = normalize_sentiment_label(predicted)
    return "Correct" if expected_norm == predicted_norm else "Incorrect"
