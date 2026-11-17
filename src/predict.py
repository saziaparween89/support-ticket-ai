"""
Score a new support ticket with all three models.

Usage (command line):
    python src/predict.py "I was charged twice and nobody is replying!!" --channel Email --previous-contacts 2

Usage (Python):
    from predict import analyze_ticket
    analyze_ticket("refund not received for 2 weeks", channel="Chat")
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from clean import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

ROUTING = {
    "Billing": "Payments & Billing team",
    "Technical Issue": "Tech Support (L2)",
    "Account Access": "Account Security team",
    "Delivery": "Logistics desk",
    "Refund": "Refunds & Returns team",
    "Product Inquiry": "Pre-sales / Product team",
}


@lru_cache(maxsize=1)
def load_models():
    return (joblib.load(MODELS / "category_model.joblib"),
            joblib.load(MODELS / "sentiment_model.joblib"),
            joblib.load(MODELS / "escalation_model.joblib"))


def analyze_ticket(text: str, channel: str = "Email", customer_tier: str = "Regular",
                   previous_contacts: int = 0, first_response_min: float = 30.0) -> dict:
    cat_model, sent_model, esc_model = load_models()
    row = pd.DataFrame([{
        "text_clean": clean_text(text),
        "channel": channel,
        "customer_tier": customer_tier,
        "previous_contacts": previous_contacts,
        "first_response_min": first_response_min,
        "exclamations": text.count("!"),
        "caps_ratio": sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text)),
    }])
    category = cat_model.predict(row["text_clean"])[0]
    sent_proba = dict(zip(sent_model.classes_, sent_model.predict_proba(row)[0].round(3)))
    sentiment = max(sent_proba, key=sent_proba.get)
    risk = float(esc_model.predict_proba(row)[0, 1])

    if risk >= 0.65:
        level, action = "HIGH", "Assign a senior agent now and call the customer back within 1 hour."
    elif risk >= 0.40:
        level, action = "MEDIUM", "Prioritise in queue; send a proactive status update."
    else:
        level, action = "LOW", "Handle in normal queue."

    return {
        "category": category,
        "route_to": ROUTING[category],
        "sentiment": sentiment,
        "sentiment_probabilities": {k: float(v) for k, v in sent_proba.items()},
        "escalation_risk": round(risk, 3),
        "risk_level": level,
        "recommended_action": action,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse a support ticket with AI")
    p.add_argument("text")
    p.add_argument("--channel", default="Email",
                   choices=["Email", "Chat", "Phone", "Social Media", "App"])
    p.add_argument("--tier", default="Regular", choices=["Regular", "Silver", "Gold", "Platinum"])
    p.add_argument("--previous-contacts", type=int, default=0)
    p.add_argument("--first-response-min", type=float, default=30.0)
    a = p.parse_args()
    print(json.dumps(analyze_ticket(a.text, a.channel, a.tier, a.previous_contacts,
                                    a.first_response_min), indent=2))


if __name__ == "__main__":
    main()
