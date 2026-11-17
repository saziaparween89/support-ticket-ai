"""
Generate a realistic, messy customer-support ticket dataset.

Why synthetic?  Real contact-centre tickets contain customer PII and can't be
published.  This script simulates ~9,000 tickets for a fictional Indian
e-commerce + fintech brand ("ShopKart") with the patterns seen in real support
operations: overlapping categories, typos, Hinglish phrasing, angry repeat
contacts, missing values, duplicates and inconsistent formatting.

The data is deliberately *dirty* so the cleaning step in the pipeline is real
work, and the labels are deliberately *noisy* so the model can't score 100%.

Run:  python data/generate_tickets.py
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TICKETS = 9000
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUT = Path(__file__).resolve().parent / "raw" / "tickets_raw.csv"

# ---------------------------------------------------------------------------
# Phrase banks per category.  Some phrases are shared on purpose
# (e.g. "money" appears in Billing AND Refund) so the classifier has to learn
# context rather than single keywords.
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Billing": {
        "weight": 0.20,
        "openers": [
            "I was charged twice for order {order}",
            "extra amount deducted from my card for {order}",
            "why is there a convenience fee on my bill",
            "my invoice shows wrong GST amount",
            "amount debited but payment shows failed",
            "EMI was charged even though I cancelled",
            "I see an unknown charge of Rs {amt} on my statement",
            "wallet balance deducted but order not placed",
            "double payment happened through UPI",
            "the price on invoice is higher than what was shown",
        ],
    },
    "Technical Issue": {
        "weight": 0.22,
        "openers": [
            "app keeps crashing when I open my orders page",
            "unable to login, OTP not coming",
            "website shows error 500 at checkout",
            "the app is very slow and payments page does not load",
            "push notifications are not working after update",
            "I can't upload my KYC documents, upload fails",
            "cart is empty every time I reopen the app",
            "search is not showing any results",
            "screen goes blank after clicking pay now",
            "app stuck on loading screen since morning",
        ],
    },
    "Account Access": {
        "weight": 0.14,
        "openers": [
            "my account got blocked without any reason",
            "I forgot my password and reset link is not working",
            "someone else logged in to my account",
            "I want to change the mobile number linked to my account",
            "account locked after too many attempts",
            "unable to verify my email id",
            "need to delete my account permanently",
            "my profile details got changed automatically",
        ],
    },
    "Delivery": {
        "weight": 0.20,
        "openers": [
            "my order {order} is not delivered yet",
            "tracking shows delivered but I did not receive the parcel",
            "delivery is delayed by {days} days",
            "courier boy was rude and did not deliver",
            "received a damaged package for {order}",
            "wrong item delivered, I ordered something else",
            "order stuck in transit for a week",
            "delivery address needs to be changed for {order}",
            "parcel was left outside, now it is missing",
        ],
    },
    "Refund": {
        "weight": 0.16,
        "openers": [
            "I returned the product but refund not received",
            "refund for {order} is pending since {days} days",
            "money not credited back to my account after cancellation",
            "I want my money back for the damaged item",
            "refund amount is less than what I paid",
            "return pickup was done but refund status still processing",
            "cancelled order {order}, when will I get refund",
        ],
    },
    "Product Inquiry": {
        "weight": 0.08,
        "openers": [
            "is this phone available in blue colour",
            "does the laptop come with warranty in India",
            "what is the size chart for this kurta",
            "can I get bulk discount for 50 units",
            "when will the item be back in stock",
            "is cash on delivery available for my pincode {pin}",
            "is this product original or refurbished",
        ],
    },
}

# Sentiment modifiers - appended to openers
NEUTRAL_TAILS = [
    "", "please help.", "kindly check.", "please look into this.",
    "can you please update?", "thanks.", "let me know the status.",
]
NEGATIVE_TAILS = [
    "this is very frustrating.", "worst service ever!!", "I am really disappointed.",
    "this is unacceptable.", "I have been waiting too long.", "very bad experience.",
    "you people are not responding at all.", "fed up with this.",
]
ANGRY_TAILS = [
    "I will go to consumer court if not fixed today!!!",
    "I will post this on twitter and leave 1 star rating.",
    "this is the 3rd time I am complaining, NOBODY helps",
    "cancel everything and refund NOW, pathetic service",
    "I want to talk to your manager immediately.",
    "totally cheated, never shopping here again!!",
]
POSITIVE_TAILS = [
    "thanks for the quick help last time.", "love your app otherwise :)",
    "appreciate your support.", "great service usually, just this one issue.",
]
HINGLISH = [
    " bhai jaldi karo", " kya ho raha hai", " please jaldi solve karo", " yaar",
    " abhi tak kuch nahi hua", "",
]

CHANNELS = ["Email", "Chat", "Phone", "Social Media", "App"]
CHANNEL_P = [0.30, 0.32, 0.20, 0.06, 0.12]
CITIES = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune",
          "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
TIERS = ["Regular", "Silver", "Gold", "Platinum"]
TIER_P = [0.55, 0.25, 0.14, 0.06]
AGENTS = [f"AGT{n:03d}" for n in range(101, 141)]
HOUR_WEIGHTS = [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.5, 4, 5.5, 6.5, 6.8,
                6, 5.5, 5.8, 5.6, 5.2, 5, 5.6, 6.2, 6.5, 5, 3.2, 1.8]


def typo(text: str, p: float = 0.04) -> str:
    """Introduce realistic typos: dropped / swapped letters."""
    out = []
    chars = list(text)
    i = 0
    while i < len(chars):
        c = chars[i]
        r = random.random()
        if c.isalpha() and r < p / 2:
            i += 1  # drop character
            continue
        if c.isalpha() and r < p and i + 1 < len(chars) and chars[i + 1].isalpha():
            out.extend([chars[i + 1], c])  # swap
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def fill(template: str) -> str:
    return template.format(
        order=f"#OD{rng.integers(10**8, 10**9)}",
        amt=int(rng.choice([199, 349, 499, 999, 1499, 2499, 4999])),
        days=int(rng.integers(3, 21)),
        pin=int(rng.integers(110001, 855117)),
    )


def make_ticket(i: int, start: datetime) -> dict:
    cats = list(CATEGORIES)
    weights = [CATEGORIES[c]["weight"] for c in cats]
    category = random.choices(cats, weights)[0]

    # Sentiment drawn first, then text built to match (with some leakage noise)
    sentiment = random.choices(
        ["Positive", "Neutral", "Negative", "Angry"],
        [0.08, 0.47, 0.32, 0.13],
    )[0]
    # Billing/Refund/Delivery skew more negative in real life
    if category in {"Refund", "Delivery", "Billing"} and sentiment == "Neutral" and random.random() < 0.3:
        sentiment = "Negative"

    text = fill(random.choice(CATEGORIES[category]["openers"]))

    # ~12% of tickets mention a second topic -> realistic overlap
    if random.random() < 0.12:
        other = random.choice([c for c in cats if c != category])
        text += ". also " + fill(random.choice(CATEGORIES[other]["openers"]))

    # Real customers don't always "say" how they feel: some angry customers write
    # politely, some neutral ones sound annoyed. Mix adjacent tones on purpose.
    tails = {"Positive": POSITIVE_TAILS, "Neutral": NEUTRAL_TAILS,
             "Negative": NEGATIVE_TAILS, "Angry": ANGRY_TAILS}
    neighbours = {"Positive": ["Neutral"], "Neutral": ["Positive", "Negative"],
                  "Negative": ["Neutral", "Angry"], "Angry": ["Negative"]}
    r = random.random()
    if r < 0.22:
        tail = random.choice(tails[random.choice(neighbours[sentiment])])
    elif r < 0.32:
        tail = ""
    else:
        tail = random.choice(tails[sentiment])
    text = f"{text}. {tail}".strip()
    if random.random() < 0.15:
        text += random.choice(HINGLISH)
    if sentiment == "Angry" and random.random() < 0.4:
        text = text.upper()
    text = typo(text)

    channel = random.choices(CHANNELS, CHANNEL_P)[0]
    tier = random.choices(TIERS, TIER_P)[0]
    # Ticket arrivals peak late morning and evening, quiet overnight (Jan-Jun)
    hour = random.choices(range(24), HOUR_WEIGHTS)[0]
    created = start + timedelta(days=int(rng.integers(0, 181)), hours=hour,
                                minutes=int(rng.integers(0, 60)))

    # How many times this customer has contacted before about the same issue
    prev_contacts = int(rng.poisson(0.6 if sentiment in {"Positive", "Neutral"} else 1.6))

    # First response time (minutes) depends on channel
    base_frt = {"Chat": 4, "Phone": 2, "App": 30, "Email": 240, "Social Media": 90}[channel]
    first_response_min = max(1, int(rng.gamma(2.0, base_frt / 2)))

    # ---- Escalation: the target for the risk model -------------------------
    logit = -3.2
    logit += {"Positive": -1.2, "Neutral": -0.3, "Negative": 0.9, "Angry": 2.1}[sentiment]
    logit += 0.55 * min(prev_contacts, 5)
    logit += {"Billing": 0.4, "Refund": 0.6, "Delivery": 0.3, "Technical Issue": 0.0,
              "Account Access": 0.5, "Product Inquiry": -1.3}[category]
    logit += 0.35 * np.log1p(first_response_min / 60)
    logit += {"Regular": 0.0, "Silver": 0.1, "Gold": 0.35, "Platinum": 0.6}[tier]
    logit += 0.4 if channel == "Social Media" else 0.0
    logit += rng.normal(0, 0.6)  # unexplained noise
    escalated = int(rng.random() < 1 / (1 + np.exp(-logit)))

    # Resolution time (hours)
    base_res = {"Product Inquiry": 2, "Technical Issue": 10, "Account Access": 8,
                "Billing": 18, "Delivery": 30, "Refund": 48}[category]
    resolution_hrs = rng.gamma(2.0, base_res / 2) * (1.8 if escalated else 1.0)
    status = random.choices(["Resolved", "Closed", "Open", "Pending"], [0.62, 0.22, 0.08, 0.08])[0]
    resolved_at = created + timedelta(hours=float(resolution_hrs)) if status in {"Resolved", "Closed"} else None

    # CSAT 1-5, only for resolved tickets, many customers skip the survey
    csat = None
    if resolved_at is not None and random.random() < 0.55:
        score = 4.3 - 0.9 * escalated - 0.025 * resolution_hrs / 2 \
            - {"Positive": -0.3, "Neutral": 0, "Negative": 0.6, "Angry": 1.1}[sentiment]
        csat = int(np.clip(round(score + rng.normal(0, 0.7)), 1, 5))

    # Labelling noise: QA reviewers disagree on ~5% of sentiment labels
    if random.random() < 0.05:
        sentiment = random.choice(neighbours[sentiment])

    # Labelling noise: agents mis-tag ~6% of tickets
    tagged_category = category
    if random.random() < 0.06:
        tagged_category = random.choice(cats)

    return {
        "ticket_id": f"TCK-{100000 + i}",
        "created_at": created,
        "resolved_at": resolved_at,
        "channel": channel,
        "customer_tier": tier,
        "city": random.choice(CITIES),
        "agent_id": random.choice(AGENTS),
        "ticket_text": text,
        "category": tagged_category,
        "sentiment": sentiment,
        "previous_contacts": prev_contacts,
        "first_response_min": first_response_min,
        "status": status,
        "escalated": escalated,
        "csat_score": csat,
    }


def make_dirty(df: pd.DataFrame) -> pd.DataFrame:
    """Inject the kinds of data-quality problems seen in real CRM exports."""
    df = df.copy()
    n = len(df)

    # 1. Inconsistent category / channel spellings
    idx = rng.choice(n, int(n * 0.07), replace=False)
    df.loc[idx, "category"] = df.loc[idx, "category"].map(
        lambda c: random.choice([c.lower(), c.upper(), f" {c} ", c.replace(" ", "_")]))
    idx = rng.choice(n, int(n * 0.05), replace=False)
    df.loc[idx, "channel"] = df.loc[idx, "channel"].map(
        lambda c: random.choice([c.lower(), c.upper(), {"Social Media": "social", "Chat": "Live Chat"}.get(c, c)]))

    # 2. Mixed date formats (stored as text, like a raw Excel/CRM export)
    df["created_at"] = df["created_at"].map(
        lambda d: d.strftime(random.choice(["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%b-%Y %I:%M %p"])))
    df["resolved_at"] = df["resolved_at"].map(lambda d: "" if pd.isna(d) else d.strftime("%Y-%m-%d %H:%M:%S"))

    # 3. Missing values
    for col, frac in [("city", 0.04), ("customer_tier", 0.02), ("first_response_min", 0.03)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 4. Outliers / bad values in first_response_min (system glitches)
    idx = rng.choice(n, 25, replace=False)
    df.loc[idx, "first_response_min"] = rng.choice([-5, 0, 99999], 25)

    # 5. Empty / junk ticket text
    idx = rng.choice(n, 30, replace=False)
    df.loc[idx, "ticket_text"] = rng.choice(["", " ", "na", "test", "..."], 30)

    # 6. Exact duplicate rows (double export)
    dups = df.sample(120, random_state=SEED)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=SEED)
    return df.reset_index(drop=True)


def main() -> None:
    start = datetime(2026, 1, 1)
    rows = [make_ticket(i, start) for i in range(N_TICKETS)]
    df = make_dirty(pd.DataFrame(rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
