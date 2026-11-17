"""
Step 1 - Data cleaning & validation.

Takes the raw CRM export (data/raw/tickets_raw.csv) and produces an
analysis-ready file (data/processed/tickets_clean.csv) plus a data-quality
report (reports/data_quality_report.md) that records every fix applied.

Run:  python src/clean.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tickets_raw.csv"
OUT = ROOT / "data" / "processed" / "tickets_clean.csv"
REPORT = ROOT / "reports" / "data_quality_report.md"

VALID_CATEGORIES = ["Billing", "Technical Issue", "Account Access", "Delivery", "Refund", "Product Inquiry"]
CHANNEL_MAP = {
    "email": "Email", "chat": "Chat", "live chat": "Chat", "phone": "Phone",
    "social media": "Social Media", "social": "Social Media", "app": "App",
}
JUNK_TEXT = {"", "na", "n/a", "test", "...", "none", "null"}


def standardise_category(value: str) -> str | float:
    if pd.isna(value):
        return np.nan
    v = str(value).strip().replace("_", " ").lower()
    for cat in VALID_CATEGORIES:
        if v == cat.lower():
            return cat
    return np.nan


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """The export mixes 3 formats. Try each explicitly - never let pandas guess
    day/month order, because 03/04 can mean 3 April or 4 March."""
    formats = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%b-%Y %I:%M %p"]
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        out = out.fillna(parsed)
    return out


def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"#od\d+", " ordernum ", t)        # mask order ids (PII-like, no signal)
    t = re.sub(r"\b\d{6}\b", " pincode ", t)       # mask pincodes
    t = re.sub(r"rs\.?\s?\d+", " amount ", t)       # mask amounts
    t = re.sub(r"[^a-z\s!?]", " ", t)               # keep ! and ? (anger signal)
    return re.sub(r"\s+", " ", t).strip()


def main() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    log: list[tuple[str, int]] = [("Raw rows", len(df))]

    # 1. Exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.append(("Exact duplicate rows removed", before - len(df)))

    # 2. Duplicate ticket IDs (keep first)
    before = len(df)
    df = df.drop_duplicates(subset="ticket_id", keep="first")
    log.append(("Duplicate ticket_id rows removed", before - len(df)))

    # 3. Junk / empty ticket text
    junk = df["ticket_text"].fillna("").str.strip().str.lower().isin(JUNK_TEXT)
    log.append(("Rows with empty/junk ticket text removed", int(junk.sum())))
    df = df[~junk].copy()

    # 4. Standardise categorical labels
    n_bad_cat = (~df["category"].isin(VALID_CATEGORIES)).sum()
    df["category"] = df["category"].map(standardise_category)
    log.append(("Category labels standardised (case/spaces/underscores)", int(n_bad_cat)))

    n_bad_ch = (~df["channel"].isin(set(CHANNEL_MAP.values()))).sum()
    df["channel"] = df["channel"].str.strip().str.lower().map(CHANNEL_MAP)
    log.append(("Channel labels standardised", int(n_bad_ch)))

    # 5. Dates
    df["created_at"] = parse_mixed_dates(df["created_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce")
    log.append(("Unparseable created_at dates", int(df["created_at"].isna().sum())))

    # 6. Invalid first-response times (<=0 or > 7 days) -> NaN, then impute
    #    with the channel median (response time differs a lot by channel)
    bad = (df["first_response_min"] <= 0) | (df["first_response_min"] > 7 * 24 * 60)
    log.append(("Invalid first_response_min values (<=0 or >7 days) set to null", int(bad.sum())))
    df.loc[bad, "first_response_min"] = np.nan
    n_missing = int(df["first_response_min"].isna().sum())
    df["first_response_min"] = df.groupby("channel")["first_response_min"].transform(
        lambda s: s.fillna(s.median()))
    log.append(("first_response_min imputed with channel median", n_missing))

    # 7. Missing categoricals -> explicit 'Unknown' (don't silently drop rows)
    for col in ["city", "customer_tier"]:
        n = int(df[col].isna().sum())
        df[col] = df[col].fillna("Unknown")
        log.append((f"Missing {col} filled with 'Unknown'", n))

    # 8. Derived features
    df["resolution_hours"] = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["created_date"] = df["created_at"].dt.date
    df["created_month"] = df["created_at"].dt.to_period("M").astype(str)
    df["created_hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.day_name()
    df["text_clean"] = df["ticket_text"].map(clean_text)
    df["exclamations"] = df["ticket_text"].str.count("!")
    df["caps_ratio"] = df["ticket_text"].map(
        lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))).round(3)

    # Validation checks - fail loudly if something is off
    assert df["ticket_id"].is_unique
    assert df["category"].notna().all(), "Unmapped category found"
    assert df["channel"].notna().all(), "Unmapped channel found"
    assert (df["first_response_min"] > 0).all()
    assert df["escalated"].isin([0, 1]).all()

    log.append(("Clean rows", len(df)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Report\n", "| Step | Rows affected |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in log]
    REPORT.write_text("\n".join(lines) + "\n")

    for k, v in log:
        print(f"{k:<60} {v:>6,}")
    return df


if __name__ == "__main__":
    main()
