"""
Support Ticket AI - interactive Streamlit app.

    streamlit run app.py

Tabs:
  1. Analyse a ticket   - paste any ticket, get category, sentiment, escalation risk
  2. Operations dashboard - KPIs and trends from the cleaned dataset
  3. Batch scoring      - upload a CSV of tickets, download them scored & prioritised
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import analyze_ticket  # noqa: E402

st.set_page_config(page_title="Support Ticket AI", page_icon="🎧", layout="wide")

EXAMPLES = {
    "Angry double charge": ("I was charged twice for my order and NOBODY is replying!! "
                            "this is the 3rd time I am complaining", "Email", "Gold", 3),
    "Refund delay": ("returned the product 2 weeks ago but refund not received yet, please check", "Chat", "Regular", 1),
    "Login problem": ("unable to login, OTP not coming since morning", "App", "Silver", 0),
    "Product question": ("is this laptop available with 16gb ram? thanks", "Chat", "Regular", 0),
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "tickets_clean.csv", parse_dates=["created_at"])


st.title("🎧 Support Ticket AI")
st.caption("Auto-route tickets, detect customer sentiment and predict escalations before they happen.")

tab1, tab2, tab3 = st.tabs(["🔍 Analyse a ticket", "📊 Operations dashboard", "📁 Batch scoring"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    ex = st.selectbox("Try an example", ["(write your own)"] + list(EXAMPLES))
    default = EXAMPLES.get(ex, ("", "Email", "Regular", 0))
    text = st.text_area("Ticket text", value=default[0], height=110,
                        placeholder="e.g. my refund is pending for 10 days, very frustrating")
    c1, c2, c3, c4 = st.columns(4)
    channels = ["Email", "Chat", "Phone", "Social Media", "App"]
    tiers = ["Regular", "Silver", "Gold", "Platinum"]
    channel = c1.selectbox("Channel", channels, index=channels.index(default[1]))
    tier = c2.selectbox("Customer tier", tiers, index=tiers.index(default[2]))
    prev = c3.number_input("Previous contacts", 0, 10, default[3])
    frt = c4.number_input("First response (min)", 1, 5000, 30)

    if st.button("Analyse", type="primary") and text.strip():
        r = analyze_ticket(text, channel, tier, int(prev), float(frt))
        k1, k2, k3 = st.columns(3)
        k1.metric("Category", r["category"], help="Predicted ticket type")
        k2.metric("Sentiment", r["sentiment"])
        k3.metric("Escalation risk", f"{r['escalation_risk']:.0%}", r["risk_level"],
                  delta_color="inverse" if r["risk_level"] != "LOW" else "off")
        st.info(f"**Route to:** {r['route_to']}  \n**Next step:** {r['recommended_action']}")
        probs = pd.DataFrame(r["sentiment_probabilities"].items(), columns=["Sentiment", "Probability"])
        st.plotly_chart(px.bar(probs, x="Probability", y="Sentiment", orientation="h",
                               range_x=[0, 1], height=220), width="stretch")

# ---------------------------------------------------------------- Tab 2
with tab2:
    df = load_data()
    cats = st.multiselect("Filter categories", sorted(df["category"].unique()),
                          default=sorted(df["category"].unique()))
    d = df[df["category"].isin(cats)]
    k = st.columns(5)
    k[0].metric("Tickets", f"{len(d):,}")
    k[1].metric("Escalation rate", f"{d['escalated'].mean():.1%}")
    k[2].metric("Avg first response", f"{d['first_response_min'].mean():.0f} min")
    k[3].metric("Avg resolution", f"{d['resolution_hours'].mean():.1f} h")
    k[4].metric("Avg CSAT", f"{d['csat_score'].mean():.2f} / 5")

    l, r = st.columns(2)
    monthly = d.groupby(d["created_at"].dt.to_period("M").astype(str)).agg(
        tickets=("ticket_id", "count"), escalation_rate=("escalated", "mean")).reset_index()
    l.plotly_chart(px.bar(monthly, x="created_at", y="tickets", title="Monthly volume"),
                   width="stretch")
    by_cat = d.groupby("category")["escalated"].mean().sort_values().reset_index()
    by_cat["escalated"] *= 100
    r.plotly_chart(px.bar(by_cat, x="escalated", y="category", orientation="h",
                          title="Escalation rate by category (%)"), width="stretch")
    l2, r2 = st.columns(2)
    by_ch = d.groupby("channel").agg(first_response=("first_response_min", "median"),
                                     csat=("csat_score", "mean")).reset_index()
    l2.plotly_chart(px.bar(by_ch, x="channel", y="first_response",
                           title="Median first response (min) by channel"), width="stretch")
    rep = d.groupby(d["previous_contacts"].clip(upper=4))["escalated"].mean().mul(100).reset_index()
    rep["previous_contacts"] = rep["previous_contacts"].map(lambda x: "4+" if x == 4 else str(x))
    r2.plotly_chart(px.bar(rep, x="previous_contacts", y="escalated",
                           title="Escalation % by previous contacts"), width="stretch")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.write("Upload a CSV with a **ticket_text** column (optional: channel, customer_tier, "
             "previous_contacts, first_response_min). You'll get it back scored and sorted by risk.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        if "ticket_text" not in raw.columns:
            st.error("The file needs a 'ticket_text' column.")
        else:
            rows = []
            for _, t in raw.iterrows():
                res = analyze_ticket(str(t["ticket_text"]), t.get("channel", "Email"),
                                     t.get("customer_tier", "Regular"),
                                     int(t.get("previous_contacts", 0)), float(t.get("first_response_min", 30)))
                rows.append({k_: v for k_, v in res.items() if k_ != "sentiment_probabilities"})
            scored = pd.concat([raw.reset_index(drop=True), pd.DataFrame(rows)], axis=1) \
                .sort_values("escalation_risk", ascending=False)
            st.dataframe(scored, width="stretch")
            st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_tickets.csv")
