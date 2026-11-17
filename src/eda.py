"""
Step 4 - Exploratory data analysis charts (saved to reports/figures).

Run:  python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import PALETTE, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "tickets_clean.csv"
FIGS = ROOT / "reports" / "figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


def main() -> None:
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN, parse_dates=["created_at"])

    # 1. Category: volume vs escalation rate
    g = df.groupby("category").agg(tickets=("ticket_id", "count"), esc=("escalated", "mean"),
                                   res=("resolution_hours", "mean")).sort_values("tickets")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    axes[0].barh(g.index, g["tickets"], color=PALETTE[0])
    axes[0].set_title("Ticket volume by category")
    axes[0].set_xlabel("Tickets")
    for y, v in enumerate(g["tickets"]):
        axes[0].text(v + 20, y, f"{v:,}", va="center", fontsize=9)
    colors = [PALETTE[3] if v > df["escalated"].mean() else PALETTE[5] for v in g["esc"]]
    axes[1].barh(g.index, g["esc"] * 100, color=colors)
    axes[1].axvline(df["escalated"].mean() * 100, ls="--", color="#374151", lw=1)
    axes[1].text(df["escalated"].mean() * 100 + 0.5, -0.6, "overall avg", fontsize=8.5, color="#374151")
    axes[1].set_title("Escalation rate by category (red = above average)")
    axes[1].set_xlabel("% of tickets escalated")
    save(fig, "eda_category.png")

    # 2. Repeat contacts -> escalation (the single strongest operational insight)
    b = df.assign(bucket=pd.cut(df["previous_contacts"], [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"]))
    r = b.groupby("bucket", observed=True)["escalated"].mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bars = ax.bar(r.index.astype(str), r.values, color=[PALETTE[5], PALETTE[2], PALETTE[2], PALETTE[3]])
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in r.values], padding=3)
    ax.set_title("Customers who contact us repeatedly escalate more")
    ax.set_xlabel("Previous contacts about the same issue")
    ax.set_ylabel("% escalated")
    save(fig, "eda_repeat_contacts.png")

    # 3. Monthly volume + CSAT trend
    m = df.groupby(df["created_at"].dt.to_period("M")).agg(tickets=("ticket_id", "count"),
                                                           csat=("csat_score", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(m.index.astype(str), m["tickets"], color=PALETTE[0], alpha=0.85, label="Tickets")
    ax.set_ylabel("Tickets")
    ax2 = ax.twinx()
    ax2.plot(m.index.astype(str), m["csat"], color=PALETTE[2], marker="o", lw=2.2, label="Avg CSAT")
    ax2.set_ylim(1, 5)
    ax2.set_ylabel("Avg CSAT (1-5)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Monthly ticket volume and customer satisfaction")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    save(fig, "eda_monthly_trend.png")

    # 4. When do tickets arrive? Hour x weekday heatmap (staffing insight)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pv = df.pivot_table(index="weekday", columns="created_hour", values="ticket_id",
                        aggfunc="count").reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(pv.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), [d[:3] for d in order])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_title("Ticket arrivals by weekday and hour - plan agent rosters around the blue peaks")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, label="Tickets")
    save(fig, "eda_heatmap.png")

    # 5. Channel: first response time vs CSAT
    c = df.groupby("channel").agg(frt=("first_response_min", "median"), csat=("csat_score", "mean"),
                                  esc=("escalated", "mean"), n=("ticket_id", "count"))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(c["frt"], c["esc"] * 100, s=c["n"] / 6, color=PALETTE[1], alpha=0.75, edgecolor="white")
    for name, row in c.iterrows():
        ax.annotate(name, (row["frt"], row["esc"] * 100), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Median first response time (minutes, log scale)")
    ax.set_ylabel("% escalated")
    ax.set_title("Slow channels escalate more (bubble = volume)")
    save(fig, "eda_channels.png")

    # 6. Resolution-time distribution by category
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cats = df.groupby("category")["resolution_hours"].median().sort_values().index
    data = [df.loc[df["category"] == c_, "resolution_hours"].dropna().clip(upper=200) for c_ in cats]
    ax.boxplot(data, orientation="horizontal", tick_labels=list(cats), showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", color=PALETTE[0]), medianprops=dict(color=PALETTE[3], lw=2))
    ax.set_xlabel("Resolution time (hours)")
    ax.set_title("Refunds and deliveries take longest to resolve")
    save(fig, "eda_resolution_time.png")

    print(f"Saved EDA charts to {FIGS}")


if __name__ == "__main__":
    main()
