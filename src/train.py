"""
Step 3 - Train the three AI models.

  1. Category classifier   -> routes a ticket to the right team automatically
  2. Sentiment classifier  -> flags angry / negative customers
  3. Escalation-risk model -> predicts which tickets will escalate, so team
                              leads can intervene BEFORE it happens

Validation uses a time-based split (train: Jan-May, test: June), which mirrors
how the model would really be used: trained on the past, scored on new tickets.

Run:  python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import apply_style, PALETTE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "tickets_clean.csv"
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics.json"

TEXT = "text_clean"
META_CAT = ["channel", "customer_tier"]
META_NUM = ["previous_contacts", "first_response_min", "exclamations", "caps_ratio"]


def text_features() -> FeatureUnion:
    """Word n-grams capture meaning; character n-grams make the model robust
    to typos like 'refnd' or 'delivred' that are everywhere in real tickets."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


def time_split(df: pd.DataFrame):
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    train = df[df["created_at"] < "2026-06-01"]
    test = df[df["created_at"] >= "2026-06-01"]
    return train, test


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "#1f2937", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Category classifier - compare 3 algorithms, keep the best
# ---------------------------------------------------------------------------
def train_category(train, test) -> tuple[Pipeline, dict]:
    candidates = {
        "Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=5),
        "Linear SVM": LinearSVC(C=0.5),
    }
    scores, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", text_features()), ("clf", clf)])
        pipe.fit(train[TEXT], train["category"])
        pred = pipe.predict(test[TEXT])
        scores[name] = {
            "accuracy": round(accuracy_score(test["category"], pred), 4),
            "macro_f1": round(f1_score(test["category"], pred, average="macro"), 4),
        }
        fitted[name] = (pipe, pred)
        print(f"  [category] {name:<20} acc={scores[name]['accuracy']:.3f}  macroF1={scores[name]['macro_f1']:.3f}")

    best = max(scores, key=lambda k: scores[k]["macro_f1"])
    pipe, pred = fitted[best]
    labels = sorted(train["category"].unique())
    plot_confusion(test["category"], pred, labels, f"Ticket category - {best}",
                   FIGS / "category_confusion_matrix.png")
    report = classification_report(test["category"], pred, output_dict=True)
    return pipe, {"model_comparison": scores, "best_model": best,
                  "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items() if k in labels}}


# ---------------------------------------------------------------------------
# 2. Sentiment classifier
# ---------------------------------------------------------------------------
def train_sentiment(train, test) -> tuple[Pipeline, dict]:
    pipe = Pipeline([
        ("features", ColumnTransformer([
            ("text", text_features(), TEXT),
            ("style", StandardScaler(), ["exclamations", "caps_ratio"]),
        ])),
        ("clf", LogisticRegression(max_iter=3000, C=3, class_weight="balanced")),
    ])
    pipe.fit(train, train["sentiment"])
    pred = pipe.predict(test)
    labels = ["Positive", "Neutral", "Negative", "Angry"]
    plot_confusion(test["sentiment"], pred, labels, "Customer sentiment", FIGS / "sentiment_confusion_matrix.png")
    m = {"accuracy": round(accuracy_score(test["sentiment"], pred), 4),
         "macro_f1": round(f1_score(test["sentiment"], pred, average="macro"), 4)}
    # Business metric: how many truly ANGRY customers do we catch?
    angry = test["sentiment"] == "Angry"
    m["angry_recall"] = round(float((pred[angry.values] == "Angry").mean()), 4)
    print(f"  [sentiment] acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} angry_recall={m['angry_recall']:.3f}")
    return pipe, m


# ---------------------------------------------------------------------------
# 3. Escalation-risk model: metadata-only baseline vs text + metadata
# ---------------------------------------------------------------------------
def escalation_pipeline(use_text: bool) -> Pipeline:
    parts = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), META_CAT),
        ("num", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), META_NUM),
    ]
    if use_text:
        # Simpler text features than the classifiers: escalation is a weaker
        # text signal, so fewer, more common words avoid over-fitting on typos.
        parts.append(("text", FeatureUnion([("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=15, stop_words="english", sublinear_tf=True))]), TEXT))
    return Pipeline([
        ("features", ColumnTransformer(parts)),
        ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
    ])


def capture_at_top(y_true, scores, frac=0.2) -> float:
    """% of all escalations found if the team reviews only the top `frac` riskiest tickets."""
    k = int(len(scores) * frac)
    top = np.argsort(-scores)[:k]
    return float(np.asarray(y_true)[top].sum() / np.asarray(y_true).sum())


def train_escalation(train, test) -> tuple[Pipeline, dict]:
    results, models = {}, {}
    for name, use_text in [("Baseline (metadata only)", False), ("AI model (text + metadata)", True)]:
        pipe = escalation_pipeline(use_text)
        pipe.fit(train, train["escalated"])
        proba = pipe.predict_proba(test)[:, 1]
        results[name] = {
            "roc_auc": round(roc_auc_score(test["escalated"], proba), 4),
            "pr_auc": round(average_precision_score(test["escalated"], proba), 4),
            "capture_top20pct": round(capture_at_top(test["escalated"], proba, 0.2), 4),
        }
        models[name] = (pipe, proba)
        print(f"  [escalation] {name:<28} AUC={results[name]['roc_auc']:.3f} "
              f"PR-AUC={results[name]['pr_auc']:.3f} top20%={results[name]['capture_top20pct']:.1%}")

    # ROC chart comparing both
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for (name, (_, proba)), color in zip(models.items(), [PALETTE[3], PALETTE[0]]):
        fpr, tpr, _ = roc_curve(test["escalated"], proba)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name}  (AUC {results[name]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Escalation prediction - ROC curve")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "escalation_roc.png", dpi=150)
    plt.close(fig)

    # Lift / gains chart: the chart that a business stakeholder understands
    pipe, proba = models["AI model (text + metadata)"]
    y = test["escalated"].to_numpy()
    order = np.argsort(-proba)
    fracs = np.linspace(0, 1, 101)
    captured = [y[order[: int(len(y) * f)]].sum() / y.sum() for f in fracs]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fracs * 100, np.array(captured) * 100, color=PALETTE[0], lw=2.4, label="AI risk ranking")
    ax.plot([0, 100], [0, 100], ls="--", color="#9ca3af", lw=1, label="Random order")
    c20 = results["AI model (text + metadata)"]["capture_top20pct"] * 100
    ax.scatter([20], [c20], color=PALETTE[0], zorder=5)
    ax.annotate(f"Review top 20% of tickets\n-> catch {c20:.0f}% of escalations",
                xy=(20, c20), xytext=(32, c20 - 22), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#4b5563"))
    ax.set_xlabel("% of tickets reviewed (highest risk first)")
    ax.set_ylabel("% of escalations caught")
    ax.set_title("Cumulative gains - escalation risk")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "escalation_gains.png", dpi=150)
    plt.close(fig)

    # Top drivers from the metadata part of the model (interpretable coefficients)
    feats = pipe.named_steps["features"]
    names = feats.get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    drivers = pd.DataFrame({"feature": names, "coef": coefs})
    drivers = drivers[~drivers["feature"].str.startswith("text__")].copy()
    drivers["feature"] = drivers["feature"].str.replace(r"^(cat|num)__", "", regex=True)
    drivers = drivers.reindex(drivers["coef"].abs().sort_values(ascending=False).index).head(10)

    # Top words pushing towards escalation
    text_mask = pd.Series(names).str.startswith("text__word__")
    words = pd.DataFrame({"word": pd.Series(names)[text_mask].str.replace("text__word__", ""),
                          "coef": coefs[text_mask.values]})
    top_words = words.sort_values("coef", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    d = drivers.iloc[::-1]
    axes[0].barh(d["feature"], d["coef"], color=[PALETTE[3] if c > 0 else PALETTE[1] for c in d["coef"]])
    axes[0].axvline(0, color="#6b7280", lw=0.8)
    axes[0].set_title("Strongest non-text drivers")
    axes[0].set_xlabel("Model coefficient (+ raises risk)")
    w = top_words.iloc[::-1]
    axes[1].barh(w["word"], w["coef"], color=PALETTE[3])
    axes[1].set_title("Words that signal an escalation")
    axes[1].set_xlabel("Model coefficient")
    fig.tight_layout()
    fig.savefig(FIGS / "escalation_drivers.png", dpi=150)
    plt.close(fig)

    out = {"comparison": results,
           "top_drivers": drivers.round(3).to_dict(orient="records"),
           "top_escalation_words": top_words["word"].tolist()}
    return pipe, out


def main() -> dict:
    apply_style()
    MODELS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN)
    train, test = time_split(df)
    print(f"Train: {len(train):,} tickets (Jan-May)   Test: {len(test):,} tickets (June)")

    cat_model, cat_m = train_category(train, test)
    sent_model, sent_m = train_sentiment(train, test)
    esc_model, esc_m = train_escalation(train, test)

    # Refit the final models on ALL data before saving (more data = better model)
    cat_model.fit(df[TEXT], df["category"])
    sent_model.fit(df, df["sentiment"])
    esc_model.fit(df, df["escalated"])
    joblib.dump(cat_model, MODELS / "category_model.joblib")
    joblib.dump(sent_model, MODELS / "sentiment_model.joblib")
    joblib.dump(esc_model, MODELS / "escalation_model.joblib")

    metrics = {"train_rows": len(train), "test_rows": len(test),
               "category": cat_m, "sentiment": sent_m, "escalation": esc_m}
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models to {MODELS} and metrics to {METRICS}")
    return metrics


if __name__ == "__main__":
    main()
