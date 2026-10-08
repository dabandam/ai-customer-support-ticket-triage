"""
train.py
--------
End-to-end training pipeline for the AI Customer Support Ticket Triage system.

Implements steps 1-5 of the project brief:
    1. Load historical ticket data (category, urgency, text).
    2. Clean text.
    3. TF-IDF feature extraction.
    4. Train a multiclass category classifier AND a separate urgency classifier.
    5. Evaluate with F1-score, confusion matrix, precision/recall by class.

Artifacts produced:
    models/category_model.joblib   - TF-IDF + LinearSVC pipeline for category
    models/urgency_model.joblib    - TF-IDF + LogisticRegression pipeline for urgency
    figures/confusion_category.png
    figures/confusion_urgency.png
    reports/metrics.json           - all numeric results, reused by the report generator
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import clean_series

DATA_PATH = "/home/claude/ticket_triage/data/support_tickets.csv"
MODELS_DIR = "/home/claude/ticket_triage/models"
FIGURES_DIR = "/home/claude/ticket_triage/figures"
REPORTS_DIR = "/home/claude/ticket_triage/reports"

CATEGORY_LABELS = ["billing", "technical", "account", "product"]
URGENCY_LABELS = ["low", "medium", "high", "critical"]

LOW_CONFIDENCE_THRESHOLD = 0.90  # step 6: route uncertain predictions to human review


def plot_confusion(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_task(X_train, X_test, y_train, y_test, labels, base_estimator, task_name):
    """Train + evaluate a TF-IDF -> classifier pipeline for one task (category or urgency)."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", base_estimator),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plot_confusion(cm, labels, f"Confusion Matrix — {task_name}",
                    f"{FIGURES_DIR}/confusion_{task_name.lower()}.png")

    print(f"\n=== {task_name} classifier ===")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f} | Weighted-F1: {weighted_f1:.3f}")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    return pipeline, {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = clean_series(df["ticket_text"])

    X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
        df["clean_text"], df["category"], df["urgency"],
        test_size=0.2, random_state=42, stratify=df["category"],
    )

    # --- Category model: LinearSVC wrapped in CalibratedClassifierCV so we get
    # predict_proba for the confidence-threshold / human-review step. ---
    category_estimator = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=42), cv=3
    )
    category_pipeline, category_metrics = train_task(
        X_train, X_test, y_cat_train, y_cat_test,
        CATEGORY_LABELS, category_estimator, "Category"
    )

    # --- Urgency model: multinomial Logistic Regression (ordinal-ish target,
    # native predict_proba, handles the smaller "critical" class with class_weight). ---
    urgency_estimator = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )
    urgency_pipeline, urgency_metrics = train_task(
        X_train, X_test, y_urg_train, y_urg_test,
        URGENCY_LABELS, urgency_estimator, "Urgency"
    )

    joblib.dump(category_pipeline, f"{MODELS_DIR}/category_model.joblib")
    joblib.dump(urgency_pipeline, f"{MODELS_DIR}/urgency_model.joblib")

    # --- Low-confidence audit on the held-out test set (step 6) ---
    cat_proba = category_pipeline.predict_proba(X_test)
    cat_conf = cat_proba.max(axis=1)
    low_conf_mask = cat_conf < LOW_CONFIDENCE_THRESHOLD
    low_conf_rate = float(low_conf_mask.mean())

    metrics = {
        "n_total_tickets": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_rate_on_test": low_conf_rate,
        "category": category_metrics,
        "urgency": urgency_metrics,
    }
    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a small held-out review log sample (what the dashboard would produce)
    review_df = pd.DataFrame({
        "ticket_text": X_test[low_conf_mask].values,
        "true_category": y_cat_test[low_conf_mask].values,
        "predicted_category": category_pipeline.predict(X_test)[low_conf_mask],
        "confidence": cat_conf[low_conf_mask].round(3),
    })
    review_df.to_csv(f"{REPORTS_DIR}/low_confidence_sample.csv", index=False)

    print(f"\nLow-confidence rate on test set (< {LOW_CONFIDENCE_THRESHOLD}): {low_conf_rate:.1%}")
    print("Saved models, figures, and metrics.json")


if __name__ == "__main__":
    main()
