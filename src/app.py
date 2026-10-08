"""
app.py
------
Streamlit dashboard for the AI Customer Support Ticket Triage system
(project brief step 6: "Expose the model through a small API/dashboard and
log low-confidence predictions for human review.")

Run with:
    streamlit run app.py
"""

import os
import csv
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
REVIEW_LOG_PATH = os.path.join(BASE_DIR, "..", "reports", "human_review_queue.csv")

LOW_CONFIDENCE_THRESHOLD = 0.90

st.set_page_config(page_title="AI Ticket Triage", page_icon="🎫", layout="wide")


@st.cache_resource
def load_models():
    category_model = joblib.load(os.path.join(MODELS_DIR, "category_model.joblib"))
    urgency_model = joblib.load(os.path.join(MODELS_DIR, "urgency_model.joblib"))
    return category_model, urgency_model


def log_for_review(ticket_text, category, confidence):
    file_exists = os.path.isfile(REVIEW_LOG_PATH)
    with open(REVIEW_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticket_text", "predicted_category", "confidence"])
        writer.writerow([datetime.utcnow().isoformat(), ticket_text, category, round(confidence, 3)])


def predict_ticket(ticket_text, category_model, urgency_model):
    cleaned = clean_text(ticket_text)

    cat_proba = category_model.predict_proba([cleaned])[0]
    cat_labels = category_model.classes_
    cat_pred = cat_labels[cat_proba.argmax()]
    cat_conf = float(cat_proba.max())

    urg_proba = urgency_model.predict_proba([cleaned])[0]
    urg_labels = urgency_model.classes_
    urg_pred = urg_labels[urg_proba.argmax()]
    urg_conf = float(urg_proba.max())

    return {
        "category": cat_pred,
        "category_confidence": cat_conf,
        "category_distribution": dict(zip(cat_labels, cat_proba)),
        "urgency": urg_pred,
        "urgency_confidence": urg_conf,
        "urgency_distribution": dict(zip(urg_labels, urg_proba)),
    }


def route_label(category, urgency):
    """Simple routing rule combining predicted category + urgency into a queue name."""
    if urgency == "critical":
        return f"🔴 {category.title()} — Escalate immediately"
    if urgency == "high":
        return f"🟠 {category.title()} — Priority queue"
    if urgency == "medium":
        return f"🟡 {category.title()} — Standard queue"
    return f"🟢 {category.title()} — Backlog"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎫 AI Customer Support Ticket Triage")
st.caption(
    "Predicts a ticket's **category** and **urgency**, routes it to the right queue, "
    "and flags low-confidence predictions for human review."
)

category_model, urgency_model = load_models()

tab_predict, tab_batch, tab_review = st.tabs(
    ["🔎 Triage a Ticket", "📂 Batch Upload", "🧑‍💻 Human Review Queue"]
)

with tab_predict:
    col_in, col_out = st.columns([1, 1])

    with col_in:
        default_text = (
            "URGENT: The billing portal charged me twice for my Pro subscription "
            "and support hasn't replied in 2 days. This is affecting my whole team."
        )
        ticket_text = st.text_area("Paste the incoming ticket text:", value=default_text, height=160)
        run = st.button("Triage Ticket", type="primary")

    with col_out:
        if run and ticket_text.strip():
            result = predict_ticket(ticket_text, category_model, urgency_model)

            st.metric("Predicted Category", result["category"].title(),
                       f"{result['category_confidence']*100:.1f}% confidence")
            st.metric("Predicted Urgency", result["urgency"].title(),
                       f"{result['urgency_confidence']*100:.1f}% confidence")
            st.markdown(f"**Suggested routing:** {route_label(result['category'], result['urgency'])}")

            st.write("Category probability breakdown")
            st.bar_chart(pd.Series(result["category_distribution"]))
            st.write("Urgency probability breakdown")
            st.bar_chart(pd.Series(result["urgency_distribution"]))

            if result["category_confidence"] < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ Category confidence ({result['category_confidence']*100:.1f}%) is below the "
                    f"{LOW_CONFIDENCE_THRESHOLD*100:.0f}% threshold. Logged for human review."
                )
                log_for_review(ticket_text, result["category"], result["category_confidence"])
            else:
                st.success("✅ High-confidence prediction — safe to auto-route.")
        elif run:
            st.error("Please paste a ticket first.")

with tab_batch:
    st.write("Upload a CSV with a `ticket_text` column to triage many tickets at once.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "ticket_text" not in batch_df.columns:
            st.error("CSV must contain a 'ticket_text' column.")
        else:
            cleaned = batch_df["ticket_text"].astype(str).map(clean_text)
            cat_proba = category_model.predict_proba(cleaned)
            urg_proba = urgency_model.predict_proba(cleaned)

            batch_df["predicted_category"] = category_model.classes_[cat_proba.argmax(axis=1)]
            batch_df["category_confidence"] = cat_proba.max(axis=1).round(3)
            batch_df["predicted_urgency"] = urgency_model.classes_[urg_proba.argmax(axis=1)]
            batch_df["urgency_confidence"] = urg_proba.max(axis=1).round(3)
            batch_df["needs_human_review"] = batch_df["category_confidence"] < LOW_CONFIDENCE_THRESHOLD

            st.dataframe(batch_df, use_container_width=True)

            low_conf = batch_df[batch_df["needs_human_review"]]
            for _, row in low_conf.iterrows():
                log_for_review(row["ticket_text"], row["predicted_category"], row["category_confidence"])

            st.download_button(
                "Download results as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="triaged_tickets.csv",
                mime="text/csv",
            )
            st.info(f"{low_conf.shape[0]} of {len(batch_df)} tickets flagged for human review.")

with tab_review:
    st.write("Tickets whose category prediction fell below the confidence threshold.")
    if os.path.isfile(REVIEW_LOG_PATH):
        review_df = pd.read_csv(REVIEW_LOG_PATH, encoding="utf-8")
        st.dataframe(review_df, use_container_width=True)
    else:
        st.info("No tickets have been logged for human review yet in this session.")
