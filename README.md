# AI Customer Support Ticket Triage

An end-to-end machine learning system that reads incoming customer support
tickets, predicts their **category** (billing / technical / account / product)
and **urgency** (low / medium / high / critical), and routes them to the
appropriate queue — flagging low-confidence predictions for human review.

## Project structure

```
ticket_triage/
├── data/
│   └── support_tickets.csv        # synthetic labeled ticket dataset (880 rows)
├── src/
│   ├── generate_data.py           # synthetic dataset generator
│   ├── preprocessing.py           # text cleaning utilities
│   ├── train.py                   # training + evaluation pipeline
│   └── app.py                     # Streamlit dashboard
├── models/
│   ├── category_model.joblib      # TF-IDF + Calibrated LinearSVC
│   └── urgency_model.joblib       # TF-IDF + Logistic Regression
├── figures/
│   ├── confusion_category.png
│   └── confusion_urgency.png
├── reports/
│   ├── metrics.json                # all evaluation numbers
│   └── low_confidence_sample.csv   # example human-review cases
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python src/generate_data.py

# 2. Train + evaluate both models
python src/train.py

# 3. Launch the dashboard
streamlit run src/app.py
```

## Approach (mapped to the assignment brief)

1. **Data** — Since no proprietary helpdesk export was available, a synthetic
   but realistic dataset was generated: 880 tickets across 4 categories, with
   varied urgency language, ~18% cross-topic ("ambiguous") tickets, ~10%
   typos, and ~4% deliberate label noise to mimic real annotation error.
2. **Cleaning** — lowercasing, URL/punctuation stripping, whitespace
   normalization (`preprocessing.py`).
3. **Features** — TF-IDF with unigrams + bigrams, sublinear TF scaling.
4. **Models** — two independent classifiers sharing the same feature
   pipeline: a calibrated Linear SVM for category (4-class) and a Logistic
   Regression for urgency (4-class, ordinal-leaning).
5. **Evaluation** — accuracy, macro/weighted F1, full precision/recall/F1 per
   class, and confusion matrices (see `figures/`).
6. **Deployment / human-in-the-loop** — a Streamlit dashboard exposes single
   and batch ticket triage, shows probability breakdowns, and automatically
   logs any prediction below a 90% confidence threshold to a human review
   queue (`reports/human_review_queue.csv`), so uncertain cases are never
   silently auto-routed.

## Suggested stack

Python (pandas, scikit-learn, matplotlib) for data prep and modeling;
Git/GitHub for version control; Streamlit for the dashboard layer.

## Notes for real-world deployment

- Replace the synthetic generator with real historical ticket exports.
- Retrain periodically as ticket vocabulary and product surface evolve.
- Consider swapping TF-IDF for sentence embeddings if paraphrase diversity
  in real tickets is higher than this synthetic set.
- Track the human-review queue size over time as a live proxy for model
  drift — a rising queue means retraining is due.
