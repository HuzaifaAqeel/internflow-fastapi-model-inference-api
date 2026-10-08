"""
Train the sentiment classifier and the text-embedding model.

Classifier pipeline:
    clean_text (FunctionTransformer)
      -> FeatureUnion(TF-IDF word 1-2 grams, TF-IDF char_wb 3-5 grams)
      -> classifier

- Main model: LogisticRegression tuned with GridSearchCV (5-fold, f1_macro).
- Baseline:   MultinomialNB on the same features.
- Stratified 80/20 train/test split (seeded).

Embedding model:
- A standalone TfidfVectorizer (word 1-2 grams, max_features=512,
  sublinear TF, L2 norm) fitted on the cleaned training texts. Served
  dense, so vectors work directly as inputs to cosine similarity.

Saves under models/:
    pipeline.joblib, embedder.joblib, metadata.json, metrics.json,
    classification_report.txt, confusion_matrix.png

Run from the project root:
    python src/train.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_batch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "reviews.csv"
MODELS = ROOT / "models"
RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]
EMBEDDING_DIM = 512
MODEL_VERSION = "1.0.0"


def build_features() -> FeatureUnion:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def main() -> None:
    df = pd.read_csv(DATA)
    X = df["text"].tolist()
    y = df["sentiment"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    features = build_features()
    grid = GridSearchCV(
        Pipeline(
            [
                ("clean", FunctionTransformer(clean_batch, validate=False)),
                ("features", features),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        {"clf__C": [0.5, 1.0, 2.0, 5.0, 10.0]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"Best C: {grid.best_params_['clf__C']}  cv f1_macro: {grid.best_score_:.4f}")

    # Baseline on the same features.
    baseline = Pipeline(
        [
            ("clean", FunctionTransformer(clean_batch, validate=False)),
            ("features", build_features()),
            ("clf", MultinomialNB()),
        ]
    )
    baseline.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, baseline.predict(X_test))
    base_f1 = f1_score(y_test, baseline.predict(X_test), average="macro")
    print(f"MultinomialNB baseline — acc: {base_acc:.4f}  f1_macro: {base_f1:.4f}")

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test — acc: {acc:.4f}  f1_macro: {f1:.4f}")

    # Standalone embedder: fit on cleaned training texts only.
    embedder = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=EMBEDDING_DIM,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    embedder.fit(clean_batch(X_train))
    print(f"Embedder fitted: dim={len(embedder.vocabulary_)} (capped at {EMBEDDING_DIM})")

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODELS / "pipeline.joblib")
    joblib.dump(embedder, MODELS / "embedder.joblib")

    metadata = {
        "model_name": "tfidf-logreg-sentiment",
        "model_version": MODEL_VERSION,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "labels": LABELS,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "dataset": "data/reviews.csv (450 rows, 150/class, seeded generation)",
        "classifier": f"LogisticRegression(C={grid.best_params_['clf__C']}) on TF-IDF word 1-2 + char_wb 3-5",
        "embedding_model": f"TF-IDF word 1-2, max_features={EMBEDDING_DIM}, sublinear_tf, l2",
        "embedding_dim": len(embedder.vocabulary_),
        "latency_target_ms": 200.0,
        "random_state": RANDOM_STATE,
    }
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "baseline_accuracy": round(float(base_acc), 4),
        "baseline_f1_macro": round(float(base_f1), 4),
        "best_C": grid.best_params_["clf__C"],
        "cv_f1_macro": round(float(grid.best_score_), 4),
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, labels=LABELS)
    )

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=20)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(MODELS / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    print(f"Artifacts saved -> {MODELS}")


if __name__ == "__main__":
    main()
