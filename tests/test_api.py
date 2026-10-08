"""
API tests for the FastAPI Model Inference & Embeddings API.

Run from the project root:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


LABELS = {"positive", "negative", "neutral"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["embedder_loaded"] is True
    assert body["model_version"] == "1.0.0"


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert set(body["labels"]) == LABELS
    assert body["embedding_dim"] == 512
    assert body["latency_target_ms"] == 200.0
    assert "/predict" in body["endpoints"] and "/embed" in body["endpoints"]


def test_predict_positive(client):
    r = client.post(
        "/predict",
        json={"text": "Excellent quality, I am very happy with this purchase"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["all_scores"].keys()) == LABELS
    assert abs(sum(body["all_scores"].values()) - 1.0) < 1e-3
    assert body["latency_ms"] < 200.0


def test_predict_negative(client):
    r = client.post(
        "/predict",
        json={"text": "Stopped working after two days, terrible quality"},
    )
    assert r.status_code == 200
    assert r.json()["sentiment"] == "negative"


def test_predict_neutral(client):
    r = client.post(
        "/predict",
        json={"text": "The item matches the listing photos"},
    )
    assert r.status_code == 200
    assert r.json()["sentiment"] == "neutral"


def test_predict_empty_text_rejected(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_missing_field_rejected(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_wrong_type_rejected(client):
    r = client.post("/predict", json={"text": 123})
    assert r.status_code == 422


def test_embed_single_text(client):
    r = client.post("/embed", json={"texts": ["The battery life is outstanding"]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["dim"] == 512
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == 512
    assert body["latency_ms"] < 200.0


def test_embed_batch(client):
    texts = [
        "Excellent quality, very happy",
        "Stopped working after two days",
        "It works as described",
    ]
    r = client.post("/embed", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert all(len(v) == 512 for v in body["embeddings"])


def test_embed_semantics_similar_texts_closer(client):
    def cosine(a, b):
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    r = client.post(
        "/embed",
        json={
            "texts": [
                "The battery life is outstanding and lasts all day",
                "Battery lasts the whole day, very impressive",
                "The screen flickers constantly, totally unusable",
            ]
        },
    )
    assert r.status_code == 200
    e0, e1, e2 = r.json()["embeddings"]
    assert cosine(e0, e1) > cosine(e0, e2)


def test_embed_empty_list_rejected(client):
    r = client.post("/embed", json={"texts": []})
    assert r.status_code == 422


def test_embed_blank_text_rejected(client):
    r = client.post("/embed", json={"texts": ["ok", "  "]})
    assert r.status_code == 422


def test_embed_too_many_texts_rejected(client):
    r = client.post("/embed", json={"texts": ["x"] * 101})
    assert r.status_code == 422


def test_embed_wrong_type_rejected(client):
    r = client.post("/embed", json={"texts": "not-a-list"})
    assert r.status_code == 422
