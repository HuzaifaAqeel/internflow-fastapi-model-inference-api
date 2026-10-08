"""
Async FastAPI service: real-time sentiment predictions + text embeddings.

Endpoints:
  GET  /health    -> {"status": "ok", "model": ..., "model_loaded": ...}
  GET  /metadata  -> model card: version, labels, metrics, embedding dim,
                     latency target
  POST /predict   -> {"text", "sentiment", "confidence", "all_scores",
                      "latency_ms"}
  POST /embed     -> {"embeddings": [[...]], "dim": N, "count": M,
                      "latency_ms"}

All inference handlers are `async def`. scikit-learn is synchronous, so
predictions/embeddings run in a bounded ThreadPoolExecutor via
`loop.run_in_executor(...)` — the event loop never blocks on inference,
and concurrent requests are served in parallel. Each response carries the
server-side inference latency in milliseconds (the sub-200ms target).

Artifacts (models/pipeline.joblib, models/embedder.joblib,
models/metadata.json) are produced by src/train.py and loaded once at
startup via the lifespan handler.
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "models" / "pipeline.joblib"
EMBEDDER_PATH = ROOT / "models" / "embedder.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

# The pickled pipeline references src.preprocess.clean_batch, so make the
# src package importable before unpickling.
sys.path.insert(0, str(ROOT / "src"))
import preprocess  # noqa: F401,E402  (imported for its side effect on unpickling)
from preprocess import clean_batch  # noqa: E402

LATENCY_TARGET_MS = 200.0

pipeline = None
embedder = None
metadata: dict = {}
executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, embedder, metadata, executor
    missing = [p for p in (PIPELINE_PATH, EMBEDDER_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Model artifacts missing: {[str(p) for p in missing]}. "
            "Run src/train.py first."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    embedder = joblib.load(EMBEDDER_PATH)
    import json

    metadata = json.loads(METADATA_PATH.read_text())
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="infer")
    yield
    executor.shutdown(wait=True)


app = FastAPI(
    title="FastAPI Model Inference & Embeddings API",
    version="1.0.0",
    description=(
        "Async inference service: sentiment predictions and TF-IDF text "
        "embeddings with a sub-200ms latency target."
    ),
    lifespan=lifespan,
)


# --- request/response models -------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    all_scores: dict[str, float]
    latency_ms: float


class EmbedRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=100, description="Texts to embed"
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str
    model_loaded: bool
    embedder_loaded: bool


# --- blocking work, offloaded to the thread pool -----------------------------
def _predict_sync(text: str) -> tuple[str, float, dict[str, float]]:
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def _embed_sync(texts: list[str]) -> tuple[list[list[float]], int]:
    cleaned = clean_batch(texts)
    mat = embedder.transform(cleaned)
    return mat.toarray().astype(float).tolist(), mat.shape[1]


# --- endpoints ----------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=metadata.get("model_name", "tfidf-logreg-sentiment"),
        model_version=metadata.get("model_version", "1.0.0"),
        model_loaded=pipeline is not None,
        embedder_loaded=embedder is not None,
    )


@app.get("/metadata")
async def get_metadata():
    return {
        **metadata,
        "endpoints": ["/health", "/metadata", "/predict", "/embed"],
        "async": True,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    if pipeline is None or executor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    sentiment, confidence, scores = await loop.run_in_executor(executor, _predict_sync, text)
    latency_ms = (time.perf_counter() - started) * 1000
    return PredictResponse(
        text=req.text,
        sentiment=sentiment,
        confidence=confidence,
        all_scores=scores,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    texts = [t.strip() for t in req.texts]
    if any(not t for t in texts):
        raise HTTPException(status_code=422, detail="texts must not contain blanks")
    if embedder is None or executor is None:
        raise HTTPException(status_code=503, detail="embedder not loaded")
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    vectors, dim = await loop.run_in_executor(executor, _embed_sync, texts)
    latency_ms = (time.perf_counter() - started) * 1000
    return EmbedResponse(
        embeddings=vectors,
        dim=dim,
        count=len(vectors),
        latency_ms=round(latency_ms, 3),
    )
