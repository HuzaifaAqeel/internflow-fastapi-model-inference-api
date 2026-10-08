"""
Latency benchmark: measures sequential and concurrent inference latency.

- Warms up the model, then times N sequential /predict and /embed calls
  through an in-process TestClient (no network overhead).
- Fires M concurrent /predict requests via httpx AsyncClient + ASGI transport.
- Prints p50/p95/mean/max and saves benchmark/results.json.

Run from the project root:
    python scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, lifespan  # noqa: E402

N_SEQ = 200
N_CONCURRENT = 50
OUT = ROOT / "benchmark" / "results.json"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def stats(name: str, values: list[float]) -> dict:
    return {
        "endpoint": name,
        "n": len(values),
        "p50_ms": round(percentile(values, 50), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "mean_ms": round(sum(values) / len(values), 2),
        "max_ms": round(max(values), 2),
    }


def sequential(client: TestClient) -> dict:
    out = {}
    # warmup
    for _ in range(10):
        client.post("/predict", json={"text": "Great product, fast delivery"})
        client.post("/embed", json={"texts": ["Great product, fast delivery"]})

    lat = []
    for i in range(N_SEQ):
        t0 = time.perf_counter()
        r = client.post("/predict", json={"text": f"Review {i}: great product, fast delivery"})
        assert r.status_code == 200
        lat.append((time.perf_counter() - t0) * 1000)
    out["predict_sequential"] = stats("POST /predict", lat)

    lat = []
    for i in range(N_SEQ):
        t0 = time.perf_counter()
        r = client.post("/embed", json={"texts": [f"Review {i}: great product, fast delivery"]})
        assert r.status_code == 200
        lat.append((time.perf_counter() - t0) * 1000)
    out["embed_sequential"] = stats("POST /embed", lat)
    return out


async def concurrent() -> dict:
    async def one(ac: httpx.AsyncClient, i: int) -> float:
        t0 = time.perf_counter()
        r = await ac.post("/predict", json={"text": f"Concurrent review {i}: great product"})
        assert r.status_code == 200
        return (time.perf_counter() - t0) * 1000

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            lat = await asyncio.gather(*[one(ac, i) for i in range(N_CONCURRENT)])
    return {"predict_concurrent_50": stats("POST /predict (50 concurrent)", list(lat))}


def main() -> None:
    results: dict = {"target_ms": 200.0}
    with TestClient(app) as client:
        results.update(sequential(client))
    results.update(asyncio.run(concurrent()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print(f"{'scenario':32s} {'p50':>8s} {'p95':>8s} {'mean':>8s} {'max':>8s}")
    for key, s in results.items():
        if key == "target_ms":
            continue
        print(
            f"{s['endpoint']:32s} {s['p50_ms']:>7.2f}m {s['p95_ms']:>7.2f}m "
            f"{s['mean_ms']:>7.2f}m {s['max_ms']:>7.2f}m"
        )
    slowest = max(
        (s["p95_ms"], s["endpoint"])
        for k, s in results.items()
        if k != "target_ms"
    )
    print(f"\nSlowest p95: {slowest[0]:.2f}ms ({slowest[1]}) — target < 200ms")
    print(f"Results saved -> {OUT}")


if __name__ == "__main__":
    main()
