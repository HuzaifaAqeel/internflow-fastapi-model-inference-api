"""
Generate demo/sample_predictions.md: 12 live predictions + embedding examples.

Runs the trained API in-process and records real outputs, so the document
shows verified behaviour rather than hand-typed claims.

Run from the project root:
    python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SAMPLES = [
    "Absolutely love it, best purchase I have made this year",
    "The screen arrived cracked and support ignored my emails",
    "It does what it says on the box, nothing more",
    "Battery lasts two full days, incredibly impressed",
    "Stopped charging after a week, total waste of money",
    "Delivery took the expected five days, package was fine",
    "Setup was effortless and everything works perfectly",
    "The app crashes constantly and the device overheats",
    "Average build quality, acceptable for the price",
    "Sound is crisp and clear, great value overall",
    "The zipper broke on day one, very poor craftsmanship",
    "Size matches the description, using it daily without issues",
]

EMBED_PAIRS = [
    ("The battery life is outstanding and lasts all day", "Battery lasts the whole day, very impressive"),
    ("Terrible quality, broke within a week", "Excellent quality, very happy with it"),
]


def cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    with TestClient(app) as client:
        lines = ["# Sample Predictions", ""]
        lines.append("Live outputs from `POST /predict` (model v1.0.0):")
        lines.append("")
        lines.append("| # | Text | Predicted | Confidence | Latency |")
        lines.append("|---|------|-----------|------------|---------|")
        for i, text in enumerate(SAMPLES, 1):
            r = client.post("/predict", json={"text": text})
            assert r.status_code == 200, text
            b = r.json()
            lines.append(
                f"| {i} | {text} | {b['sentiment']} | {b['confidence']:.4f} | "
                f"{b['latency_ms']:.2f} ms |"
            )

        lines += ["", "## Embedding similarity", ""]
        lines.append(
            "Cosine similarity between `POST /embed` vectors (512-dim TF-IDF):"
        )
        lines.append("")
        for t1, t2 in EMBED_PAIRS:
            r = client.post("/embed", json={"texts": [t1, t2]})
            assert r.status_code == 200
            e1, e2 = r.json()["embeddings"]
            sim = cosine(e1, e2)
            lines.append(f"- sim({t1!r},")
            lines.append(f"      {t2!r}) = **{sim:.4f}**")

    out = ROOT / "demo" / "sample_predictions.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(SAMPLES)} predictions, {len(EMBED_PAIRS)} similarity pairs)")


if __name__ == "__main__":
    main()
