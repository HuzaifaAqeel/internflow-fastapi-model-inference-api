"""
Generate a small labelled sentiment dataset for product reviews.

Approach (seeded + reproducible):
  1. ~30 hand-written seed reviews per class (positive / negative / neutral).
  2. Deterministic augmentation (random.Random(seed=42)) turns each seed
     into ~5 variants using light paraphrase templates: synonym swaps,
     intensifier insertion, and prefix/suffix framing.
  3. Exact-duplicate rows are dropped; the final table is balanced.

Output: data/reviews.csv with columns: text, sentiment

Run from the project root:
    python scripts/make_dataset.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "reviews.csv"
SEED = 42
TARGET_PER_CLASS = 150

POSITIVE_SEEDS = [
    "The product works perfectly and exceeded my expectations",
    "Excellent quality, I am very happy with this purchase",
    "Delivery was fast and the packaging was great",
    "This is the best thing I have bought all year",
    "Amazing value for the price, highly recommended",
    "The customer support team resolved my issue in minutes",
    "Super easy to set up, worked right out of the box",
    "The build quality feels premium and solid",
    "I love the design, it looks great on my desk",
    "Battery life is outstanding, lasts all day",
    "Very comfortable to use for long hours",
    "The app interface is clean and intuitive",
    "Great seller, item arrived exactly as described",
    "Performance is smooth with no lag at all",
    "I would definitely buy from this brand again",
    "The colors are vibrant and true to the pictures",
    "Installation took less than five minutes",
    "Works flawlessly with all my other devices",
    "The sound quality is rich and clear",
    "Very durable, survived a drop without a scratch",
    "The instructions were clear and easy to follow",
    "Perfect fit, no adjustments needed",
    "The material feels soft and high quality",
    "Exceeded the quality of the more expensive brand I used before",
    "Fast shipping and the item was well protected",
    "The features are exactly what I was looking for",
    "Very impressed with the attention to detail",
    "The screen is bright and easy to read",
    "My whole family loves using this",
    "A fantastic purchase, worth every rupee",
]

NEGATIVE_SEEDS = [
    "Stopped working after just two days, very disappointed",
    "The quality is terrible, feels cheap and flimsy",
    "Completely useless, a waste of money",
    "The battery drains in less than an hour",
    "Setup was a nightmare, instructions made no sense",
    "Arrived damaged and the seller refused a refund",
    "It overheats within minutes of use",
    "The worst purchase I have made this year",
    "Does not work as advertised at all",
    "The screen flickers constantly, unusable",
    "Customer support never replied to my emails",
    "Broke the first time I tried to use it",
    "The material smells awful and gave me a headache",
    "Very slow and laggy, freezes all the time",
    "Missing parts in the box, incomplete product",
    "The color is nothing like the pictures",
    "Fell apart within a week of normal use",
    "The app crashes every time I open it",
    "Overpriced for such poor performance",
    "The sound is distorted even at low volume",
    "Buttons are unresponsive and sticky",
    "The strap broke on the very first day",
    "Takes forever to charge and dies quickly",
    "The packaging was torn and the item was scratched",
    "I regret buying this, do not recommend it",
    "The lens is blurry, photos look terrible",
    "It disconnects from wifi constantly",
    "The zipper jammed and ripped the fabric",
    "Poorly made, edges are rough and uneven",
    "The return process was painful and slow",
]

NEUTRAL_SEEDS = [
    "The product arrived on time in standard packaging",
    "It works as described, nothing special",
    "Average quality, does the job for the price",
    "The item matches the listing photos",
    "Delivery took about a week as expected",
    "It is okay for basic everyday use",
    "The size is as listed in the description",
    "Functions normally, I have no complaints so far",
    "The color is close to what was shown online",
    "Standard product, meets the minimum requirements",
    "It does what it says on the box",
    "The material is ordinary, neither good nor bad",
    "I received the correct model and variant",
    "Setup followed the manual without issues",
    "The performance is acceptable for light use",
    "Neither impressed nor disappointed so far",
    "The weight is as specified in the specs",
    "It serves its purpose for now",
    "The buttons work, the screen turns on",
    "A plain product with no standout features",
    "The invoice and warranty card were included",
    "It fits in the space I measured",
    "The manual is written in two languages",
    "Charging takes the time stated in the manual",
    "The product code matches my order",
    "It is fine, I use it occasionally",
    "No problems during the first month",
    "The design is simple and functional",
    "Comparable to other products in this range",
    "I would rate it as average overall",
]

# Light paraphrase machinery -------------------------------------------------
PREFIXES = [
    "Honestly, ", "In my opinion, ", "After using it for a week, ",
    "So far, ", "To be fair, ", "",
]
SUFFIXES = [
    "", " overall.", " in my experience.", " for now.", " at this price point.",
]
SYNONYMS = {
    "great": ["great", "excellent", "wonderful", "fantastic"],
    "good": ["good", "decent", "solid", "fine"],
    "bad": ["bad", "poor", "awful", "terrible"],
    "love": ["love", "really like", "enjoy"],
    "hate": ["hate", "dislike", "cannot stand"],
    "fast": ["fast", "quick", "speedy"],
    "slow": ["slow", "sluggish"],
    "easy": ["easy", "simple", "straightforward"],
    "perfect": ["perfect", "flawless", "ideal"],
    "broken": ["broken", "faulty", "defective"],
}


def augment(seed: str, rng: random.Random) -> str:
    words = seed.split()
    out = []
    for w in words:
        key = w.lower().strip(".,!")
        if key in SYNONYMS and rng.random() < 0.35:
            out.append(rng.choice(SYNONYMS[key]))
        else:
            out.append(w)
    text = " ".join(out)
    text = f"{rng.choice(PREFIXES)}{text}{rng.choice(SUFFIXES)}".strip()
    return " ".join(text.split())


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: dict[str, list[str]] = {}
    seeds = {"positive": POSITIVE_SEEDS, "negative": NEGATIVE_SEEDS, "neutral": NEUTRAL_SEEDS}
    for label, seed_list in seeds.items():
        seen: set[str] = set()
        variants: list[str] = []
        for seed in seed_list:
            seen.add(seed)
            variants.append(seed)
        guard = 0
        while len(variants) < TARGET_PER_CLASS and guard < 4000:
            guard += 1
            cand = augment(rng.choice(seed_list), rng)
            if cand not in seen:
                seen.add(cand)
                variants.append(cand)
        rows[label] = variants[:TARGET_PER_CLASS]
    flat = [(t, label) for label, texts in rows.items() for t in texts]
    rng.shuffle(flat)
    return pd.DataFrame(flat, columns=["text", "sentiment"])


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}")
    print(df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
