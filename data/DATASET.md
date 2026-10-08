# Dataset

`data/reviews.csv` — 450 short product-review texts, balanced across three
sentiment classes (`positive`, `negative`, `neutral`, 150 each).

## How it was created

1. **Hand-written seeds.** 30 seed review sentences were written per class
   (90 total). They cover common review topics: quality, delivery,
   battery, setup, support, packaging, and value for money.
2. **Seeded augmentation.** `scripts/make_dataset.py` expands each seed to
   ~5 variants with `random.Random(42)`:
   - synonym swaps from a small curated dictionary,
   - optional review-style prefixes ("Honestly, ", "So far, ", ...) and
     suffixes (" overall.", " in my experience.", ...).
3. **Deduplication.** Exact-duplicate rows are dropped; the final table is
   capped at exactly 150 rows per class and shuffled.

The process is fully deterministic: re-running `python scripts/make_dataset.py`
regenerates the identical CSV.

## Format

| column    | type   | description                                |
|-----------|--------|--------------------------------------------|
| text      | string | the review text                            |
| sentiment | string | one of `positive`, `negative`, `neutral`   |

## Limitations

- Synthetic data: reviews were written/augmented by the author, not scraped
  from a real marketplace.
- Reviews are short (5–15 words) and single-topic; real reviews mix topics.
- Neutral reviews lean toward "average product" phrasing, so the class
  boundary with mild positives/negatives is thin.
