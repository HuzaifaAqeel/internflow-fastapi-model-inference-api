# Sample Predictions

Live outputs from `POST /predict` (model v1.0.0):

| # | Text | Predicted | Confidence | Latency |
|---|------|-----------|------------|---------|
| 1 | Absolutely love it, best purchase I have made this year | negative | 0.6822 | 53.90 ms |
| 2 | The screen arrived cracked and support ignored my emails | negative | 0.6133 | 5.00 ms |
| 3 | It does what it says on the box, nothing more | neutral | 0.9156 | 2.88 ms |
| 4 | Battery lasts two full days, incredibly impressed | positive | 0.4289 | 26.18 ms |
| 5 | Stopped charging after a week, total waste of money | negative | 0.8169 | 10.64 ms |
| 6 | Delivery took the expected five days, package was fine | neutral | 0.5748 | 3.85 ms |
| 7 | Setup was effortless and everything works perfectly | positive | 0.7270 | 13.79 ms |
| 8 | The app crashes constantly and the device overheats | negative | 0.9096 | 3.10 ms |
| 9 | Average build quality, acceptable for the price | neutral | 0.7617 | 2.60 ms |
| 10 | Sound is crisp and clear, great value overall | positive | 0.9455 | 2.96 ms |
| 11 | The zipper broke on day one, very poor craftsmanship | negative | 0.9261 | 2.62 ms |
| 12 | Size matches the description, using it daily without issues | neutral | 0.9035 | 2.72 ms |

## Embedding similarity

Cosine similarity between `POST /embed` vectors (512-dim TF-IDF):

- sim('The battery life is outstanding and lasts all day',
      'Battery lasts the whole day, very impressive') = **0.4982**
- sim('Terrible quality, broke within a week',
      'Excellent quality, very happy with it') = **0.2646**
