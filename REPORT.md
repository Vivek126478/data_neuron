# Semantic Textual Similarity API

## Part A: Modeling Approach

- We use a supervised CrossEncoder model (`cross-encoder/stsb-distilroberta-base`) from Sentence-Transformers. Unlike TF‑IDF, Word2Vec, or vanilla cosine over embeddings, a CrossEncoder jointly attends to both input texts using Transformers and outputs a calibrated similarity score in [0,1]. This captures nuanced interactions (negation, word order, paraphrases) that bag‑of‑words or independent-encoding methods miss.
- Robust preprocessing normalizes unicode, casing, and punctuation while preserving intra‑word apostrophes and hyphens. This improves stability on noisy inputs without discarding semantics.
- Symmetry: we score both (text1, text2) and (text2, text1) and average. This reduces directional variance and yields a more reliable similarity estimate.
- Post‑processing constrains scores to [0,1] and rounds to 4 decimals for stable API responses.

## Part B: API and Deployment

- API built with Flask, endpoint `POST /calculate-similarity`.
  - Request: `{ "text1": "...", "text2": "..." }`
  - Response: `{ "similarity score": <float between 0 and 1> }`
  - Input validation ensures a proper JSON object and acceptable datatypes (string/number).
- Gunicorn WSGI server with 2 workers and 8 threads for concurrency. `Procfile` is provided for platform deployment (e.g., Render, Railway, Heroku-style dynos). Requirements pin modern versions of `sentence-transformers`, `torch`, and `transformers`.
- For cloud deployment on a free/low‑cost provider:
  1. Create a new web service (e.g., Render, Railway, or Heroku-compatible environment).
  2. Set the start command via `Procfile`: `web: gunicorn --workers 2 --threads 8 --timeout 120 --bind 0.0.0.0:$PORT app:app`.
  3. Ensure build installs from `requirements.txt`.
  4. After deploy, test with `client_example.py` or `curl`.

## Example

Request:
```json
{ "text1": "nuclear body seeks new tech .......", "text2": "terror suspects face arrest ......" }
```
Response:
```json
{ "similarity score": 0.20 }
```

## Notes
- CrossEncoder is compute‑heavier than cosine similarity over static embeddings but offers superior semantic alignment for STS. For production scaling, consider quantization or distillation, caching frequent pairs, and autoscaling workers.
