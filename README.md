# Review-Powered Skincare Recommender

A product recommendation engine that surfaces skincare products based on **what real users actually said in their reviews** — not ingredient marketing copy or brand-sponsored labels.

## The Problem

Skincare shoppers face a paradox of choice: thousands of products, each promising transformative results through curated marketing language. Ingredient lists are technically accurate but clinically opaque to most consumers. Star ratings compress nuanced experiences into a single number. The most trustworthy signal — *what people with similar skin actually experienced* — is buried across hundreds of thousands of individual reviews that no shopper has time to read.

## The Solution

This system reads **238,929 customer reviews** across **8,494 Sephora skincare products** and learns a vocabulary of real user language — phrases like "cleared my cystic acne", "felt greasy on my oily skin", or "reduced my redness overnight". When a user selects their skin type, concern, and product category, the engine matches their profile against this review-derived vocabulary and returns the top 10 products whose aggregate review language most closely aligns with their needs.

The recommendations are grounded in **collective user experience**, not marketing claims.

## Why TF-IDF + Cosine Similarity?

This project deliberately uses TF-IDF vectorization and cosine similarity over alternatives like collaborative filtering, deep learning embeddings, or LLM-based approaches. Here's why each design choice fits this problem:

**TF-IDF captures what makes each product's reviews distinctive.**
Term Frequency–Inverse Document Frequency doesn't just count words — it upweights terms that are frequent in a product's reviews but rare across the full corpus. If hundreds of reviewers mention "cystic acne" for one moisturizer but the phrase is uncommon across all 8,494 products, TF-IDF assigns it a high weight. This naturally surfaces the language that differentiates products from each other, which is exactly what a recommendation needs.

**Cosine similarity measures topical alignment, not magnitude.**
A product with 500 reviews and one with 50 reviews produce TF-IDF vectors of very different magnitudes, but cosine similarity normalizes for this — it measures the *angle* between vectors, not their length. This means a niche product with 50 highly relevant reviews can outrank a blockbuster with 500 generic ones, as long as the review language aligns more closely with the user's concern. This is the right behavior for personalized recommendations.

**Bigrams preserve meaningful skincare phrases.**
Unigram-only models lose critical context: "dry" and "skin" are common individually, but "dry skin" as a bigram is a precise signal. "Fine lines", "clogged pores", "dark spots" — skincare language is inherently phrasal. The (1,2)-gram range captures both individual terms and these two-word compounds without the sparsity explosion of higher n-grams.

**Why not collaborative filtering?**
Collaborative filtering ("users who bought X also bought Y") requires user purchase or interaction histories. This dataset has reviews, not purchase graphs. TF-IDF operates on the text content itself — it works with what we have and extracts richer signal from it.

**Why not deep learning embeddings (Word2Vec, BERT)?**
Transformer embeddings capture semantic similarity ("moisturizing" ≈ "hydrating") but require significant compute, are harder to interpret, and can over-generalize — mapping different concerns into the same embedding neighborhood. TF-IDF's explicit vocabulary makes the system inspectable: you can see exactly which terms drove a recommendation. For a precomputed lookup table of 210 profile combinations, the added complexity of neural embeddings offers diminishing returns over a well-tuned TF-IDF pipeline.

## Architecture

```
Sephora SQLite DB
       │
       ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Aggregate       │────▶│  Fit TF-IDF       │────▶│  Precompute       │
│  Reviews         │     │  Vectorizer       │     │  Top-10 per       │
│  per Product     │     │  (10K features)   │     │  Profile (×210)   │
└─────────────────┘     └──────────────────┘     └───────┬───────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                        ┌──────────────────┐      │ RDS         │
                        │  Streamlit       │◀─────│ PostgreSQL  │
                        │  Dashboard       │      │             │
                        └──────────────────┘      └─────────────┘
```

**AWS deployment:** S3 (raw data + model artifacts) → SageMaker (processing + batch transform) → RDS PostgreSQL (precomputed lookup) → Streamlit on EC2 (user-facing dashboard).

## Dataset

Source: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle)

| Dimension | Count |
|-----------|-------|
| Customer reviews | 238,929 |
| Unique products | 8,494 |
| Skin types | 5 (dry, oily, normal, combination, sensitive) |
| Skin concerns | 7 (acne, dryness, redness, sensitivity, anti-aging, brightening, pores) |
| Product categories | 6 (Moisturizers, Treatments, Cleansers, Sunscreen, Eye Care, Lip Balms & Treatments) |
| Profile combinations | 210 (5 × 7 × 6) |
| Precomputed recommendations | 2,100 (210 profiles × top 10 each) |

Each review includes the reviewer's skin type, skin tone, product rating, recommendation flag, and full review text — providing both the text signal for TF-IDF and the skin profile metadata for query construction.

## Project Structure

```
skincare-recommender/
├── src/
│   ├── processing/
│   │   ├── aggregate_reviews.py    # Concatenate review text per product
│   │   └── build_profiles.py       # Generate 210 skin-profile queries
│   ├── modeling/
│   │   ├── fit_tfidf.py            # Fit vectorizer, save .pkl + .npz
│   │   ├── compute_similarity.py   # Cosine sim with category masking
│   │   └── precompute_all.py       # Loop all profiles → recommendations.csv
│   ├── database/
│   │   ├── schema.sql              # PostgreSQL CREATE TABLE + index
│   │   └── load_rds.py             # Bulk-load products + recs into RDS
│   ├── dashboard/
│   │   └── app.py                  # Streamlit UI (card grid, star ratings)
│   └── utils/
│       ├── config.py               # --env local|aws path resolution
│       └── logger.py               # Consistent logging setup
├── sagemaker/
│   ├── processing_job.py           # SageMaker Processing Job config
│   └── batch_transform.py          # Batch Transform config
├── tests/
│   └── test_similarity.py          # Ranking + masking smoke tests
├── data/
│   ├── raw/                        # Original SQLite DB
│   └── processed/                  # Parquets + precomputed CSV
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in RDS + S3 values
```

## Running the Pipeline

Every script supports `--env local` (filesystem) or `--env aws` (S3/SageMaker). Local mode reads the SQLite DB from the repo root.

```bash
# Step 1 — Aggregate review text per product
python -m src.processing.aggregate_reviews --env local

# Step 2 — Generate all 210 skin-profile query strings
python -m src.processing.build_profiles --env local

# Step 3 — Fit TF-IDF vectorizer on aggregated corpus
python -m src.modeling.fit_tfidf --env local

# Step 4 — Precompute top-10 recommendations per profile
python -m src.modeling.precompute_all --env local

# Step 5 — Create RDS schema and load data
python -m src.database.load_rds --env local

# Step 6 — Launch dashboard
streamlit run src/dashboard/app.py
```

## How It Works

1. **Aggregate reviews:** All review text for each product is concatenated into a single document, creating one text corpus entry per product (8,494 documents).

2. **Fit TF-IDF:** A vectorizer (unigrams + bigrams, 10K features, sublinear TF, min_df=2) is fitted on the corpus. Each product becomes a sparse vector in a 10,000-dimensional term space.

3. **Build profile queries:** For each of 210 skin profile combinations, a natural-language query is constructed using expanded concern terms (e.g., "acne" → "acne breakouts pimples blemishes clogged pores").

4. **Compute similarity:** Each query is transformed through the fitted vectorizer and compared against the product matrix using cosine similarity. Products are filtered by category — a moisturizer query only ranks moisturizers — and the top 10 are kept.

5. **Store and serve:** The 2,100 recommendation rows are loaded into RDS PostgreSQL. The Streamlit dashboard queries RDS by skin profile and renders a ranked card grid with product name, brand, price, average rating, and similarity score.

## Dashboard

The Streamlit app ("The Glow Lab") presents three inputs — skin type, skin concern, and product category — and returns the top 10 matching products as styled cards showing:

- Rank badge
- Brand and product name
- Price
- Star rating with review count
- Match score (similarity as a percentage bar)

## Tests

```bash
pytest tests/
```

The test suite validates that cosine similarity ranking returns expected products and that category masking correctly restricts the candidate pool.

## Model Validation

A Logistic Regression classifier was trained on TF-IDF features as a baseline signal check, achieving AUC 0.9816 and F1 0.79 on the negative class. This confirmed that review text carries strong discriminative signal for product quality — validating TF-IDF as the right feature representation. The classifier itself is not part of the deployed system; the production model is the cosine similarity recommender described above.

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | scikit-learn (TF-IDF), scipy (sparse matrices, cosine similarity) |
| Data | pandas, pyarrow (parquet I/O) |
| Database | PostgreSQL on AWS RDS, psycopg2 |
| Dashboard | Streamlit |
| Cloud | AWS S3, SageMaker, EC2, RDS |
| Testing | pytest |
