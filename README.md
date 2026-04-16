# Review-Powered Skincare Recommender

A product recommendation engine that surfaces skincare products based on **what real users actually said in their reviews** — not ingredient marketing copy or brand-sponsored labels.

## The Problem

Skincare shoppers face a paradox of choice: thousands of products, each promising transformative results through curated marketing language. Ingredient lists are technically accurate but clinically opaque to most consumers. Star ratings compress nuanced experiences into a single number. The most trustworthy signal — *what people with similar skin actually experienced* — is buried across hundreds of thousands of individual reviews that no shopper has time to read.

## The Solution

This system reads **238,929 customer reviews** across **8,494 Sephora skincare products** and learns a vocabulary of real user language — phrases like "cleared my cystic acne", "felt greasy on my oily skin", or "reduced my redness overnight". When a user selects their skin type, concern, and product category, the engine matches their profile against this review-derived vocabulary and returns the top 10 products.

Ranking blends two signals: **TF-IDF cosine similarity** (topical match between the profile query and a product's aggregated reviews) and a **Logistic Regression classifier** trained on review-level `is_recommended` labels (a learned quality signal derived from whether real reviewers recommended the product). Both signals are min-max normalized within each profile's candidate set and combined as `0.7 × cosine + 0.3 × classifier`, so a recommendation requires a product to be *both* topically relevant and positively received.

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
┌─────────────────┐     ┌──────────────────┐
│  Aggregate       │────▶│  Fit TF-IDF       │─────┐
│  Reviews         │     │  Vectorizer       │     │
│  per Product     │     │  (10K features)   │     │
└─────────────────┘     └──────────┬───────┘     │
                                   │              │
                                   ▼              ▼
                        ┌──────────────────┐  ┌────────────────────┐
                        │  Fit LR on       │  │  Precompute Top-10 │
                        │  review-level    │─▶│  per Profile (×210)│
                        │  is_recommended  │  │  — blend cosine +  │
                        └──────────────────┘  │    classifier      │
                                              └──────────┬─────────┘
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
│   │   ├── fit_classifier.py       # Train LR on review-level is_recommended
│   │   ├── score_products.py       # Per (product, skin_type) LR log-odds → parquet
│   │   ├── compute_similarity.py   # Cosine sim with category masking
│   │   └── precompute_all.py       # Blend cosine + classifier, write recommendations.csv
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

# Step 4 — Train the Logistic Regression classifier on is_recommended
python -m src.modeling.fit_classifier --env local

# Step 5 — Score every (product_id, skin_type) with the trained classifier
python -m src.modeling.score_products --env local

# Step 6 — Precompute blended top-10 recommendations per profile
python -m src.modeling.precompute_all --env local

# Step 7 — Create RDS schema and load data
python -m src.database.load_rds --env local

# Step 8 — Launch dashboard
streamlit run src/dashboard/app.py
```

## How It Works

1. **Aggregate reviews:** All review text for each product is concatenated into a single document, creating one text corpus entry per product.

2. **Fit TF-IDF:** A vectorizer (unigrams + bigrams, 10K features, sublinear TF, min_df=2) is fitted on the corpus. Each product becomes a sparse vector in a 10,000-dimensional term space.

3. **Train the Logistic Regression classifier:** LR is fit on review-level rows using `is_recommended` (1/0) as the label and the TF-IDF vectorizer above to transform each review's text. Held-out AUC is 0.9612 and F1 on the negative class is 0.7479 on a 20% stratified test split.

4. **Score products per skin type:** For each `(product_id, skin_type)` combination (plus an "overall" bucket used as a fallback for skin types missing from the data), reviews are aggregated and the classifier's `decision_function` produces a log-odds score. Log-odds are used instead of `predict_proba` because aggregating many positive-leaning reviews saturates probabilities near 1.0 and flattens discriminative signal.

5. **Build profile queries:** For each of 210 skin profile combinations, a natural-language query is constructed using expanded concern terms (e.g., "acne" → "acne breakouts pimples blemishes clogged pores").

6. **Blend and rank:** Each query is transformed through the TF-IDF vectorizer and compared against the product matrix using cosine similarity. Products are filtered to the profile's category. Within that candidate set, cosine sims and classifier log-odds are **both min-max normalized to [0, 1]**, then combined as `0.7 × norm_cosine + 0.3 × norm_classifier`. The top 10 by combined score are kept.

7. **Store and serve:** The 2,100 recommendation rows (210 profiles × 10) are loaded into RDS PostgreSQL with the blended score in `similarity_score`. The Streamlit dashboard queries RDS by skin profile and renders a ranked card grid with product name, brand, price, average rating, and match score.

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

The deployed system combines two components, each evaluated on its own terms:

- **TF-IDF + cosine similarity (ranking signal):** validated by inspecting top-10 outputs across representative profiles and by per-profile score spread after blending.
- **Logistic Regression classifier (quality signal):** held-out **AUC 0.9612** and **F1 = 0.7479 on the negative class** on a 20% stratified test split of 284K labeled review rows. `class_weight='balanced'` is used because the positive class (`is_recommended=1`) makes up ~84% of rows.

An earlier exploratory LR baseline (AUC 0.9816 on a different split) was the validation step that confirmed review text carries strong signal and motivated promoting the classifier from context-only into an **active ranking component** blended with cosine similarity at weight 0.3.

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | scikit-learn (TF-IDF, LogisticRegression), scipy (sparse matrices, cosine similarity) |
| Data | pandas, pyarrow (parquet I/O) |
| Database | PostgreSQL on AWS RDS, psycopg2 |
| Dashboard | Streamlit |
| Cloud | AWS S3, SageMaker, EC2, RDS |
| Testing | pytest |
