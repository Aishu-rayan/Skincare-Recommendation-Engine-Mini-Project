# Design Plan — Review-Powered Skincare Recommender

## Project overview
- Name: Review-Powered Skincare Recommender
- Goal: Build an end-to-end TF-IDF product recommendation system using the Sephora Products and Skincare Reviews dataset and deploy it on AWS with a Streamlit dashboard.

## The ML approach 
- Aggregate all review text per product_id into a single string
- Fit a TF-IDF vectorizer (scikit-learn, unigrams + bigrams, max 
  10,000 features) on that aggregated corpus
- Transform each product into a TF-IDF vector → sparse product matrix
- For each skin profile query (skin_type × skin_concern × category), 
  compute cosine similarity between the query vector and the product 
  matrix
- Precompute top-10 results for every profile combination and store in RDS — no real-time inference
- Model statement: "We fit TF-IDF on aggregated review text grouped by product, compute cosine similarity against skin-profile queries, and precompute top-10 recommendations per profile into RDS for fast lookup."

## Dataset
- 238,929 customer reviews
- 8,494 unique skincare products
- Two main tables: select_customer_reviews and product_info
Customer Review Structure (238K+ records)
Each review includes rich profiling:

- Skin Profile: skin_type (dry/oily/normal/combination), skin_tone, eye_color
- Review Quality: rating (1-5), is_recommended flag, loves_count, helpfulness score
- Product Details: Brand, category (3-level hierarchy), price, ingredients, highlights
- Review Content: Title, full review text, submission date
- Product Data (8.5K products)
- Complete product information with:
Detailed ingredients lists and product highlights
Multiple price points (regular, sale, value pricing)
Product variations and sizing options
Categorization and availability flags
- Also refer to @dataset_description.md for better clarity

## AWS architecture
- S3 bucket structure:
  skincare-recommender/
  ├── raw/                     ← original CSVs (never overwrite)
  ├── processed/               ← aggregated reviews, joined profiles
  ├── models/                  ← tfidf_vectorizer.pkl, 
  │                               product_vectors.npz
  ├── precomputed/             ← recommendations.csv (top-10 per 
  │                               profile before RDS load)
  └── logs/                    ← job logs
- SageMaker Processing Job: reads raw/ → outputs processed/ 
  and models/
- SageMaker Batch Transform: reads models/ → outputs precomputed/
- RDS PostgreSQL: two tables — products and recommendations 
  (see schema below)
- Streamlit on EC2: queries RDS by skin profile → renders ranked 
  product cards

## RDS schema
products table:
  product_id VARCHAR PK
  product_name TEXT
  brand_name TEXT
  category TEXT
  price_usd FLOAT
  avg_rating FLOAT
  review_count INT

recommendations table:
  profile_id VARCHAR        ← e.g. "oily_acne_moisturizer"
  skin_type TEXT
  skin_concern TEXT
  category TEXT
  rank INT                  ← 1 to 10
  product_id VARCHAR FK → products
  similarity_score FLOAT

## Project file structure
skincare-recommender/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/               ← local copies of CSVs
│   └── processed/         ← local intermediate outputs
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── processing/
│   │   ├── aggregate_reviews.py    ← group review text by product
│   │   └── build_profiles.py       ← generate all query combinations
│   ├── modeling/
│   │   ├── fit_tfidf.py            ← fit vectorizer, save artifacts
│   │   ├── compute_similarity.py   ← cosine sim, produce top-10
│   │   └── precompute_all.py       ← loop over all profile combos
│   ├── database/
│   │   ├── schema.sql              ← CREATE TABLE statements
│   │   └── load_rds.py             ← load recommendations.csv → RDS
│   └── dashboard/
│       └── app.py                  ← Streamlit app
├── sagemaker/
│   ├── processing_job.py           ← SageMaker Processing Job config
│   └── batch_transform.py          ← Batch Transform config
└── tests/
    └── test_similarity.py

## Key implementation decisions
- TF-IDF: unigrams + bigrams, max_features=10000, 
  min_df=2, sublinear_tf=True
- Cosine similarity: scipy.sparse or sklearn.metrics.pairwise
- Skin concerns to cover: acne, dryness, redness, sensitivity, 
  anti-aging, brightening, pores
- Profile combinations: 5 skin types × 7 concerns × 6 categories 
  = 210 combinations → 2,100 rows in recommendations table
- Similarity ranking: argsort descending, take top 10 — no 
  score threshold
- Vectorizer and product matrix saved as .pkl and .npz to S3 
  models/ before batch job runs

## Baseline model (context only — not the final system)
- A Logistic Regression classifier was run on TF-IDF features to validate text signal strength
- Results: AUC 0.9816, F1 negative class 0.7892
- This confirmed review text is the right input — it is NOT the deployed model
- The deployed system is the TF-IDF cosine similarity recommender 
  above, not the classifier

## What the Dashboard Displays
View 1 — Input panel: Three dropdowns: skin type, skin concern, product category. On submit → SQL query hits recommendations joined to products.
View 2 — Ranked results: A table or card grid showing rank, product name, brand, price, avg rating, and similarity score — so users see why a product ranked high.
View 3 — Profile coverage summary (optional): A heatmap or bar chart showing how many products exist per profile combination — useful for showing class coverage in your presentation.
