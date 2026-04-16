# Project Update 1: Model Training & Evaluation

**Course:** 46-887 Machine Learning for Business Applications

---

## 1. Models Trained and Evaluated

Our business problem is predicting whether a user will recommend a skincare product given their review text and profile metadata. We trained four classification models on 211,121 Sephora skincare reviews, predicting the binary target `is_recommended` (1 = recommended, 0 = not recommended).

**Features used:**

- **Review text** — TF-IDF vectorization of the review body (10,000 features, unigrams + bigrams, sublinear TF weighting)
- **Skin type** — One-hot encoded (combination, dry, normal, oily, unknown)
- **Product category** — One-hot encoded secondary category (Cleansers, Moisturizers, Treatments, Sunscreen, Eye Care, Lip Balms & Treatments)
- **Price** — Standardized `price_usd`

We deliberately **excluded `rating`** from the feature set because it is near-perfectly correlated with the target variable (99%+ of 5-star reviews are recommended; 95%+ of 1-star reviews are not). Including it would produce trivially high accuracy that does not generalize to new-product inference scenarios where no rating exists yet.

**Models trained** (replicating and extending the NLP classification approach from Source 3 in our proposal):

1. **Logistic Regression** with balanced class weights — strong, interpretable linear baseline for text classification
2. **Multinomial Naive Bayes** — classical probabilistic NLP classifier
3. **Random Forest** (200 trees, balanced class weights) — ensemble of decision trees capturing non-linear interactions
4. **XGBoost** (200 boosting rounds, scale_pos_weight adjusted) — gradient-boosted trees, typically strong on tabular + sparse features

## 2. Evaluation Metrics and Results

**Why these metrics:** Since `is_recommended` is heavily imbalanced (89.2% positive, 10.8% negative — an 8.2:1 ratio), accuracy alone is misleading. A model predicting "recommended" for every review achieves 89% accuracy while being useless. We therefore evaluate with:

- **Precision** — Of products the model recommends, how many are actually good fits? False positives (recommending bad-fit products) erode user trust and drive returns.
- **Recall** — Of truly good-fit products, how many does the model surface? False negatives mean missed revenue.
- **F1-score (negative class)** — Harmonic mean for the minority class; the hardest and most informative metric for imbalanced problems.
- **AUC-ROC** — Measures the model's ability to rank recommended above not-recommended across all thresholds. Critical for a recommendation system that ranks products.

**Results on held-out test set (20%, stratified):**

| Model | Accuracy | Precision | Recall | F1 (Positive) | F1 (Negative) | AUC-ROC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9462 | 0.9910 | 0.9482 | 0.9692 | 0.7892 | **0.9816** |
| Multinomial NB | 0.9524 | 0.9562 | 0.9921 | 0.9738 | 0.7401 | 0.9719 |
| Random Forest | 0.9293 | 0.9710 | 0.9491 | 0.9599 | 0.7014 | 0.9512 |
| XGBoost | 0.9316 | 0.9300 | 0.9984 | 0.9630 | 0.5473 | 0.9683 |

**Best model: Logistic Regression** — highest AUC-ROC (0.9816) and highest F1 on the negative class (0.7892), meaning it best identifies products that users would *not* recommend. Multinomial NB achieves the highest raw accuracy (0.9524) and recall (0.9921), but at the cost of lower precision and weaker negative-class F1. XGBoost, despite its expressiveness, overpredicts the positive class (recall 0.998 but F1-negative only 0.547), likely due to the extreme class imbalance.

**5-fold stratified cross-validation** on the training set for Logistic Regression confirms stable performance with low variance:

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.9475 | 0.0013 |
| F1 | 0.9699 | 0.0007 |
| AUC-ROC | 0.9819 | 0.0008 |
| Precision | 0.9909 | 0.0008 |
| Recall | 0.9498 | 0.0010 |

The narrow standard deviations across all five folds confirm that performance is robust and not an artifact of a particular train/test split.

**Feature importance** from the Logistic Regression coefficients reveals interpretable patterns: the strongest positive predictors (toward "recommended") include terms like *love*, *great*, *amazing*, *best*, *highly recommend*, while the strongest negative predictors include *return*, *broke out*, *waste*, *disappointed*, *irritated*. This confirms the model is learning meaningful sentiment and effect signals from the review text.

## 3. Best Practices Applied

- **Stratified train/test split (80/20):** Preserves the class ratio in both sets, preventing biased evaluation. Fixed random seed (42) ensures reproducibility.
- **Class imbalance handling:** Logistic Regression and Random Forest use `class_weight='balanced'` to upweight the minority class; XGBoost uses `scale_pos_weight` to achieve the same effect. We also report F1 on the negative class (not just overall accuracy) to ensure the minority is not ignored.
- **5-fold stratified cross-validation:** Applied to the best model to verify that test-set performance is not a lucky split. All metrics show standard deviations below 0.002, confirming stability.
- **Data leakage prevention:** `rating` excluded from features because it is a near-perfect proxy for `is_recommended`, which would inflate results artificially.
- **TF-IDF hyperparameters:** `max_features=10,000` and `min_df=5` prevent overfitting to rare terms; `sublinear_tf=True` dampens the effect of very high term counts; `max_df=0.95` removes terms appearing in >95% of documents.
- **Reproducible random seeds:** All models and splits use `random_state=42`.

## 4. Code Organization and Documentation

**Repository structure:**

```
skincare_app/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_model_training.ipynb   # Feature engineering, training, evaluation
├── models/
│   ├── best_model.joblib         # Saved Logistic Regression model
│   ├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer
│   ├── ohe_skin.joblib           # Skin type one-hot encoder
│   ├── ohe_category.joblib       # Product category one-hot encoder
│   ├── price_scaler.joblib       # Price standard scaler
│   ├── model_comparison.csv      # Metrics summary table
│   ├── confusion_matrices.png    # Confusion matrices for all models
│   ├── roc_curves.png            # ROC curves for all models
│   └── feature_importance.png    # Top feature coefficients
├── sephora_select_reviews.db     # Source data (SQLite)
└── Project_Update_1.md           # This report
```

**Version control:** The project is managed in a Git repository. Notebooks, model artifacts, and reports are tracked.

**Reproducibility:** To replicate our results, run the two notebooks in order (`01_eda.ipynb` then `02_model_training.ipynb`) from the `notebooks/` directory. Both notebooks load data directly from the SQLite database, require only standard Python ML libraries (pandas, scikit-learn, xgboost, matplotlib, seaborn), and produce all outputs (metrics, plots, saved models) automatically. All random seeds are fixed at 42.
