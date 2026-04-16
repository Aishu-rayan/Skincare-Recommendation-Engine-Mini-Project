I have a sephora dataset @sephora_select_reviews from Kaggle. This is already a SQL DB file. I want to build a mini project - This is a skincare product recommendation engine based on product user reviews
I want to host the final output (dashboard using streamlit) on AWS platform and also use all of AWS capabilities like S3, EC2, RDS, Sagemaker and other relevant tools

# This is the basic logic of the overall flow
## "We train a TF-IDF vectorizer on review text to build per-product feature vectors, then recommend products via cosine similarity to a user's skin concern query."
## Model: Scikit-learn TF-IDF + cosine similarity (no LLM needed)
## Output stored in RDS: Precomputed top-10 recommendations per skin type × concern combination
## Dashboard: User inputs skin type + concern → ranked product list
## AWS: S3 → SageMaker → RDS (precomputed lookup table) → Streamlit

# Clear model statement: We build TF-IDF vectors from aggregated review text per product, then recommend products via cosine similarity to a query vector constructed from a user's selected skin type and skin concern — precomputing the top-10 results for every skin profile combination and storing them in RDS for fast dashboard retrieval.