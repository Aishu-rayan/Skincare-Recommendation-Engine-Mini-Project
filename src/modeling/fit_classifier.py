"""Fit a logistic regression classifier on review-level is_recommended.

Reuses the previously fitted TF-IDF vectorizer (tfidf_vectorizer.pkl) to
transform review text, then trains LogisticRegression against the
is_recommended label. Writes classifier.pkl to the models directory
alongside the vectorizer.
"""
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_CSV = "../Cleaned_Reviews_Training_1.csv"


def load_labeled_reviews(csv_path: str, product_ids: set[str]) -> pd.DataFrame:
    """Load review-level rows with is_recommended for products in scope."""
    log.info("Reading %s", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=["product_id", "review_text", "is_recommended", "skin_type"],
    )
    df = df[df["product_id"].isin(product_ids)]
    df = df.dropna(subset=["review_text", "is_recommended"])
    df = df[df["review_text"].str.strip() != ""]
    df["is_recommended"] = df["is_recommended"].astype(int)
    log.info(
        "Labeled rows: %d  (pos=%d, neg=%d)",
        len(df),
        int((df["is_recommended"] == 1).sum()),
        int((df["is_recommended"] == 0).sum()),
    )
    return df


def main() -> None:
    """CLI entry: fit LR, save classifier.pkl, report held-out metrics."""
    parser = argparse.ArgumentParser(description="Fit LR on review-level is_recommended.")
    add_env_arg(parser)
    parser.add_argument(
        "--reviews-csv",
        default=os.environ.get("REVIEWS_CSV", DEFAULT_CSV),
        help="Path to the review-level CSV (local mode only).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode wired in Phase 5")

    models_dir = Path(paths.models) / "recommender"
    with open(models_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    agg = pd.read_parquet(Path(paths.processed) / "aggregated_reviews.parquet")
    product_ids = set(agg["product_id"].astype(str))

    df = load_labeled_reviews(args.reviews_csv, product_ids)

    X = vectorizer.transform(df["review_text"].astype(str))
    y = df["is_recommended"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    log.info("Train shape %s  Test shape %s", X_train.shape, X_test.shape)

    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    f1_neg = f1_score(y_test, preds, pos_label=0)
    log.info("Held-out AUC: %.4f", auc)
    log.info("Held-out F1 (negative class): %.4f", f1_neg)
    log.info("\n%s", classification_report(y_test, preds, digits=4))

    out = models_dir / "classifier.pkl"
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
