"""Score every (product_id, skin_type) with the trained LR classifier.

For each product, concatenates all reviews filtered by skin_type and by
the full set of reviews ("overall"), transforms with the existing TF-IDF
vectorizer, and records classifier.decision_function(...). Log-odds have
a wider dynamic range than saturated predict_proba values, which matters
because aggregating many positive-leaning reviews pushes probs toward 1.

Output: processed/classifier_scores.parquet
    columns: product_id, skin_type, score
    skin_type in {combination, dry, normal, oily, overall}
"""
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_CSV = "../Cleaned_Reviews_Training_1.csv"
SKIN_TYPES_IN_DATA = ["combination", "dry", "normal", "oily"]


def build_aggregated_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form (product_id, skin_type, text) including 'overall'."""
    per_type = (
        df.dropna(subset=["skin_type"])
        .groupby(["product_id", "skin_type"], as_index=False)
        .agg(text=("review_text", lambda s: " ".join(s.astype(str))))
    )
    per_type = per_type[per_type["skin_type"].isin(SKIN_TYPES_IN_DATA)]

    overall = (
        df.groupby("product_id", as_index=False)
        .agg(text=("review_text", lambda s: " ".join(s.astype(str))))
    )
    overall["skin_type"] = "overall"

    out = pd.concat([per_type, overall[["product_id", "skin_type", "text"]]], ignore_index=True)
    log.info("Aggregated texts: %d rows", len(out))
    return out


def main() -> None:
    """CLI entry: score all (product, skin_type) combos and save parquet."""
    parser = argparse.ArgumentParser(description="Score products per skin_type with LR.")
    add_env_arg(parser)
    parser.add_argument(
        "--reviews-csv",
        default=os.environ.get("REVIEWS_CSV", DEFAULT_CSV),
        help="Path to the review-level CSV (local mode only).",
    )
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode wired in Phase 5")

    models_dir = Path(paths.models) / "recommender"
    with open(models_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(models_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    agg = pd.read_parquet(Path(paths.processed) / "aggregated_reviews.parquet")
    product_ids = set(agg["product_id"].astype(str))

    log.info("Reading %s", args.reviews_csv)
    df = pd.read_csv(
        args.reviews_csv,
        usecols=["product_id", "review_text", "skin_type"],
    )
    df = df[df["product_id"].isin(product_ids)]
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]
    log.info("Review rows in scope: %d", len(df))

    texts = build_aggregated_texts(df)
    X = vectorizer.transform(texts["text"].astype(str))
    raw_scores = clf.decision_function(X)
    scores = pd.DataFrame(
        {
            "product_id": texts["product_id"].values,
            "skin_type": texts["skin_type"].values,
            "score": raw_scores,
        }
    )

    out = Path(paths.processed) / "classifier_scores.parquet"
    scores.to_parquet(out, index=False)
    log.info(
        "Wrote %s (rows=%d, score mean=%.4f, min=%.4f, max=%.4f)",
        out,
        len(scores),
        float(scores["score"].mean()),
        float(scores["score"].min()),
        float(scores["score"].max()),
    )


if __name__ == "__main__":
    main()
