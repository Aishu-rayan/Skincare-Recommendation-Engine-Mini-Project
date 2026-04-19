"""Filter source reviews to the top-5 most helpful per (product, skin_type).

Keeps reviews for products that survived into the aggregated product set.
Ranks within each (product_id, skin_type) bucket by positive-feedback
count (then rating, then length), takes top 5, and also emits an
'overall' bucket using all reviews for the product. Output is a parquet
file small enough to push into RDS as a `reviews` table.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_CSV = "../Cleaned_Reviews_Training_1.csv"
SKIN_TYPES_IN_DATA = ["combination", "dry", "normal", "oily"]
TOP_K_PER_BUCKET = 5
MIN_REVIEW_CHARS = 80
MAX_REVIEW_CHARS = 1000


def load_reviews(csv_path: str, product_ids: set[str]) -> pd.DataFrame:
    """Load + clean review-level rows restricted to in-scope products."""
    log.info("Reading %s", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=[
            "product_id",
            "review_text",
            "review_title",
            "skin_type",
            "rating",
            "helpfulness",
            "total_pos_feedback_count",
            "total_feedback_count",
        ],
    )
    df = df[df["product_id"].isin(product_ids)]
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() >= MIN_REVIEW_CHARS]
    df["review_text"] = df["review_text"].str[:MAX_REVIEW_CHARS]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["helpfulness"] = pd.to_numeric(df["helpfulness"], errors="coerce").fillna(0.0)
    df["total_pos_feedback_count"] = pd.to_numeric(
        df["total_pos_feedback_count"], errors="coerce"
    ).fillna(0).astype(int)
    df["review_length"] = df["review_text"].str.len()
    log.info("Eligible review rows: %d", len(df))
    return df


def top_k_per_bucket(df: pd.DataFrame, bucket_cols: list[str]) -> pd.DataFrame:
    """Rank within each bucket and keep the top K reviews."""
    ranked = df.sort_values(
        by=[*bucket_cols, "total_pos_feedback_count", "rating", "review_length"],
        ascending=[True] * len(bucket_cols) + [False, False, False],
    )
    return ranked.groupby(bucket_cols, as_index=False).head(TOP_K_PER_BUCKET)


def build_review_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form reviews with buckets for each known skin_type + overall."""
    per_type = df[df["skin_type"].isin(SKIN_TYPES_IN_DATA)].copy()
    per_type = top_k_per_bucket(per_type, ["product_id", "skin_type"])

    overall = df.copy()
    overall["skin_type"] = "overall"
    overall = top_k_per_bucket(overall, ["product_id", "skin_type"])

    out = pd.concat([per_type, overall], ignore_index=True)
    return out[
        [
            "product_id",
            "skin_type",
            "rating",
            "helpfulness",
            "total_pos_feedback_count",
            "review_title",
            "review_text",
        ]
    ].rename(columns={"total_pos_feedback_count": "pos_feedback"})


def main() -> None:
    """CLI entry: write reviews.parquet with top-5 per (product, skin_type)."""
    parser = argparse.ArgumentParser(
        description="Filter reviews to top-5 helpful per (product, skin_type)."
    )
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

    agg = pd.read_parquet(Path(paths.processed) / "aggregated_reviews.parquet")
    product_ids = set(agg["product_id"].astype(str))

    df = load_reviews(args.reviews_csv, product_ids)
    out_df = build_review_table(df)

    out_path = Path(paths.processed) / "reviews.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info(
        "Wrote %s (rows=%d, unique products=%d)",
        out_path,
        len(out_df),
        out_df["product_id"].nunique(),
    )


if __name__ == "__main__":
    main()
