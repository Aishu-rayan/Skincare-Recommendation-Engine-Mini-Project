"""Aggregate all review text per product_id into a single corpus row.

Reads the Sephora SQLite DB, filters to Skincare, groups reviews by
product_id, and writes two parquet files:
    aggregated_reviews.parquet  — product_id, corpus, review_count
    products.parquet            — product master for the RDS products table
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_reviews(sqlite_path: str) -> pd.DataFrame:
    """Return all skincare reviews with the columns needed downstream."""
    log.info("Opening SQLite at %s", sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        df = pd.read_sql(
            """
            SELECT product_id, product_name, brand_name, secondary_category,
                   review_text, rating, price_usd
            FROM select_customer_reviews
            WHERE primary_category = 'Skincare'
              AND review_text IS NOT NULL
              AND TRIM(review_text) <> ''
            """,
            conn,
        )
    log.info("Loaded %d review rows", len(df))
    return df


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corpus_df, products_df) grouped by product_id."""
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

    corpus = (
        df.groupby("product_id", as_index=False)
        .agg(
            corpus=("review_text", lambda s: " ".join(s.astype(str))),
            review_count=("review_text", "size"),
        )
    )
    log.info("Aggregated to %d unique products", len(corpus))

    products = (
        df.groupby("product_id", as_index=False)
        .agg(
            product_name=("product_name", "first"),
            brand_name=("brand_name", "first"),
            category=("secondary_category", "first"),
            price_usd=("price_usd", "mean"),
            avg_rating=("rating", "mean"),
            review_count=("review_text", "size"),
        )
    )
    return corpus, products


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate skincare reviews per product.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode for aggregate_reviews is wired in Phase 5")

    df = load_reviews(paths.sqlite_db)
    corpus, products = aggregate(df)

    out = Path(paths.processed)
    out.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out / "aggregated_reviews.parquet", index=False)
    products.to_parquet(out / "products.parquet", index=False)
    log.info("Wrote %s and %s", out / "aggregated_reviews.parquet", out / "products.parquet")


if __name__ == "__main__":
    main()
