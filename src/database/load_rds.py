"""Load products, recommendations, summaries, explanations, and reviews into RDS.

Creates the schema (dropping any existing tables) and then bulk-inserts
rows from the parquet/CSV artifacts produced by the modeling phase.
"""
import argparse
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.utils.config import add_env_arg, rds_config, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def run_schema(conn) -> None:
    """Apply schema.sql to the connected database."""
    sql = SCHEMA_PATH.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    log.info("Schema applied")


def load_products(conn, products: pd.DataFrame) -> None:
    """Insert the products master rows."""
    rows = [
        (
            r.product_id,
            r.product_name,
            r.brand_name,
            r.category,
            None if pd.isna(r.price_usd) else float(r.price_usd),
            None if pd.isna(r.avg_rating) else float(r.avg_rating),
            int(r.review_count),
        )
        for r in products.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO products VALUES %s", rows)
    conn.commit()
    log.info("Inserted %d products", len(rows))


def load_recs(conn, recs: pd.DataFrame) -> None:
    """Insert the precomputed recommendation rows with budget/quality flags."""
    rows = [
        (
            r.profile_id,
            r.skin_type,
            r.skin_concern,
            r.category,
            int(r.rank),
            r.product_id,
            float(r.similarity_score),
            bool(r.is_budget_pick),
            bool(r.is_quality_pick),
        )
        for r in recs.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO recommendations VALUES %s", rows)
    conn.commit()
    log.info("Inserted %d recommendation rows", len(rows))


def load_summaries(conn, summaries: pd.DataFrame) -> None:
    """Insert per-profile narrative summaries (top picks + budget/quality picks)."""
    rows = [
        (r.profile_id, r.summary_top, r.summary_picks or None)
        for r in summaries.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO profile_summaries VALUES %s", rows)
    conn.commit()
    log.info("Inserted %d profile summaries", len(rows))


def load_card_explanations(conn, blurbs: pd.DataFrame) -> None:
    """Insert per-card 'why we picked this' blurbs."""
    rows = [
        (r.profile_id, r.product_id, r.blurb)
        for r in blurbs.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO card_explanations VALUES %s", rows)
    conn.commit()
    log.info("Inserted %d card explanations", len(rows))


def load_reviews(conn, reviews: pd.DataFrame) -> None:
    """Insert top-5 helpful reviews per (product_id, skin_type)."""
    rows = [
        (
            r.product_id,
            r.skin_type,
            None if pd.isna(r.rating) else float(r.rating),
            None if pd.isna(r.helpfulness) else float(r.helpfulness),
            int(r.pos_feedback),
            None if pd.isna(r.review_title) else str(r.review_title),
            str(r.review_text),
        )
        for r in reviews.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO reviews VALUES %s", rows, page_size=500)
    conn.commit()
    log.info("Inserted %d reviews", len(rows))


def main() -> None:
    """CLI entry: rebuild schema and load all tables in one transaction flow."""
    parser = argparse.ArgumentParser(description="Load all artifacts into RDS.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    processed = Path(paths.processed)

    products = pd.read_parquet(processed / "products.parquet")
    recs = pd.read_csv(processed / "precomputed" / "recommendations.csv")
    summaries = pd.read_parquet(processed / "profile_summaries.parquet")
    blurbs = pd.read_parquet(processed / "card_explanations.parquet")
    reviews = pd.read_parquet(processed / "reviews.parquet")

    cfg = rds_config()
    log.info("Connecting to RDS at %s", cfg["host"])
    conn = psycopg2.connect(**cfg)
    try:
        run_schema(conn)
        load_products(conn, products)
        load_recs(conn, recs)
        load_summaries(conn, summaries)
        load_card_explanations(conn, blurbs)
        load_reviews(conn, reviews)
    finally:
        conn.close()
    log.info("Done")


if __name__ == "__main__":
    main()
