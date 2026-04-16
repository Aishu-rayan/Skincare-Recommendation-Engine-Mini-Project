"""Load products + recommendations into RDS PostgreSQL.

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
        execute_values(
            cur,
            "INSERT INTO products VALUES %s",
            rows,
        )
    conn.commit()
    log.info("Inserted %d products", len(rows))


def load_recs(conn, recs: pd.DataFrame) -> None:
    """Insert the precomputed recommendation rows."""
    rows = [
        (
            r.profile_id,
            r.skin_type,
            r.skin_concern,
            r.category,
            int(r.rank),
            r.product_id,
            float(r.similarity_score),
        )
        for r in recs.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO recommendations VALUES %s",
            rows,
        )
    conn.commit()
    log.info("Inserted %d recommendation rows", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load products + recommendations into RDS.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    processed = Path(paths.processed)
    products = pd.read_parquet(processed / "products.parquet")
    recs = pd.read_csv(processed / "precomputed" / "recommendations.csv")

    cfg = rds_config()
    log.info("Connecting to RDS at %s", cfg["host"])
    conn = psycopg2.connect(**cfg)
    try:
        run_schema(conn)
        load_products(conn, products)
        load_recs(conn, recs)
    finally:
        conn.close()
    log.info("Done")


if __name__ == "__main__":
    main()
