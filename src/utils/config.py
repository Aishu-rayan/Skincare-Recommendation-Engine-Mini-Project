"""Environment-aware path and credential resolution.

Every CLI script takes --env {local,aws}. In local mode we read from
./data/ and the SQLite DB at the project root; in aws mode we read/write
to S3 under $S3_BUCKET.
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # skincare-recommender/
REPO_ROOT = PROJECT_ROOT.parent                     # skincare_app/


@dataclass
class Paths:
    """Resolved input/output locations for the current --env."""
    env: str
    sqlite_db: str
    raw: str
    processed: str
    models: str
    precomputed: str
    logs: str


def add_env_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the standard --env flag to a CLI parser."""
    parser.add_argument(
        "--env",
        choices=["local", "aws"],
        default="local",
        help="Run locally against ./data or against S3 in AWS.",
    )


def resolve_paths(env: str) -> Paths:
    """Return Paths for local filesystem or s3:// prefixes based on --env."""
    if env == "local":
        data = PROJECT_ROOT / "data"
        sqlite_path = os.environ.get(
            "SQLITE_DB_PATH", str(REPO_ROOT / "sephora_select_reviews.db")
        )
        return Paths(
            env="local",
            sqlite_db=sqlite_path,
            raw=str(data / "raw"),
            processed=str(data / "processed"),
            models=str(PROJECT_ROOT.parent / "models"),
            precomputed=str(data / "processed" / "precomputed"),
            logs=str(PROJECT_ROOT / "logs"),
        )

    bucket = os.environ["S3_BUCKET"]
    base = f"s3://{bucket}"
    return Paths(
        env="aws",
        sqlite_db=f"{base}/raw/sephora_select_reviews.db",
        raw=f"{base}/raw",
        processed=f"{base}/processed",
        models=f"{base}/models",
        precomputed=f"{base}/precomputed",
        logs=f"{base}/logs",
    )


def rds_config() -> dict:
    """Read RDS connection settings from environment variables."""
    return {
        "host": os.environ.get("RDS_HOST"),
        "port": int(os.environ.get("RDS_PORT", "5432")),
        "dbname": os.environ.get("RDS_DB"),
        "user": os.environ.get("RDS_USER"),
        "password": os.environ.get("RDS_PASSWORD"),
    }
