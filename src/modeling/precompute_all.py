"""Precompute top-10 recommendations for every skin-profile query.

Output: recommendations.csv (one row per (profile_id, rank)) matching the
RDS recommendations table schema.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.modeling.compute_similarity import top_k_for_query
from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)
TOP_K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute top-10 per profile.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode wired in Phase 5")

    processed = Path(paths.processed)
    models_dir = Path(paths.models) / "recommender"

    with open(models_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    product_matrix = sparse.load_npz(models_dir / "product_vectors.npz")
    product_ids = pd.read_csv(models_dir / "product_ids.csv")["product_id"]

    products = pd.read_parquet(processed / "products.parquet")
    profiles = pd.read_parquet(processed / "profiles.parquet")

    cat_lookup = products.set_index("product_id")["category"]
    cat_per_row = product_ids.map(cat_lookup).values

    rows = []
    for _, prof in profiles.iterrows():
        mask = cat_per_row == prof["category"]
        top = top_k_for_query(
            prof["query"], vectorizer, product_matrix, product_ids, mask, k=TOP_K
        )
        for rank, (pid, score) in enumerate(top, start=1):
            rows.append(
                {
                    "profile_id": prof["profile_id"],
                    "skin_type": prof["skin_type"],
                    "skin_concern": prof["skin_concern"],
                    "category": prof["category"],
                    "rank": rank,
                    "product_id": pid,
                    "similarity_score": score,
                }
            )

    recs = pd.DataFrame(rows)
    log.info("Computed %d recommendation rows across %d profiles",
             len(recs), len(profiles))

    out_dir = processed / "precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "recommendations.csv"
    recs.to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
