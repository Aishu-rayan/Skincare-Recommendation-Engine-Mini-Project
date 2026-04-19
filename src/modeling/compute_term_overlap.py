"""Compute the top TF-IDF terms driving each (profile, product) match.

For every (profile_id, product_id) pair already selected into the top-10,
take the element-wise product of the profile's query vector and the
product's TF-IDF row. The terms with highest joint weight are the ones
that drove the cosine similarity — i.e. the review-vocabulary overlap
between what the user asked for and what reviewers talked about.

Output: processed/term_overlap.parquet
    columns: profile_id, product_id, top_terms (comma-separated string)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

TOP_TERMS_PER_PAIR = 5


def main() -> None:
    """CLI entry: write term_overlap.parquet keyed on (profile_id, product_id)."""
    parser = argparse.ArgumentParser(description="Compute top overlap terms per (profile, product).")
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
    pid_to_row = {pid: i for i, pid in enumerate(product_ids)}

    profiles = pd.read_parquet(processed / "profiles.parquet")
    recs = pd.read_csv(processed / "precomputed" / "recommendations.csv")

    feature_names = vectorizer.get_feature_names_out()
    rows_out = []
    for _, prof in profiles.iterrows():
        q_vec = vectorizer.transform([prof["query"]])
        prof_recs = recs[recs["profile_id"] == prof["profile_id"]]
        if prof_recs.empty:
            continue
        for _, rec in prof_recs.iterrows():
            pid = rec["product_id"]
            if pid not in pid_to_row:
                continue
            p_vec = product_matrix[pid_to_row[pid]]
            joint = q_vec.multiply(p_vec).toarray().ravel()
            top_idx = np.argsort(-joint)[:TOP_TERMS_PER_PAIR]
            top = [feature_names[i] for i in top_idx if joint[i] > 0]
            rows_out.append(
                {
                    "profile_id": prof["profile_id"],
                    "product_id": pid,
                    "top_terms": ", ".join(top),
                }
            )

    out = pd.DataFrame(rows_out)
    out_path = processed / "term_overlap.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Wrote %s (rows=%d)", out_path, len(out))


if __name__ == "__main__":
    main()
