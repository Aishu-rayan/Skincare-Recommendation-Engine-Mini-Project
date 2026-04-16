"""Precompute top-10 recommendations for every skin-profile query.

Ranking blends TF-IDF cosine similarity with a logistic-regression
is_recommended log-odds score per (product_id, skin_type). Both signals
are min-max normalized within each profile's eligible set before the
weighted blend so neither dominates just because of its native range.

Output: recommendations.csv (one row per (profile_id, rank)) matching the
RDS recommendations table schema. similarity_score carries the blended
score used for ranking.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)
TOP_K = 10


def minmax_normalize(a: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]; return zeros if range is degenerate."""
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)


def main() -> None:
    """CLI entry: blend cosine sim with classifier prob, write top-10 per profile."""
    parser = argparse.ArgumentParser(description="Precompute top-10 per profile.")
    add_env_arg(parser)
    parser.add_argument("--w-sim", type=float, default=0.7, help="Weight on normalized cosine sim.")
    parser.add_argument("--w-clf", type=float, default=0.3, help="Weight on normalized classifier score.")
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
    scores = pd.read_parquet(processed / "classifier_scores.parquet")

    cat_lookup = products.set_index("product_id")["category"]
    cat_per_row = product_ids.map(cat_lookup).values

    score_lookup = {
        (pid, st): float(s)
        for pid, st, s in zip(scores["product_id"], scores["skin_type"], scores["score"])
    }
    overall_lookup = {
        pid: float(s)
        for pid, st, s in zip(scores["product_id"], scores["skin_type"], scores["score"])
        if st == "overall"
    }

    rows = []
    for _, prof in profiles.iterrows():
        mask = cat_per_row == prof["category"]
        eligible_idx = np.where(mask)[0]
        if eligible_idx.size == 0:
            continue

        query_vec = vectorizer.transform([prof["query"]])
        sims = cosine_similarity(query_vec, product_matrix[eligible_idx]).ravel()
        norm_sims = minmax_normalize(sims)

        eligible_pids = product_ids.iloc[eligible_idx].values
        clf_raw = np.array(
            [
                score_lookup.get((pid, prof["skin_type"]), overall_lookup.get(pid, 0.0))
                for pid in eligible_pids
            ]
        )
        norm_clf = minmax_normalize(clf_raw)

        combined = args.w_sim * norm_sims + args.w_clf * norm_clf
        order = np.argsort(-combined)[:TOP_K]
        for rank, i in enumerate(order, start=1):
            rows.append(
                {
                    "profile_id": prof["profile_id"],
                    "skin_type": prof["skin_type"],
                    "skin_concern": prof["skin_concern"],
                    "category": prof["category"],
                    "rank": rank,
                    "product_id": eligible_pids[i],
                    "similarity_score": float(combined[i]),
                }
            )

    recs = pd.DataFrame(rows)
    log.info("Computed %d recommendation rows across %d profiles",
             len(recs), len(profiles))

    out_dir = processed / "precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "recommendations.csv"
    recs.to_csv(out, index=False)
    log.info("Wrote %s (w_sim=%.2f, w_clf=%.2f)", out, args.w_sim, args.w_clf)


if __name__ == "__main__":
    main()
