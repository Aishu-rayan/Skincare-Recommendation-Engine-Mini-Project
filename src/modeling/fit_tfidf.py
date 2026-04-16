"""Fit a TF-IDF vectorizer on the aggregated review corpus.

Writes two artifacts to the models directory:
    tfidf_vectorizer.pkl   — fitted sklearn TfidfVectorizer
    product_vectors.npz    — sparse CSR matrix aligned with product_ids.csv
    product_ids.csv        — row-order lookup for the sparse matrix
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)


def fit(corpus: pd.DataFrame) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Fit TF-IDF and return (vectorizer, product matrix)."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(corpus["corpus"].fillna(""))
    log.info("TF-IDF matrix shape: %s", matrix.shape)
    return vectorizer, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TF-IDF on aggregated reviews.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode wired in Phase 5")

    corpus_path = Path(paths.processed) / "aggregated_reviews.parquet"
    corpus = pd.read_parquet(corpus_path)
    log.info("Loaded %d product corpora from %s", len(corpus), corpus_path)

    vectorizer, matrix = fit(corpus)

    models_dir = Path(paths.models) / "recommender"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(models_dir / "product_vectors.npz", matrix)
    corpus[["product_id"]].to_csv(models_dir / "product_ids.csv", index=False)
    log.info("Saved vectorizer, matrix, and product_ids to %s", models_dir)


if __name__ == "__main__":
    main()
