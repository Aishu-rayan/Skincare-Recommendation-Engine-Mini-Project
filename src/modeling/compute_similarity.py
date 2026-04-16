"""Cosine similarity helpers used by precompute_all.

The products table is filtered to one category before similarity is
computed, so each profile only ranks products from its own category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def top_k_for_query(
    query: str,
    vectorizer: TfidfVectorizer,
    product_matrix: sparse.csr_matrix,
    product_ids: pd.Series,
    eligible_mask: np.ndarray,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k (product_id, similarity) for one query string.

    eligible_mask is a boolean array over product_matrix rows restricting
    ranking to e.g. a single product category.
    """
    query_vec = vectorizer.transform([query])
    eligible_idx = np.where(eligible_mask)[0]
    if eligible_idx.size == 0:
        return []
    sims = cosine_similarity(query_vec, product_matrix[eligible_idx]).ravel()
    order = np.argsort(-sims)[:k]
    return [(product_ids.iloc[eligible_idx[i]], float(sims[i])) for i in order]
