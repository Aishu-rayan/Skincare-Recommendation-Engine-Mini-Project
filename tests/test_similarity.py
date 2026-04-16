"""Smoke tests for cosine similarity ranking."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.modeling.compute_similarity import top_k_for_query


def _fixture():
    corpus = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3", "p4"],
            "corpus": [
                "hydrating moisturizer dry skin",
                "acne breakout treatment salicylic",
                "brightening vitamin c serum",
                "gentle cleanser sensitive skin",
            ],
        }
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(corpus["corpus"])
    return vec, matrix, corpus["product_id"]


def test_top_k_returns_expected_product():
    vec, matrix, ids = _fixture()
    mask = np.ones(matrix.shape[0], dtype=bool)
    top = top_k_for_query("acne breakouts pimples", vec, matrix, ids, mask, k=2)
    assert top[0][0] == "p2"
    assert top[0][1] > top[1][1]


def test_top_k_respects_eligible_mask():
    vec, matrix, ids = _fixture()
    mask = np.array([True, False, True, False])
    top = top_k_for_query("acne", vec, matrix, ids, mask, k=4)
    returned_ids = {pid for pid, _ in top}
    assert returned_ids <= {"p1", "p3"}


def test_top_k_empty_mask_returns_empty():
    vec, matrix, ids = _fixture()
    mask = np.zeros(matrix.shape[0], dtype=bool)
    assert top_k_for_query("acne", vec, matrix, ids, mask) == []
