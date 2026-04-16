"""Generate the 210 skin-profile queries used by the recommender.

Each profile is a (skin_type, skin_concern, category) triple mapped to a
natural-language query string that will be transformed by the same
TF-IDF vectorizer used on product corpora.
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

SKIN_TYPES = ["dry", "oily", "normal", "combination", "sensitive"]
SKIN_CONCERNS = [
    "acne",
    "dryness",
    "redness",
    "sensitivity",
    "anti-aging",
    "brightening",
    "pores",
]
CATEGORIES = [
    "Moisturizers",
    "Treatments",
    "Cleansers",
    "Sunscreen",
    "Eye Care",
    "Lip Balms & Treatments",
]

CONCERN_EXPANSION = {
    "acne": "acne breakouts pimples blemishes clogged pores",
    "dryness": "dry skin hydration moisturizing flaky tight",
    "redness": "redness irritation calming soothing rosacea",
    "sensitivity": "sensitive skin gentle fragrance free hypoallergenic",
    "anti-aging": "wrinkles fine lines firming anti aging collagen",
    "brightening": "brightening dark spots even tone glow radiance",
    "pores": "large pores minimizing refining pore",
}


def build_query(skin_type: str, concern: str, category: str) -> str:
    """Return the natural-language query string for one profile."""
    return f"{skin_type} skin {CONCERN_EXPANSION[concern]} {category.lower()}"


def build_profiles() -> pd.DataFrame:
    """Return a DataFrame of all skin_type × concern × category combos."""
    rows = []
    for st, sc, cat in itertools.product(SKIN_TYPES, SKIN_CONCERNS, CATEGORIES):
        profile_id = f"{st}_{sc}_{cat}".lower().replace(" ", "").replace("&", "and")
        rows.append(
            {
                "profile_id": profile_id,
                "skin_type": st,
                "skin_concern": sc,
                "category": cat,
                "query": build_query(st, sc, cat),
            }
        )
    df = pd.DataFrame(rows)
    log.info("Generated %d profiles", len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all skin-profile queries.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode wired in Phase 5")

    df = build_profiles()
    out = Path(paths.processed) / "profiles.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
