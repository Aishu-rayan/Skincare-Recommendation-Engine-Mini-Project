"""Build benefit-oriented summaries for each skin profile and per-card blurbs.

Renders two artifacts purely from data — no LLM. Quality comes from a
curated term→benefit lookup so outputs read as advantages ("relief from
breakouts", "non-greasy finish") rather than keyword lists.

Outputs:
    processed/profile_summaries.parquet  — profile_id, summary
    processed/card_explanations.parquet  — profile_id, product_id, blurb

Also tags each top-10 row with is_budget_pick / is_quality_pick in
recommendations.csv so the dashboard can surface badges.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import add_env_arg, resolve_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

# Map lower-cased TF-IDF terms (or substrings) to benefit phrases shown
# to the end user. Order matters — first match wins per term.
BENEFIT_MAP: list[tuple[tuple[str, ...], str]] = [
    (("acne breakouts", "breakouts", "pimples", "blemishes", "acne", "cystic"),
     "fewer breakouts and clearer skin"),
    (("clogged pores", "pores", "pore"),
     "unclogged, minimized pores"),
    (("dry skin", "dryness", "flaky", "tight"),
     "deep, lasting hydration"),
    (("hydration", "hydrating", "moisturizing", "moisture"),
     "long-lasting hydration without heaviness"),
    (("oily", "greasy", "sebum", "shine"),
     "a matte, non-greasy finish"),
    (("redness", "irritation", "rosacea", "calming", "soothing"),
     "calm, less reddened skin"),
    (("sensitive", "gentle", "fragrance free", "hypoallergenic"),
     "gentle performance on sensitive skin"),
    (("wrinkles", "fine lines", "firming", "anti aging", "anti-aging", "collagen"),
     "smoother texture and softened fine lines"),
    (("dark spots", "brightening", "even tone", "glow", "radiance"),
     "brighter, more even tone"),
    (("spf", "sunscreen", "sun protection", "uv"),
     "broad-spectrum sun protection"),
    (("eye", "dark circles", "puffiness"),
     "refreshed, less-tired eyes"),
    (("lips", "chapped", "balm"),
     "softer, more conditioned lips"),
]

# Generic terms we should never surface as benefits directly.
STOP_TERMS = {
    "skin", "skin acne", "product", "products", "moisturizers", "moisturizer",
    "treatments", "cleansers", "sunscreen", "eye care", "lip balms",
    "dry", "oily", "normal", "combination",
}


def extract_benefits(top_terms: str, max_benefits: int = 3) -> list[str]:
    """Translate raw overlap terms into a de-duplicated list of benefit phrases."""
    terms = [t.strip().lower() for t in top_terms.split(",") if t.strip()]
    terms = [t for t in terms if t not in STOP_TERMS]

    picked: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for patterns, benefit in BENEFIT_MAP:
            if any(p in term for p in patterns) and benefit not in seen:
                picked.append(benefit)
                seen.add(benefit)
                break
        if len(picked) >= max_benefits:
            break
    return picked


def list_phrase(items: list[str]) -> str:
    """Join ['a', 'b', 'c'] as 'a, b, and c'; handles 1/2/3+ gracefully."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def display_concern(concern: str) -> str:
    """Turn a raw concern slug into natural-language phrasing."""
    mapping = {
        "acne": "acne concerns",
        "dryness": "dryness",
        "redness": "redness",
        "sensitivity": "sensitivity",
        "anti-aging": "anti-aging needs",
        "brightening": "brightening goals",
        "pores": "visible pores",
    }
    return mapping.get(concern, concern)


def pick_budget(top10: pd.DataFrame) -> pd.Series | None:
    """Return the cheapest product in top-10 (skip if price missing)."""
    priced = top10.dropna(subset=["price_usd"])
    if priced.empty:
        return None
    return priced.sort_values("price_usd").iloc[0]


def pick_quality(top10: pd.DataFrame) -> pd.Series | None:
    """Return the highest-rated product with >=50 reviews (fall back if none qualify)."""
    candidates = top10[top10["review_count"].fillna(0) >= 50]
    if candidates.empty:
        candidates = top10
    return candidates.sort_values(
        ["avg_rating", "review_count"], ascending=[False, False]
    ).iloc[0]


def render_profile_summary(
    profile: pd.Series,
    top10: pd.DataFrame,
    overlap: pd.DataFrame,
) -> tuple[str, str]:
    """Return (top_picks_paragraph, budget_quality_paragraph) for one profile."""
    skin = profile["skin_type"]
    concern = display_concern(profile["skin_concern"])
    category = profile["category"].lower()
    top10 = top10.sort_values("rank")

    top2 = top10.head(2)
    combined_terms = []
    for _, row in top2.iterrows():
        match = overlap[
            (overlap["profile_id"] == profile["profile_id"])
            & (overlap["product_id"] == row["product_id"])
        ]
        if not match.empty:
            combined_terms.extend(extract_benefits(match.iloc[0]["top_terms"], 3))
    dedup_benefits = list(dict.fromkeys(combined_terms))[:3]
    benefit_clause = (
        f"reviewers consistently report {list_phrase(dedup_benefits)}"
        if dedup_benefits
        else "reviewers with similar skin profiles gave them strong endorsements"
    )

    names = [f"**{r['brand_name']} {r['product_name']}**" for _, r in top2.iterrows()]
    top_phrase = list_phrase(names)
    summary_top = (
        f"For **{skin} skin** with {concern} in the {category} category, "
        f"we recommend {top_phrase} as top picks — {benefit_clause}."
    )

    budget = pick_budget(top10)
    budget_sentence = ""
    if budget is not None and budget["product_id"] not in top2["product_id"].values:
        budget_match = overlap[
            (overlap["profile_id"] == profile["profile_id"])
            & (overlap["product_id"] == budget["product_id"])
        ]
        budget_benefits = (
            extract_benefits(budget_match.iloc[0]["top_terms"], 2)
            if not budget_match.empty
            else []
        )
        b_benefit = (
            list_phrase(budget_benefits)
            if budget_benefits
            else "similar benefits at a lower price"
        )
        budget_sentence = (
            f"For a budget-friendly option, **{budget['brand_name']} "
            f"{budget['product_name']}** at ${budget['price_usd']:.2f} "
            f"offers {b_benefit}."
        )

    quality = pick_quality(top10)
    quality_sentence = ""
    if (
        quality is not None
        and quality["product_id"] not in top2["product_id"].values
        and (budget is None or quality["product_id"] != budget["product_id"])
        and not pd.isna(quality["avg_rating"])
    ):
        quality_sentence = (
            f"If you want the highest-rated pick, **{quality['brand_name']} "
            f"{quality['product_name']}** has a {quality['avg_rating']:.1f}/5 "
            f"average from {int(quality['review_count'])} reviewers."
        )

    picks_parts = [s for s in (budget_sentence, quality_sentence) if s]
    summary_picks = " ".join(picks_parts)
    return summary_top, summary_picks


def render_card_blurb(
    profile: pd.Series,
    product_row: pd.Series,
    top_terms: str,
) -> str:
    """One-sentence 'why we picked this' for a specific product card."""
    benefits = extract_benefits(top_terms, 3)
    skin = profile["skin_type"]
    if benefits:
        return (
            f"Reviewers with {skin} skin report "
            f"{list_phrase(benefits)}."
        )
    return (
        f"Closely matches what reviewers with {skin} skin described "
        f"for this concern."
    )


def main() -> None:
    """CLI entry: write summaries, card blurbs, and badge flags."""
    parser = argparse.ArgumentParser(description="Build templated summaries + card blurbs.")
    add_env_arg(parser)
    args = parser.parse_args()

    paths = resolve_paths(args.env)
    if paths.env != "local":
        raise NotImplementedError("aws mode wired in Phase 5")

    processed = Path(paths.processed)
    products = pd.read_parquet(processed / "products.parquet")
    profiles = pd.read_parquet(processed / "profiles.parquet")
    recs = pd.read_csv(processed / "precomputed" / "recommendations.csv")
    overlap = pd.read_parquet(processed / "term_overlap.parquet")

    recs_full = recs.merge(
        products[["product_id", "product_name", "brand_name", "price_usd",
                  "avg_rating", "review_count"]],
        on="product_id",
        how="left",
    )
    recs_full["is_budget_pick"] = False
    recs_full["is_quality_pick"] = False

    summaries = []
    card_blurbs = []
    for _, prof in profiles.iterrows():
        top10 = recs_full[recs_full["profile_id"] == prof["profile_id"]].copy()
        if top10.empty:
            continue

        summary_top, summary_picks = render_profile_summary(prof, top10, overlap)
        summaries.append(
            {
                "profile_id": prof["profile_id"],
                "summary_top": summary_top,
                "summary_picks": summary_picks,
            }
        )

        budget = pick_budget(top10)
        if budget is not None:
            recs_full.loc[
                (recs_full["profile_id"] == prof["profile_id"])
                & (recs_full["product_id"] == budget["product_id"]),
                "is_budget_pick",
            ] = True

        quality = pick_quality(top10)
        if quality is not None:
            recs_full.loc[
                (recs_full["profile_id"] == prof["profile_id"])
                & (recs_full["product_id"] == quality["product_id"]),
                "is_quality_pick",
            ] = True

        for _, rec in top10.iterrows():
            match = overlap[
                (overlap["profile_id"] == prof["profile_id"])
                & (overlap["product_id"] == rec["product_id"])
            ]
            terms = match.iloc[0]["top_terms"] if not match.empty else ""
            card_blurbs.append(
                {
                    "profile_id": prof["profile_id"],
                    "product_id": rec["product_id"],
                    "blurb": render_card_blurb(prof, rec, terms),
                }
            )

    pd.DataFrame(summaries).to_parquet(processed / "profile_summaries.parquet", index=False)
    pd.DataFrame(card_blurbs).to_parquet(processed / "card_explanations.parquet", index=False)

    out_cols = [
        "profile_id", "skin_type", "skin_concern", "category",
        "rank", "product_id", "similarity_score",
        "is_budget_pick", "is_quality_pick",
    ]
    recs_full[out_cols].to_csv(processed / "precomputed" / "recommendations.csv", index=False)

    log.info(
        "Wrote profile_summaries (%d), card_explanations (%d); "
        "updated recommendations.csv with budget/quality flags.",
        len(summaries),
        len(card_blurbs),
    )


if __name__ == "__main__":
    main()
