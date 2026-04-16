"""Streamlit dashboard for the skincare recommender.

Sidebar inputs, pastel palette, card-based results with star ratings
and similarity progress bars.
"""
import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="🌸",
    layout="wide",
)

PALETTE = {
    "bg": "#FDF7F4",
    "card": "#FFFFFF",
    "accent": "#E8A5B8",
    "accent_dark": "#C97B91",
    "text": "#3D2B3D",
    "muted": "#9B8A9B",
    "bar_bg": "#F3E4EA",
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {PALETTE['bg']} !important;
        color: {PALETTE['text']} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #FBEEF0 !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {PALETTE['text']} !important;
    }}
    h1, h2, h3, h4, p, span, label, div {{
        color: {PALETTE['text']};
    }}
    h1, h2, h3 {{
        font-family: 'Georgia', serif;
    }}
    .hero {{
        padding: 28px 32px;
        border-radius: 16px;
        background: linear-gradient(135deg, #FBEEF0 0%, #F4D7DF 100%);
        margin-bottom: 24px;
        border: 1px solid #F3E4EA;
    }}
    .hero h1 {{
        margin: 0 0 8px 0;
        font-size: 38px;
        color: {PALETTE['text']} !important;
        font-weight: 700;
    }}
    .hero .subtitle {{
        color: {PALETTE['accent_dark']};
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }}
    .hero p {{
        margin: 0;
        color: {PALETTE['text']};
        font-size: 15px;
        opacity: 0.75;
    }}
    .card {{
        background: {PALETTE['card']};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(201, 123, 145, 0.08);
        border: 1px solid #F3E4EA;
        min-height: 220px;
    }}
    .rank-badge {{
        display: inline-block;
        background: {PALETTE['accent']};
        color: white;
        font-weight: bold;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 13px;
        margin-bottom: 8px;
    }}
    .brand {{
        color: {PALETTE['muted']};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 2px;
    }}
    .product-name {{
        color: {PALETTE['text']};
        font-size: 15px;
        font-weight: 600;
        line-height: 1.3;
        margin-bottom: 10px;
        min-height: 40px;
    }}
    .price {{
        color: {PALETTE['accent_dark']};
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
    }}
    .stars {{
        color: {PALETTE['accent_dark']};
        font-size: 15px;
        letter-spacing: 2px;
    }}
    .review-count {{
        color: {PALETTE['muted']};
        font-size: 12px;
        margin-left: 6px;
    }}
    .sim-label {{
        color: {PALETTE['muted']};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }}
    .sim-bar {{
        background: {PALETTE['bar_bg']};
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
        margin-top: 4px;
    }}
    .sim-fill {{
        background: linear-gradient(90deg, {PALETTE['accent']} 0%, {PALETTE['accent_dark']} 100%);
        height: 100%;
        border-radius: 999px;
    }}
    .sim-value {{
        color: {PALETTE['accent_dark']};
        font-size: 11px;
        font-weight: bold;
        margin-top: 2px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_conn():
    """Return a cached psycopg2 connection to RDS."""
    return psycopg2.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", "5432")),
        dbname=os.environ["RDS_DB"],
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
    )


@st.cache_data(ttl=300)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a parameterised SQL query and return a DataFrame."""
    conn = get_conn()
    return pd.read_sql(sql, conn, params=params)


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


def render_stars(rating: float | None) -> str:
    """Return a 5-character star string for a 0-5 rating."""
    if rating is None or pd.isna(rating):
        return "☆☆☆☆☆"
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def render_card(row: pd.Series) -> str:
    """Build an HTML card for one recommended product."""
    rating = row["avg_rating"] if not pd.isna(row["avg_rating"]) else None
    stars = render_stars(rating)
    rating_text = f"{rating:.1f}" if rating is not None else "—"
    review_count = int(row["review_count"]) if not pd.isna(row["review_count"]) else 0
    price = f"${row['price_usd']:.2f}" if not pd.isna(row["price_usd"]) else "—"
    sim_pct = max(0.0, min(1.0, float(row["similarity_score"]))) * 100
    return f"""
    <div class="card">
        <span class="rank-badge">#{int(row['rank'])}</span>
        <div class="brand">{row['brand_name'] or ''}</div>
        <div class="product-name">{row['product_name'] or ''}</div>
        <div class="price">{price}</div>
        <div class="stars">{stars} <span class="review-count">{rating_text} · {review_count} reviews</span></div>
        <div class="sim-label">Match score</div>
        <div class="sim-bar"><div class="sim-fill" style="width: {sim_pct:.0f}%;"></div></div>
        <div class="sim-value">{sim_pct:.1f}%</div>
    </div>
    """


with st.sidebar:
    st.markdown("### Your skin profile")
    st.caption("Tell us about your skin and we'll find matching products.")
    skin_type = st.selectbox("Skin type", SKIN_TYPES, format_func=lambda s: s.title())
    skin_concern = st.selectbox("Primary concern", SKIN_CONCERNS, format_func=lambda s: s.title())
    category = st.selectbox("Product category", CATEGORIES)
    submitted = st.button("Find my products", type="primary", use_container_width=True)


st.markdown(
    """
    <div class="hero">
        <div class="subtitle">Personalized Skincare Recommendations</div>
        <h1>The Glow Lab</h1>
        <p>Skincare picks grounded in what people like you actually said.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if submitted or "last_query" in st.session_state:
    if submitted:
        st.session_state["last_query"] = (skin_type, skin_concern, category)
    st_type, st_concern, st_cat = st.session_state["last_query"]

    sql = """
        SELECT r.rank, p.product_name, p.brand_name, p.price_usd,
               p.avg_rating, p.review_count, r.similarity_score
        FROM recommendations r
        JOIN products p ON p.product_id = r.product_id
        WHERE r.skin_type = %s AND r.skin_concern = %s AND r.category = %s
        ORDER BY r.rank
    """
    df = query(sql, (st_type, st_concern, st_cat))

    if df.empty:
        st.warning("No recommendations for that profile yet.")
    else:
        st.subheader(f"Top {len(df)} {st_cat.lower()} for {st_type} skin · {st_concern}")
        cols_per_row = 4
        for start in range(0, len(df), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, (_, row) in zip(cols, df.iloc[start:start + cols_per_row].iterrows()):
                col.markdown(render_card(row), unsafe_allow_html=True)
else:
    st.info("← Use the sidebar to tell us about your skin, then click **Find my products**.")


with st.expander("How coverage looks across categories"):
    coverage = query(
        """
        SELECT category, COUNT(DISTINCT product_id) AS product_count
        FROM products
        GROUP BY category
        ORDER BY product_count DESC
        """
    )
    st.bar_chart(coverage.set_index("category"), color=PALETTE["accent_dark"])
