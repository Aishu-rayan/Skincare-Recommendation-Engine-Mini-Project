DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS profile_summaries;
DROP TABLE IF EXISTS card_explanations;
DROP TABLE IF EXISTS recommendations;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id    VARCHAR PRIMARY KEY,
    product_name  TEXT,
    brand_name    TEXT,
    category      TEXT,
    price_usd     FLOAT,
    avg_rating    FLOAT,
    review_count  INT
);

CREATE TABLE recommendations (
    profile_id        VARCHAR,
    skin_type         TEXT,
    skin_concern      TEXT,
    category          TEXT,
    rank              INT,
    product_id        VARCHAR REFERENCES products(product_id),
    similarity_score  FLOAT,
    is_budget_pick    BOOLEAN DEFAULT FALSE,
    is_quality_pick   BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (profile_id, rank)
);

CREATE INDEX idx_rec_profile ON recommendations (skin_type, skin_concern, category);

CREATE TABLE profile_summaries (
    profile_id     VARCHAR PRIMARY KEY,
    summary_top    TEXT,
    summary_picks  TEXT
);

CREATE TABLE card_explanations (
    profile_id  VARCHAR,
    product_id  VARCHAR,
    blurb       TEXT,
    PRIMARY KEY (profile_id, product_id)
);

CREATE TABLE reviews (
    product_id      VARCHAR,
    skin_type       TEXT,
    rating          FLOAT,
    helpfulness     FLOAT,
    pos_feedback    INT,
    review_title    TEXT,
    review_text     TEXT
);

CREATE INDEX idx_reviews_lookup ON reviews (product_id, skin_type);
