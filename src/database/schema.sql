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
    PRIMARY KEY (profile_id, rank)
);

CREATE INDEX idx_rec_profile ON recommendations (skin_type, skin_concern, category);
