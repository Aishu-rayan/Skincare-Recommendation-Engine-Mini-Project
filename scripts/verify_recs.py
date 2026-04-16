"""Verify the blended similarity_score landed in RDS."""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["RDS_HOST"],
    port=int(os.environ.get("RDS_PORT", "5432")),
    dbname=os.environ["RDS_DB"],
    user=os.environ["RDS_USER"],
    password=os.environ["RDS_PASSWORD"],
)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM recommendations")
print("Total rec rows:", cur.fetchone()[0])

cur.execute(
    """
    SELECT MIN(similarity_score), AVG(similarity_score), MAX(similarity_score)
    FROM recommendations
    """
)
lo, avg, hi = cur.fetchone()
print(f"similarity_score min={lo:.4f} avg={avg:.4f} max={hi:.4f}")

cur.execute(
    """
    SELECT rank, product_id, ROUND(similarity_score::numeric, 4) AS score
    FROM recommendations
    WHERE skin_type='oily' AND skin_concern='acne' AND category='Moisturizers'
    ORDER BY rank
    """
)
print("\noily x acne x Moisturizers:")
for r in cur.fetchall():
    print(" ", r)

conn.close()
