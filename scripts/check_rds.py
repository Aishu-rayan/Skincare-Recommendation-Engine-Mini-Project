"""Quick RDS connectivity check using values from .env."""
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
    connect_timeout=10,
)
cur = conn.cursor()
cur.execute("SELECT version()")
print("CONNECTED:", cur.fetchone()[0])
cur.execute(
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema='public' ORDER BY table_name"
)
print("TABLES:", [r[0] for r in cur.fetchall()])
conn.close()
