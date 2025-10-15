# app/db.py
import os
import re
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")


# -----------------------------------------------------------
# ✅ Initialize or create the database
# -----------------------------------------------------------
def init_db():
    """Initialize SQLite database if not exists."""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        print(f"[DB] Created new database at {DB_PATH}")
    else:
        print(f"[DB] Using existing database at {DB_PATH}")


# -----------------------------------------------------------
# ✅ Create table from DataFrame
# -----------------------------------------------------------
def create_table_from_df(df: pd.DataFrame, table_name: str, if_exists="replace"):
    """Save a pandas DataFrame into SQLite DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        rows = df.shape[0]
        conn.close()
        return {"status": "ok", "table": table_name, "rows": rows}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def get_db_path():
    """
    Return the absolute path to the sqlite DB file.
    """
    # adjust if your DB path variable name is different
    try:
        return DB_PATH
    except NameError:
        # fallback to legacy location inside project
        return os.path.join(os.path.dirname(__file__), "data.db")

def read_table(table_name: str, limit: int = 1000) -> pd.DataFrame:
    """
    Read a table from the sqlite DB and return a pandas DataFrame (up to `limit` rows).
    """
    dbp = get_db_path()
    if not os.path.exists(dbp):
        raise FileNotFoundError(f"Database not found at {dbp}")

    safe_name = sanitize_table_name(table_name)
    conn = sqlite3.connect(dbp)
    try:
        query = f"SELECT * FROM \"{safe_name}\" LIMIT {int(limit)}"
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


# -----------------------------------------------------------
# ✅ List all tables in DB
# -----------------------------------------------------------
def list_tables():
    """Return list of all tables in DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


# -----------------------------------------------------------
# ✅ Sanitize table name
# -----------------------------------------------------------
def sanitize_table_name(name: str) -> str:
    """Make a safe table name from filename."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).lower()
