# app/check_db.py
import sys
import pathlib

# Ensure project root is on sys.path so "app" package imports work when running as a script.
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.db import engine, sales_table
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

def print_rows(limit: int | None = None):
    try:
        stmt = select(sales_table)
        with engine.connect() as conn:
            result = conn.execute(stmt)   # keep Result object to read keys()
            rows = result.fetchall()      # list of Row objects
            cols = result.keys()          # column names (from Result)

            if not rows:
                print("No rows found in 'sales' table.")
                return

            print("Columns:", cols)
            count = 0
            for row in rows:
                # row._mapping is a dict-like mapping of column->value (safe)
                print(dict(row._mapping))
                count += 1
                if limit and count >= limit:
                    break

            print(f"\nTotal rows printed: {count}")

    except SQLAlchemyError as e:
        print("DB error:", e)

if __name__ == "__main__":
    # Recommend running with: python -m app.check_db  (from project root)
    # But this file also works with: python app/check_db.py (from project root)
    print_rows()
