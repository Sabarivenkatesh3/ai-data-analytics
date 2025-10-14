# app/db.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, text
from sqlalchemy.exc import OperationalError

DB_URL = "sqlite:///./data/cleaned_data.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

# Define table including the new column (so SQLAlchemy metadata knows about it)
sales_table = Table(
    "sales", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", Integer),
    Column("date", String),
    Column("region", String),
    Column("product", String),
    Column("quantity", Integer),
    Column("price", Float),
    Column("total_amount", Float)  # new column
)

def init_db():
    try:
        metadata.create_all(engine)
    except OperationalError as e:
        print("DB init error:", e)

def ensure_total_amount_column():
    """
    Ensure the 'total_amount' column exists in the existing SQLite table.
    We attempt to SELECT it; if that fails we ALTER TABLE to add it.
    """
    try:
        with engine.begin() as conn:
            # Try selecting the column — if it fails an exception is raised
            try:
                conn.execute(text("SELECT total_amount FROM sales LIMIT 1"))
            except Exception:
                # Column doesn't exist, so add it.
                # Note: SQLite allows adding a column with ALTER TABLE ... ADD COLUMN
                conn.execute(text("ALTER TABLE sales ADD COLUMN total_amount FLOAT"))
                print("Added total_amount column to sales table.")
    except Exception as e:
        print("Error ensuring column:", e)
