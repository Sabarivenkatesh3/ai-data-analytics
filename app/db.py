# app/db.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.exc import OperationalError

DB_URL = "sqlite:///./data/cleaned_data.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

# Simple table for sales
sales_table = Table(
    "sales", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", Integer),
    Column("date", String),
    Column("region", String),
    Column("product", String),
    Column("quantity", Integer),
    Column("price", Float)
)

def init_db():
    try:
        metadata.create_all(engine)
    except OperationalError as e:
        print("DB init error:", e)
