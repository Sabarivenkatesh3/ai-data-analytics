# app/print_columns.py
from sqlalchemy import create_engine, inspect
engine = create_engine("sqlite:///./data/cleaned_data.db", connect_args={"check_same_thread": False})
insp = inspect(engine)
print("Tables:", insp.get_table_names())
cols = insp.get_columns('sales')
print("Columns in 'sales':", [c['name'] for c in cols])
