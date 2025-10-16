# app/check_db.py
import sys
import pathlib
import sqlite3

# Ensure project root is on sys.path
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.db import DB_PATH, list_tables

def print_tables():
    """Print all tables in the database"""
    tables = list_tables()
    
    if not tables:
        print("No tables found in database.")
        return
    
    print(f"\n📊 Tables in database ({DB_PATH}):")
    print("=" * 60)
    
    for table in tables:
        print(f"\n🔹 Table: {table}")
        
        # Get sample rows
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM "{table}" LIMIT 5')
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute(f'PRAGMA table_info("{table}")')
        columns = [col[1] for col in cursor.fetchall()]
        
        print(f"   Columns: {columns}")
        print(f"   Sample rows: {len(rows)}")
        
        for row in rows:
            print(f"   {dict(zip(columns, row))}")
        
        conn.close()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print_tables()