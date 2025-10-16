# app/print_columns.py
import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db import DB_PATH, list_tables
import sqlite3

def print_all_columns():
    """Print columns for all tables"""
    tables = list_tables()
    
    if not tables:
        print("No tables in database.")
        return
    
    print(f"\n📊 Database: {DB_PATH}")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        columns = cursor.fetchall()
        
        print(f"\n🔹 Table: {table}")
        print(f"   Columns ({len(columns)}):")
        for col in columns:
            print(f"      - {col[1]} ({col[2]})")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print_all_columns()