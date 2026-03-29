import sqlite3
import os

db_path = 'quantix_db.sqlite3'
if not os.path.exists(db_path):
    print(f"Database file {db_path} NOT FOUND")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in {db_path}: {tables}")
    if tables:
        for table in tables:
            name = table[0]
            cursor.execute(f"PRAGMA table_info({name});")
            print(f"Columns in {name}: {[col[1] for col in cursor.fetchall()]}")
    conn.close()
