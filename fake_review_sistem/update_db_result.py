import sqlite3
from datetime import datetime

conn = sqlite3.connect('data/fake_review.db')
c = conn.cursor()

# buat tabel results jika belum ada
c.execute('''
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    review TEXT,
    model TEXT,
    prediction INTEGER,
    confidence REAL,
    created_at TEXT
)
''')

conn.commit()
conn.close()
print("✅ Tabel 'results' berhasil dibuat di database!")