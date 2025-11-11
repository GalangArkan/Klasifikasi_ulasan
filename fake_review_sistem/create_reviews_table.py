# create_reviews_table.py
import sqlite3, os

db_path = 'data/fake_review.db'

# Pastikan folder data ada
os.makedirs('data', exist_ok=True)

# Koneksi ke database
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Buat tabel reviews jika belum ada
cur.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    product_id TEXT NOT NULL,
    user TEXT,
    rating INTEGER,
    content TEXT NOT NULL,
    label INTEGER,            -- 0=Asli, 1=Palsu (auto-label)
    created_at TEXT           -- waktu dari platform (jika ada)
);
""")

# Buat index untuk mempercepat query
cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_cat ON reviews(category);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_prod ON reviews(product_id);")

conn.commit()
conn.close()

print("✅ Tabel 'reviews' berhasil dibuat atau sudah ada.")