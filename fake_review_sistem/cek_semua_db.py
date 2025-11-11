import sqlite3
import pandas as pd

# Path ke database
db_path = 'data/fake_review.db'

# Koneksi ke database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Ambil semua nama tabel
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

print("📂 Daftar tabel di database:")
for t in tables:
    print(f" - {t}")
print("\n=============================================\n")

# Tampilkan isi setiap tabel
for t in tables:
    print(f"🧩 Isi tabel: {t}")
    print("---------------------------------------------")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
        if df.empty:
            print("(Tabel ini kosong)\n")
        else:
            print(df.head(50).to_string(index=False))  # tampilkan max 50 baris biar rapi
            print(f"\n📊 Total data di tabel '{t}': {len(df)}\n")
    except Exception as e:
        print(f"❌ Gagal membaca tabel {t}: {e}\n")

    print("=============================================\n")

conn.close()
print("✅ Semua data berhasil ditampilkan.")