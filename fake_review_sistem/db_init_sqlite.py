import sqlite3

# buat atau sambungkan ke database lokal di folder data/
conn = sqlite3.connect('data/fake_review.db')
c = conn.cursor()

# buat tabel user (kalau belum ada)
c.execute('''
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE,
  password TEXT,
  fullname TEXT
)
''')

# tambahkan akun login simulasi
try:
    c.execute("INSERT INTO users (username, password, fullname) VALUES (?, ?, ?)",
              ('galang', '12345', 'Galang'))
    conn.commit()
    print("✅ Akun dummy dibuat: username=galang, password=12345")
except:
    print("⚠️ Akun sudah ada, dilewati.")
finally:
    conn.close()
    print("📁 Database tersimpan di data/fake_review.db")