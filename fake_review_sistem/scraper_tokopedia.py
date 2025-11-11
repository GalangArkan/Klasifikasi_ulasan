import json, os, sqlite3, time, random, requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from auto_label import auto_label
from datetime import datetime

DB_PATH = 'data/fake_review.db'
PRODUCTS_JSON = 'data/products_tokopedia.json'

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7"
}


# ====================== SCRAPER ======================

def infer_store_from_url(url: str):
    """Ambil slug toko dari URL Tokopedia."""
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    store = parts[0] if parts else "unknown-store"
    return store, store  # pakai slug toko sebagai store_id & store_name


def get_tokopedia_reviews_from_html(product_url: str, limit: int = 20):
    """Ambil beberapa ulasan dari HTML publik produk Tokopedia."""
    print(f"🔍 Scraping HTML: {product_url}")
    try:
        res = requests.get(product_url, headers=HEADERS, timeout=12)
        if res.status_code != 200:
            print(f"⚠️ Gagal ambil {product_url} (status {res.status_code})")
            return []

        soup = BeautifulSoup(res.text, "lxml")
        reviews = []

        # Cari blok teks yang mirip review
        candidates = soup.find_all(
            lambda tag: tag.name in ['div', 'article'] and tag.get_text(strip=True),
            limit=1000
        )

        for c in candidates:
            txt = c.get_text(" ", strip=True)
            if len(txt) < 40:
                continue
            low = txt.lower()
            if not any(k in low for k in ["produk", "pengiriman", "kualitas", "sesuai", "tidak", "bagus", "kurang"]):
                continue

            # Ambil rating kasar dari teks (misal "5 bintang")
            rating = 0
            if "bintang" in low:
                import re
                m = re.search(r'([1-5])\s*bintang', low)
                if m:
                    rating = int(m.group(1))

            reviews.append({
                "user": "Anonim",
                "rating": rating,
                "content": txt,
                "created_at": datetime.now().strftime("%Y-%m-%d")
            })
            if len(reviews) >= limit:
                break

        # Hilangkan duplikat
        unique = []
        seen = set()
        for r in reviews:
            key = r["content"][:120]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        print(f"✅ {len(unique)} ulasan ditemukan.")
        return unique

    except Exception as e:
        print(f"⚠️ Error scraping: {e}")
        return []


def scrape_and_save():
    """Scrape dari daftar produk di products_tokopedia.json dan simpan ke DB."""
    if not os.path.exists(PRODUCTS_JSON):
        print(f"⚠️ File {PRODUCTS_JSON} tidak ditemukan.")
        return 0

    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        product_map = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    total_saved = 0

    for category, urls in product_map.items():
        for url in urls:
            if not url:
                continue
            store_name, store_id = infer_store_from_url(url)
            reviews = get_tokopedia_reviews_from_html(url, limit=5)
            time.sleep(1 + random.random() * 1.5)  # delay ringan

            for r in reviews:
                lbl = auto_label(r.get("content", ""))
                cur.execute("""
                    INSERT INTO reviews
                    (category, product_id, product_url, store_id, store_name,
                     user, rating, content, label, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    category,
                    url,
                    url,
                    store_id,
                    store_name,
                    r.get("user", "Anonim"),
                    int(r.get("rating", 0)),
                    r.get("content", "").strip(),
                    int(lbl),
                    r.get("created_at", datetime.now().strftime("%Y-%m-%d"))
                ))
                total_saved += 1

    conn.commit()
    conn.close()
    print(f"🎉 {total_saved} ulasan berhasil disimpan ke database (HTML scraping).")
    return total_saved


# ====================== DUMMY DATA GENERATOR ======================

def generate_dummy_reviews(total: int = 700):
    """
    Membuat dataset tambahan secara otomatis (dummy).
    Semua data tetap dilabeli oleh fungsi auto_label().
    """
    import random
    import sqlite3
    from datetime import datetime

    positive_texts = [
        "Produk sangat bagus, pengiriman cepat dan sesuai deskripsi.",
        "Kualitas oke banget, harga terjangkau.",
        "Barang diterima dengan baik, recommended seller.",
        "Sesuai harapan, berfungsi dengan baik.",
        "Pelayanan cepat dan responsif, puas banget!"
    ]

    negative_texts = [
        "Produk tidak sesuai, kualitas buruk.",
        "Barang rusak saat diterima, sangat mengecewakan.",
        "Tidak sesuai deskripsi, pengiriman lama.",
        "Kualitas jauh dari harapan.",
        "Barang palsu, tidak direkomendasikan."
    ]

    categories = ["Elektronik", "Fashion", "Kecantikan", "Makanan", "Kesehatan"]
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total_saved = 0
    for _ in range(total):
        cat = random.choice(categories)
        txt = random.choice(positive_texts + negative_texts)
        label = auto_label(txt)
        cur.execute("""
            INSERT INTO reviews 
            (category, product_id, product_url, store_id, store_name, 
             user, rating, content, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cat,
            f"DUMMY_{random.randint(1000, 9999)}",
            "dummy",
            f"store_{cat.lower()}",
            f"Toko {cat}",
            "Anonim",
            random.randint(1, 5),
            txt,
            int(label),
            datetime.now().strftime("%Y-%m-%d")
        ))
        total_saved += 1

    conn.commit()
    conn.close()
    print(f"🎉 {total_saved} data dummy berhasil dibuat dan disimpan ke database!")
    return total_saved


# ====================== DATASET BUILDER ======================

def build_dataset_from_db():
    """
    🔧 Ambil seluruh ulasan dari database dan bangun dataset training (700) dan testing (300).
    Dataset otomatis dibuat di folder data/.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT 
            store_name,
            category,
            rating,
            content AS review_text,
            label
        FROM reviews
        WHERE content IS NOT NULL AND TRIM(content) != ''
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ Tidak ada data ulasan di database. Jalankan scraping dulu.")
        return

    print(f"📊 Total data ditemukan: {len(df)} ulasan")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df)
    train_size = min(700, int(total * 0.7))
    test_size = min(300, total - train_size)

    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:train_size + test_size]

    os.makedirs("data", exist_ok=True)
    df_train.to_csv("data/train_dataset.csv", index=False, encoding="utf-8")
    df_test.to_csv("data/test_dataset.csv", index=False, encoding="utf-8")

    print(f"✅ Dataset otomatis dibuat:")
    print(f"   - Total : {total} data")
    print(f"   - Train : {len(df_train)} data")
    print(f"   - Test  : {len(df_test)} data")
    print("📁 Disimpan di: data/train_dataset.csv dan data/test_dataset.csv")


# ====================== MAIN ======================

if __name__ == "__main__":
    scrape_and_save()