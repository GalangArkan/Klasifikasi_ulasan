-- Tambah kolom toko ke reviews (abaikan error jika kolom sudah ada)
ALTER TABLE reviews ADD COLUMN product_url TEXT;
ALTER TABLE reviews ADD COLUMN store_id TEXT;
ALTER TABLE reviews ADD COLUMN store_name TEXT;

-- Tambah kolom toko ke results (abaikan error jika kolom sudah ada)
ALTER TABLE results ADD COLUMN product_id TEXT;
ALTER TABLE results ADD COLUMN store_id TEXT;
ALTER TABLE results ADD COLUMN store_name TEXT;

-- Indeks untuk kueri cepat per toko
CREATE INDEX IF NOT EXISTS idx_reviews_store ON reviews(store_id);
CREATE INDEX IF NOT EXISTS idx_results_store ON results(store_id);
CREATE INDEX IF NOT EXISTS idx_reviews_label ON reviews(label);