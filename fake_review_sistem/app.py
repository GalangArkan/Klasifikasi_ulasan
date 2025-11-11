# === Import Library ===
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import json
import numpy as np
from joblib import load
from datetime import datetime
from preprocess_and_train import clean_text
from sklearn.metrics.pairwise import cosine_similarity  # (opsional, tidak digunakan langsung)
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# === Konfigurasi Aplikasi Flask ===
app = Flask(__name__)
app.secret_key = "secret_key_galang"

# === Path Database ===
DB_PATH = 'data/fake_review.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# === Fungsi bantu untuk ambil akurasi model ===
def get_model_accuracy(model_name):
    try:
        return accuracies.get(model_name, 0)
    except:
        return 0

# === Load Model & TF-IDF ===
MODEL_DIR = 'models'
import pickle
try:
    with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'naive_bayes.pkl'), 'rb') as f:
        nb = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'logreg.pkl'), 'rb') as f:
        lr = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'decision_tree.pkl'), 'rb') as f:
        dt = pickle.load(f)
    print("✅ Model & TF-IDF berhasil dimuat dari folder 'models'")
except Exception as e:
    print(f"⚠️ Gagal memuat model: {e}")

# === Load Data Akurasi Model ===
try:
    with open(os.path.join(MODEL_DIR, 'model_accuracy.json'), 'r') as f:
        model_info = json.load(f)
        accuracies = model_info.get("accuracies", {})
        category_stats = model_info.get("category_stats", {})
except Exception as e:
    print(f"⚠️ Tidak bisa membaca 'model_accuracy.json': {e}")
    accuracies, category_stats = {}, {}

# ================= AUTH =================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nama = request.form['nama'].strip()
        email = request.form['email'].strip()
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        confirm = request.form['confirm'].strip()

        if not nama or not email or not username or not password:
            flash('Semua kolom harus diisi!', 'danger')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Password dan verifikasi password tidak sama!', 'danger')
            return redirect(url_for('register'))

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username=? OR email=?", (username, email))
        if cur.fetchone():
            flash('Username atau email sudah digunakan!', 'danger')
            conn.close()
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        cur.execute("INSERT INTO users (nama, email, username, password) VALUES (?, ?, ?, ?)",
                    (nama, email, username, hashed_pw))
        conn.commit()
        conn.close()
        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', title="Buat Akun Baru")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user'] = username
            flash('Berhasil login!', 'success')
            return redirect(url_for('dashboard'))

        flash('Username atau password salah!', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html', title="Login")


@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip()
        new_pass = request.form['new_pass'].strip()
        confirm = request.form['confirm'].strip()

        if new_pass != confirm:
            flash('Password baru dan konfirmasi tidak sama!', 'danger')
            return redirect(url_for('forgot_password'))

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE email=?", (email,))
        user = cur.fetchone()

        if not user:
            flash('Email tidak ditemukan!', 'danger')
            conn.close()
            return redirect(url_for('forgot_password'))

        hashed_pw = generate_password_hash(new_pass)
        cur.execute("UPDATE users SET password=? WHERE email=?", (hashed_pw, email))
        conn.commit()
        conn.close()

        flash('Password berhasil diperbarui! Silakan login.', 'success')
        return redirect(url_for('login'))

    return render_template('forgot.html', title="Lupa Kata Sandi")


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Anda telah keluar.', 'info')
    return redirect(url_for('login'))

# ================= DASHBOARD =================

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session.get('user')
    conn = get_db()
    cur = conn.cursor()
    # gunakan confidence SEJATI: jika prediksi 1 (Asli) -> confidence; jika 0 (Palsu) -> 1-confidence
    cur.execute('''
        SELECT 
            CASE 
                WHEN AVG(CASE WHEN prediction=1 THEN confidence ELSE (1.0 - confidence) END) >= 0.5 THEN 'Asli'
                ELSE 'Palsu'
            END AS final_pred
        FROM results
        WHERE username = ?
        GROUP BY review
    ''', (username,))
    rows = cur.fetchall()
    conn.close()

    total = len(rows)
    asli = sum(1 for r in rows if r[0] == 'Asli')
    palsu = sum(1 for r in rows if r[0] == 'Palsu')

    return render_template('dashboard.html', total=total, asli=asli, palsu=palsu)

# ================= ANALYZE =================

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        review = request.form['review'].strip()
        category = request.form.get('category', 'Manual Input')
        model_choice = request.form.get('model', 'all')

        clean = clean_text(review)
        vect = tfidf.transform([clean]) if tfidf else None
        results = {}
        username = session.get('user')

        if tfidf is None or nb is None or lr is None or dt is None:
            flash('Model/vektorizer belum termuat dengan benar.', 'danger')
            return redirect(url_for('analyze'))

        conn = get_db()
        cur = conn.cursor()

        def save_result_to_db(model_name, pred, proba, category_value):
            # proba = [p_palsu, p_asli]
            confidence_value = max(proba)  # ambil nilai tertinggi (asli atau palsu)
            cur.execute(
                "INSERT INTO results (username, review, model, prediction, confidence, category, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now','localtime'))",
                (username, review, model_name, int(pred), float(confidence_value), category_value)
             )

        models = [
            ('Naive Bayes', nb),
            ('Logistic Regression', lr),
            ('Decision Tree', dt)
        ]

        for model_name, model in models:
            key = model_name.lower().replace(' ', '_')
            if model is None:
                continue
            if model_choice in ('all', key):
                proba = model.predict_proba(vect)[0]
                p_palsu = round(proba[0] * 100, 2)
                p_asli = round(proba[1] * 100, 2)

                pred_label = 1 if p_asli >= 50 else 0
                conf_value = proba[1] if pred_label == 1 else proba[0]

                results[model_name] = {"p_asli": p_asli, "p_palsu": p_palsu}
                save_result_to_db(model_name, pred_label, proba, category)

        conn.commit()
        conn.close()

        # === Tambahkan interpretasi untuk setiap model ===
        for m in results.keys():
            p_asli = results[m]["p_asli"]
            p_palsu = results[m]["p_palsu"]
            if p_asli > p_palsu:
                results[m]["interpretasi"] = "Asli"
            elif p_palsu > p_asli:
                results[m]["interpretasi"] = "Palsu"
            else:
                results[m]["interpretasi"] = "Netral"

        # === Kesimpulan Akhir ===
        if results:
            avg_asli = np.mean([v["p_asli"] for v in results.values()])
            avg_palsu = np.mean([v["p_palsu"] for v in results.values()])
            final_label = "Asli" if avg_asli > avg_palsu else "Palsu"
            final_color = "success" if final_label == "Asli" else "danger"
            final_confidence_value = avg_asli if avg_asli > avg_palsu else avg_palsu
            final_text = f"{final_confidence_value:.2f}%"
        else:
            final_label = "Tidak Diketahui"
            final_color = "secondary"
            final_confidence_value = 0
            final_text = "0.00%"

        return render_template(
            'results.html',
            review=review,
            results=results,
            final_label=final_label,
            final_color=final_color,
            final_text=final_text,
            final_confidence_value=final_confidence_value,
            accuracies=accuracies
        )

    return render_template('analyze.html')

# ================= HISTORY =================

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session.get('user')
    conn = get_db()
    cur = conn.cursor()

    # ✅ Ambil hanya hasil TERBARU untuk setiap (review, model)
    cur.execute('''
        WITH latest_results AS (
            SELECT r1.*
            FROM results r1
            INNER JOIN (
                SELECT review, model, MAX(created_at) AS max_time
                FROM results
                WHERE username = ?
                GROUP BY review, model
            ) r2
            ON r1.review = r2.review AND r1.model = r2.model AND r1.created_at = r2.max_time
            WHERE r1.username = ?
        )
        SELECT 
            MIN(id) AS id,
            review,
            COALESCE(MAX(category), 'Tidak Diketahui') AS category,
            GROUP_CONCAT(model, ', ') AS models,
            CASE 
                WHEN AVG(prediction) >= 0.5 THEN 'Asli'
                ELSE 'Palsu'
            END AS final_pred,
            ROUND(AVG(confidence) * 100, 2) AS avg_confidence,
            MAX(created_at) AS created_at
        FROM latest_results
        GROUP BY review
        ORDER BY MAX(created_at) DESC;
    ''', (username, username))

    rows = cur.fetchall()
    conn.close()

    formatted_data = []
    for row in rows:
        id_, review, category, models, final_pred, avg_conf, created_at = row
        if created_at:
            try:
                tgl = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                created_at = tgl.strftime("%d %B %Y, %H:%M")
            except:
                pass

        formatted_data.append({
            'id': id_,
            'review': review,
            'category': category,
            'models': models,
            'final_pred': final_pred,
            'conf': avg_conf,
            'created_at': created_at
        })

    return render_template('history.html', data=formatted_data)

@app.route('/recommendations')
def recommendations():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cur = conn.cursor()
    # (versi produk; nantinya bisa diganti per toko)
    cur.execute('''
        SELECT product_id, category, COUNT(*) AS total_ulasan,
               SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS asli_count
        FROM results
        GROUP BY product_id, category
        HAVING total_ulasan > 0
        ORDER BY asli_count DESC, total_ulasan DESC
        LIMIT 5
    ''')
    rows = cur.fetchall()
    conn.close()

    rekomendasi = []
    for r in rows:
        rekomendasi.append({
            "product_id": r["product_id"],
            "category": r["category"],
            "total_ulasan": r["total_ulasan"],
            "asli_count": r["asli_count"]
        })

    return render_template("recommendations.html", rekomendasi=rekomendasi)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    ids_to_delete = request.form.getlist('selected_ids')
    if not ids_to_delete:
        flash('Tidak ada data yang dipilih untuk dihapus.', 'warning')
        return redirect(url_for('history'))

    conn = get_db()
    cur = conn.cursor()
    cur.executemany("DELETE FROM results WHERE id = ?", [(i,) for i in ids_to_delete])
    conn.commit()
    conn.close()

    flash(f"{len(ids_to_delete)} riwayat berhasil dihapus.", "success")
    return redirect(url_for('history'))

# ================= DETAIL =================

@app.route('/detail/<int:id>')
def detail_analysis(id):
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session.get('user')
    conn = get_db()
    cur = conn.cursor()

    # ambil review & kategori dari id
    cur.execute('SELECT review, COALESCE(category, "Tidak Diketahui") AS category FROM results WHERE id = ?', (id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        flash("Data tidak ditemukan.", "warning")
        return redirect(url_for('history'))

    review = row['review']
    category = row['category']

    # ambil semua hasil terakhir per model untuk review yang sama
    cur.execute('''
        SELECT model, prediction, confidence, MAX(created_at) AS created_at
        FROM results
        WHERE username = ? AND review = ?
        GROUP BY model
        ORDER BY model
    ''', (username, review))
    rows = cur.fetchall()
    conn.close()

    explanations = []
    vect = tfidf.transform([clean_text(review)]) if tfidf else None

    for r in rows:
        model = r['model']
        pred = "Asli" if r['prediction'] == 1 else "Palsu"

        if model == "Naive Bayes" and nb is not None:
            proba = nb.predict_proba(vect)[0]
        elif model == "Logistic Regression" and lr is not None:
            proba = lr.predict_proba(vect)[0]
        elif model == "Decision Tree" and dt is not None:
            proba = dt.predict_proba(vect)[0]
        else:
            proba = [0.5, 0.5]

        p_palsu = round(proba[0] * 100, 2)
        p_asli  = round(proba[1] * 100, 2)

        low = model.lower()
        if low == "naive bayes":
            reason = ("Model <b>Naive Bayes</b> menilai probabilitas kata pada tiap kelas. "
                      f"Ulasan ini cenderung <b>{pred.lower()}</b> karena pola kata lebih dekat ke distribusi kelas tersebut.")
        elif low == "logistic regression":
            reason = ("Model <b>Logistic Regression</b> menggunakan bobot fitur (koefisien). "
                      f"Kombinasi kata mendorong skor logit ke arah kelas <b>{pred.lower()}</b>.")
        elif low == "decision tree":
            reason = ("Model <b>Decision Tree</b> memakai aturan if–then. "
                      f"Pola fitur teks mengikuti jalur yang berakhir pada kelas <b>{pred.lower()}</b>.")
        else:
            reason = "Model tidak dikenali."

        created_at = r['created_at']
        if created_at:
            try:
                tgl = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                created_at = tgl.strftime("%d %B %Y, %H:%M")
            except:
                pass

        explanations.append({
            "model": model,
            "p_asli": p_asli,
            "p_palsu": p_palsu,
            "prediction": pred,
            "created_at": created_at,
            "reason": reason
        })

    # === Ringkasan confidence & label mayoritas ===
    if rows:
        # confidence sejati rata-rata
        model_accs = [get_model_accuracy(r['model']) for r in rows]
        avg_confidence_pct = round(float(np.mean([r['confidence'] for r in rows])) * 100, 2)

        # majority vote
        pred_asli = sum(1 for r in rows if r['prediction'] == 1)
        pred_palsu = sum(1 for r in rows if r['prediction'] == 0)
        conf_label = "Asli" if pred_asli >= pred_palsu else "Palsu"
        conf_color = "success" if conf_label == "Asli" else "danger"
    else:
        avg_confidence_pct = 0.0
        conf_label = "Tidak tersedia"
        conf_color = "secondary"

    # === Cari Ulasan Serupa (>= 2 kata sama) ===
    similar_table = []
    dataset_path = "data/dataset_ulasan_unik_1000.csv"
    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if "review_text" in df.columns:
                df["review_text"] = df["review_text"].fillna("")
                main_words = set(clean_text(review).split())

                for _, data in df.iterrows():
                    candidate = str(data["review_text"])
                    words_candidate = set(clean_text(candidate).split())
                    common_words = main_words.intersection(words_candidate)

                    if len(common_words) >= 2:
                        highlighted = candidate
                        for w in common_words:
                            highlighted = highlighted.replace(w, f"<mark>{w}</mark>")
                        similar_table.append({"ulasan": highlighted})
                similar_table = similar_table[:3]
    except Exception as e:
        print(f"⚠️ Gagal mencari ulasan serupa: {e}")
        similar_table = []

    return render_template(
        'detail.html',
        review=review,
        category=category,
        explanations=explanations,
        similar_table=similar_table,
        avg_confidence=avg_confidence_pct,
        conf_label=conf_label,
        conf_color=conf_color,
        accuracies=accuracies
    )

# ================= SCRAPER (TOKOPEDIA) =================

@app.route('/scrape_tokopedia')
def scrape_tokopedia():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        import scraper_tokopedia
        total = scraper_tokopedia.scrape_and_save()
        flash(f"Berhasil mengambil {total} ulasan dari Tokopedia!", "success")
    except Exception as e:
        flash(f"Gagal mengambil data: {str(e)}", "danger")

    return redirect(url_for('dashboard'))

# ================= PROFILE / HOME =================

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    username = session['user']
    conn = get_db()
    cur = conn.cursor()
    # gunakan kolom 'nama' (bukan 'name')
    cur.execute("SELECT nama, username, email, created_at FROM users WHERE username=?", (username,))
    user_data = cur.fetchone()
    conn.close()
    user = {
        'name': user_data[0] if user_data else '',
        'username': user_data[1] if user_data else username,
        'email': user_data[2] if user_data else '',
        'created_at': user_data[3] if user_data else ''
    }
    return render_template('profile.html', user=user)

@app.route('/')
def home():
    return redirect(url_for('dashboard')) if 'user' in session else redirect(url_for('login'))

@app.route('/blackbox')
def blackbox():
    return render_template('blackbox.html')

if __name__ == '__main__':
    app.run(debug=True)