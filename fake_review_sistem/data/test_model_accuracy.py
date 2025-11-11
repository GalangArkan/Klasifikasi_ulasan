import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import json

# 🔧 Tambahan penting agar bisa impor modul dari folder utama
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocess_and_train import clean_text

# 🚫 Matikan peringatan metrik (precision ill-defined, dll)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 📂 Path dataset dan model
DATA_PATH = os.path.join(os.path.dirname(__file__), 'dataset_ulasan_unik_1000.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')

# ✅ Pastikan file dataset ada
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

# 📥 Muat dataset
df = pd.read_csv(DATA_PATH)

# ✅ Pastikan kolom yang dibutuhkan ada
required_cols = {'review_text', 'label'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Dataset harus memiliki kolom: {required_cols}")

# 🧹 Bersihkan teks ulasan
df['clean'] = df['review_text'].astype(str).apply(clean_text)

# 🎯 Split data train-test
X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['label'], test_size=0.2, random_state=42)

# 🔤 Muat TF-IDF
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf.joblib'))

X_test_tfidf = tfidf.transform(X_test)

# 📦 Muat semua model
models = {
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, 'naive_bayes.joblib')),
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, 'logreg.joblib')),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, 'dt.joblib'))
}

# 📊 Simpan hasil
results = {}

for name, model in models.items():
    print(f"\n🔍 Melatih model: {name} ...")
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(report['weighted avg']['precision'] * 100, 2),
        "recall": round(report['weighted avg']['recall'] * 100, 2),
        "f1-score": round(report['weighted avg']['f1-score'] * 100, 2)
    }
    
    print(f"✅ Akurasi {name}: {acc * 100:.2f}%")
    print(pd.DataFrame(report).transpose())

# 💾 Simpan hasil evaluasi ke CSV
result_df = pd.DataFrame(results).T
output_csv = os.path.join(os.path.dirname(__file__), 'hasil_pengujian_model.csv')
result_df.to_csv(output_csv)
print(f"\n📂 Hasil evaluasi disimpan di: {output_csv}")

# 💾 Simpan juga ke JSON (untuk sistem utama)
output_json = os.path.join(MODEL_DIR, 'model_accuracy.json')
with open(output_json, 'w') as f:
    json.dump({k: v['accuracy'] for k, v in results.items()}, f, indent=4)
print(f"📊 File model_accuracy.json diperbarui di folder models/")