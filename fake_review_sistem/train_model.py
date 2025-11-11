# === train_model.py ===
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Baca dataset
train_path = "data/train_dataset.csv"
test_path = "data/test_dataset.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("⚠️ Dataset belum ditemukan. Jalankan dulu build_dataset_from_db() di scraper_tokopedia.py")
    exit()

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"📊 Data Training: {len(train_df)} | Data Testing: {len(test_df)}")

# 2️⃣ Persiapan data
X_train = train_df["review_text"].astype(str)
y_train = train_df["label"]
X_test = test_df["review_text"].astype(str)
y_test = test_df["label"]

# 3️⃣ TF-IDF Vectorizer
print("🔧 Membuat TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4️⃣ Latih semua model
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = {}

print("\n🚀 Mulai Training Model...")
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc*100:.2f}%")
    results[name] = acc
    print(classification_report(y_test, preds, digits=2))

# 5️⃣ Simpan model dan TF-IDF
os.makedirs("models", exist_ok=True)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(models["Naive Bayes"], f)

with open("models/logreg.pkl", "wb") as f:
    pickle.dump(models["Logistic Regression"], f)

with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(models["Decision Tree"], f)

# 6️⃣ Simpan hasil akurasi untuk dashboard
info = {"accuracies": results}
with open("models/model_accuracy.json", "w") as f:
    import json
    json.dump(info, f, indent=2)

print("\n✅ Semua model dan TF-IDF berhasil disimpan ke folder 'models/'!")