import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_save_models():
    print("🚀 Memulai pelatihan model...")

    train_path = "data/train_dataset.csv"
    test_path = "data/test_dataset.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("⚠️ Dataset belum tersedia. Jalankan build_dataset_from_db() dulu.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["review_text"].apply(clean_text)
    y_train = train_df["label"]
    X_test = test_df["review_text"].apply(clean_text)
    y_test = test_df["label"]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=10)
    }

    accuracies = {}

    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        print(f"\n🔹 Melatih model: {name}")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        print(classification_report(y_test, preds))
        accuracies[name] = round(acc * 100, 2)
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.joblib")

    joblib.dump(tfidf, "models/tfidf.joblib")

    # Simpan hasil akurasi
    model_info = {"accuracies": accuracies}
    with open("models/model_accuracy.json", "w") as f:
        json.dump(model_info, f, indent=4)

    print("\n✅ Semua model berhasil disimpan ke folder 'models'")
    print(json.dumps(accuracies, indent=4))

if __name__ == "__main__":
    train_and_save_models()