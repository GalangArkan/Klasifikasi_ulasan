import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ======================================
# 1️⃣ LOAD MODEL DAN DATA
# ======================================
print("🔹 Memuat model dan data...")
tfidf = joblib.load('models/tfidf.joblib')

models = {
    "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
    "Logistic Regression": joblib.load("models/logreg.joblib"),
    "Decision Tree": joblib.load("models/dt.joblib")
}

df = pd.read_csv("data/dataset_ulasan_unik_1000.csv")
X_test = tfidf.transform(df["review_text"])
y_test = df["label"]

# ======================================
# 2️⃣ HITUNG AKURASI DAN CONFUSION MATRIX
# ======================================
print("\n📊 Menghitung hasil evaluasi...")
results = {}
conf_matrices = {}

for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    cm = confusion_matrix(y_test, preds)
    results[name] = acc
    conf_matrices[name] = cm
    print(f"{name} - Akurasi: {acc:.2f}%")

# ======================================
# 3️⃣ GABUNGKAN PLOTTING
# ======================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Grafik Perbandingan Akurasi
ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies = list(results.values())
bars = ax1.bar(model_names, accuracies, color=["#FF6B6B", "#4D96FF", "#74C69D"])
ax1.set_ylim(0, 100)
ax1.set_title("Perbandingan Akurasi Model Machine Learning", fontsize=13, fontweight='bold')
ax1.set_xlabel("Model", fontsize=11)
ax1.set_ylabel("Akurasi (%)", fontsize=11)

# Tambahkan label akurasi di atas batang
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 1, f"{acc:.2f}%", ha='center', fontsize=10, fontweight='bold')

# Confusion Matrix tiap model
for ax, (name, cm) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Asli', 'Palsu'], yticklabels=['Asli', 'Palsu'], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}", fontsize=11)
    ax.set_xlabel("Prediksi", fontsize=10)
    ax.set_ylabel("Sebenarnya", fontsize=10)

plt.tight_layout()
plt.savefig("models/combined_result.png", dpi=300)
plt.show()

print("\n✅ Grafik gabungan disimpan di: models/combined_result.png")