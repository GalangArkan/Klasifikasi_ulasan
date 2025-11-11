import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Akurasi hasil revisi sistem ---
models = ['Naive Bayes', 'Logistic Regression', 'Decision Tree']
accuracies = [98.0, 97.0, 96.67]

# --- Confusion Matrix hasil pengujian simulasi (asumsi 300 data testing, seimbang 150 asli & 150 palsu) ---
cm_nb = np.array([[147, 3],   # True Negative, False Positive
                  [3, 147]])  # False Negative, True Positive

cm_lr = np.array([[145, 5],
                  [4, 146]])

cm_dt = np.array([[143, 7],
                  [3, 147]])

# --- Membuat layout 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# --- Bar Chart Akurasi ---
sns.barplot(x=models, y=accuracies, palette=['#ff6b6b', '#4dabf7', '#63e6be'], ax=axes[0, 0])
axes[0, 0].set_ylim(90, 100)
axes[0, 0].set_ylabel("Akurasi (%)", fontsize=11)
axes[0, 0].set_xlabel("Model", fontsize=11)
axes[0, 0].set_title("Perbandingan Akurasi Model Machine Learning", fontsize=13, fontweight='bold')

# Tambahkan nilai akurasi di atas batang
for i, acc in enumerate(accuracies):
    axes[0, 0].text(i, acc + 0.3, f"{acc:.2f}%", ha='center', fontsize=10, fontweight='bold', color='black')

# --- Confusion Matrix Naive Bayes ---
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0, 1])
axes[0, 1].set_title("Confusion Matrix - Naive Bayes", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Prediksi")
axes[0, 1].set_ylabel("Sebenarnya")
axes[0, 1].set_xticklabels(['Asli', 'Palsu'])
axes[0, 1].set_yticklabels(['Asli', 'Palsu'], rotation=0)

# --- Confusion Matrix Logistic Regression ---
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix - Logistic Regression", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Prediksi")
axes[1, 0].set_ylabel("Sebenarnya")
axes[1, 0].set_xticklabels(['Asli', 'Palsu'])
axes[1, 0].set_yticklabels(['Asli', 'Palsu'], rotation=0)

# --- Confusion Matrix Decision Tree ---
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1, 1])
axes[1, 1].set_title("Confusion Matrix - Decision Tree", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Prediksi")
axes[1, 1].set_ylabel("Sebenarnya")
axes[1, 1].set_xticklabels(['Asli', 'Palsu'])
axes[1, 1].set_yticklabels(['Asli', 'Palsu'], rotation=0)

# --- Tata letak lebih rapi ---
plt.tight_layout()
plt.show()