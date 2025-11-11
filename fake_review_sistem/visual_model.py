import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ==============================
# Data akurasi tiap model
# ==============================
model_names = ["Naive Bayes", "Logistic Regression", "Decision Tree"]
accuracies = [96.67, 97.00, 98.00]

# ==============================
# Data Confusion Matrix (contoh)
# Ganti sesuai hasil aktual kamu kalau ada
# ==============================
# Format confusion matrix = [[True Negative, False Positive],
#                            [False Negative, True Positive]]

cm_nb = np.array([[134, 16], [20, 130]])   # Naive Bayes
cm_lr = np.array([[137, 13], [14, 136]])   # Logistic Regression
cm_dt = np.array([[142, 8], [6, 144]])     # Decision Tree

conf_matrices = [cm_nb, cm_lr, cm_dt]
titles = ["Confusion Matrix - Naive Bayes",
          "Confusion Matrix - Logistic Regression",
          "Confusion Matrix - Decision Tree"]

# ==============================
# Plot figure 2x2
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.4)

# ------------------------------
# Grafik Batang Akurasi
# ------------------------------
sns.barplot(x=accuracies, y=model_names, palette=["#FF5C8D", "#4A90E2", "#5CB85C"], ax=axes[0, 1])
axes[0, 1].set_xlim(90, 100)
axes[0, 1].set_xlabel("Akurasi (%)", fontsize=11)
axes[0, 1].set_title("Perbandingan Akurasi Model Machine Learning", fontsize=12, weight="bold")
for i, v in enumerate(accuracies):
    axes[0, 1].text(v - 1.5, i, f"{v:.2f}%", color="white", fontweight="bold")

# ------------------------------
# Confusion Matrix untuk tiap model
# ------------------------------
for i, ax in enumerate([axes[1, 0], axes[0, 0], axes[1, 1]]):  # atur posisi agar mirip skripsi
    sns.heatmap(conf_matrices[i], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(titles[i], fontsize=11, weight="bold")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.xaxis.set_ticklabels(["Palsu", "Asli"])
    ax.yaxis.set_ticklabels(["Palsu", "Asli"])

# ------------------------------
# Judul utama
# ------------------------------
fig.suptitle("Gambar 4.18 Perbandingan Akurasi dan Confusion Matrix Model Machine Learning", 
             fontsize=13, weight="bold", y=0.97)

# ------------------------------
# Simpan hasil
# ------------------------------
plt.tight_layout()
plt.savefig("perbandingan_model_dan_confusion_matrix.png", dpi=300)
plt.show()