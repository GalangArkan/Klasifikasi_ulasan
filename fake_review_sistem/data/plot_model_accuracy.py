import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 📂 Path file hasil akurasi
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
ACCURACY_FILE = os.path.join(MODEL_DIR, 'model_accuracy.json')

# 📥 Muat data akurasi
with open(ACCURACY_FILE, 'r') as f:
    accuracies = json.load(f)

# 🎨 Siapkan data
models = list(accuracies.keys())
values = list(accuracies.values())

# Ubah ke array numpy untuk kemudahan
values = np.array(values)

# 🎨 Warna tiap model
colors = ['#ff6666', '#66b3ff', '#99ff99']

# 🧭 Membuat plot
plt.figure(figsize=(8, 5))
bars = plt.bar(models, values, color=colors, edgecolor='black')

# Tambahkan label di atas bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 🧾 Pengaturan tampilan
plt.title('Perbandingan Akurasi Model Machine Learning', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Akurasi (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 💾 Simpan hasil grafik
output_path = os.path.join(os.path.dirname(__file__), 'grafik_akurasi_model.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"📊 Grafik akurasi model disimpan di: {output_path}")