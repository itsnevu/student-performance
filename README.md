#  Student Performance Prediction — ML Audit Pipeline (PERFECT VERSION)

Proyek ini adalah sistem audit Machine Learning end-to-end yang modular dan siap produksi untuk memprediksi kelulusan mahasiswa berdasarkan berbagai parameter demografis dan akademik.

---

##  Referensi & Dasar Audit
Sistem ini dibangun berdasarkan standar **Principal ML Engineer** dengan mematuhi 5 Fase Audit:
1. **Fase 1: Data Loading & EDA** — Deteksi profil data dan distribusi target.
2. **Fase 2: Data Preparation** — 7 teknik (Cleaning, Imputation, Transformation, Outlier, Splitting, Normalization, Imbalance).
3. **Fase 3: Feature Engineering** — 5 teknik (Addition, Extraction, Reduction, Selection, PCA).
4. **Fase 4: Modeling** — Audit perbandingan 10 model sekaligus (RF, XGBoost, SVM, KNN, dll) dengan Hyperparameter Tuning.
5. **Fase 5: Evaluation** — Dashboard performa (F1, Accuracy, Precision, Recall, AUC-ROC, dan CV Mean/Std).

---

##  Instalasi & Persiapan

### 1. Persiapkan Environment
Pastikan Anda memiliki Python 3.9+ terinstal.

### 2. Instal Dependencies
Jalankan perintah berikut untuk menginstal semua pustaka yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Struktur Folder
Pastikan dataset CSV Anda diletakkan di dalam folder:
`data/raw/`

---

## Cara Penggunaan

### A. Jalankan Audit Otomatis (Hasil Akhir)
Untuk menjalankan seluruh proses audit untuk semua dataset di `data/raw/` sekaligus:
```bash
python main.py
```
**Hasil akan tersimpan di:**
- `outputs/models/` — File `.pkl` untuk model dan preprocessor.
- `outputs/` — Grafik perbandingan (`.png`) dan tabel CSV hasil audit.

### B. Jalankan Audit Interaktif (Langkah-demi-Langkah)
Jika ingin melihat prosesnya secara detail di setiap fase, buka Notebook:
`notebooks/02-full-audit-pipeline.ipynb`

---

## Teknologi yang Digunakan
- **Core**: Python, Pandas, NumPy
- **ML Framework**: Scikit-Learn, XGBoost
- **Sampling**: Imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn
- **Persistence**: Joblib

---
**Status Audit:** ✅ PERFECT VERSION (Inference Ready, No Data Leakage)
