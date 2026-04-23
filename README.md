# Professional Student Graduation Prediction System

Proyek ini menggunakan struktur standar industri (MLOps-ready) untuk membangun sistem prediksi kelulusan mahasiswa.

## Struktur Proyek
- **`data/raw/`**: Simpan dataset asli (`student-mat.csv`) di sini.
- **`src/`**: Kode sumber modular.
  - `data_loader.py`: Modul untuk membaca data.
  - `preprocess.py`: Feature engineering & pembersihan data.
  - `train.py`: Pelatihan model Logistic Regression & Decision Tree.
  - `predict.py`: Evaluasi model & penyimpanan metrik.
  - `logger.py` & `exception.py`: Logging dan penanganan error profesional.
- **`models/`**: Folder penyimpanan model yang sudah dilatih (`.pkl`).
- **`artifacts/`**: Log harian dan artefak proses lainnya.
- **`outputs/`**: Hasil visualisasi (`.png`) dan metrik (`metrics.csv`).
- **`main.py`**: Titik masuk utama untuk menjalankan seluruh pipeline.

## Cara Instalasi & Penggunaan
1.  **Persiapkan Environment**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Instalasi Package**:
    ```bash
    pip install -e .
    ```
3.  **Jalankan Pipeline**:
    ```bash
    python main.py
    ```

## Fitur Profesional
- **Logging**: Setiap langkah dicatat dalam folder `artifacts/logs/`.
- **Custom Exception**: Penanganan error yang mendetail (file, baris, pesan).
- **Modularitas**: Kode dipisahkan berdasarkan fungsinya untuk memudahkan testing dan maintenance.
