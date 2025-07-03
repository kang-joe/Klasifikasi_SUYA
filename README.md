# 🎓 Sistem Klasifikasi Aspirasi Mahasiswa (SUYA) dengan Random Forest


## 📌 Deskripsi Proyek
Sistem klasifikasi otomatis untuk mengkategorikan aspirasi mahasiswa Fakultas Ilmu Komputer UDINUS ke dalam 5 kategori utama:
- Fasilitas
- Dosen
- Ormawa
- Pembelajaran
- Administrasi

Model dikembangkan menggunakan algoritma **Random Forest**, menghasilkan akurasi keseluruhan **82%** dan skor F1 **0.83**.

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.11  
- **Library Utama**:
  - `scikit-learn==1.2.2`
  - `Sastrawi==1.0.1` (stemmer Bahasa Indonesia)
  - `gradio==3.x` (antarmuka pengguna)
- **Infrastruktur**:
  - Google Colab (untuk training model)
  - Hugging Face Spaces (untuk deployment)

## 📂 Struktur File

```
suya-classification/
├── app.py                         # Aplikasi Gradio untuk prediksi
├── suyastrong.py                 # Script training model
├── suya_rf_model.pkl             # Model Random Forest terlatih
├── suya_tfidf_vectorizer.pkl     # TF-IDF Vectorizer
├── suya_label_encoder.pkl        # Label Encoder
└── suya_prioritized_keywords.pkl # Kata kunci prioritas
```

## 🔧 Instalasi

1. **Clone repository:**
   ```bash
   git clone https://github.com/[username]/suya-classification.git
   cd suya-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Cara Menjalankan

### Prediksi via Gradio
```bash
python app.py
```
Aplikasi akan berjalan di `http://localhost:7860`

### Training Model Baru
```bash
python suyastrong.py
```

## 📊 Performa Model

| Kategori       | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Administrasi   | 0.86      | 0.83   | 0.85     |
| Dosen          | 0.71      | 0.80   | 0.75     |
| Fasilitas      | 0.90      | 0.90   | 0.90     |
| Ormawa         | 0.94      | 0.97   | 0.95     |
| Pembelajaran   | 0.69      | 0.60   | 0.64     |

**Akurasi Keseluruhan**: **82%**

## 🧠 Alur Pemrosesan Teks

### Cleaning
- Case folding
- Menghapus tanda baca
- Stopword removal
- Stemming *(dengan pengecualian pada istilah penting)*

### Feature Engineering
- TF-IDF (4000 fitur, n-gram 1–3)
- Panjang teks
- Kehadiran kata kunci prioritas
- Jumlah kata kunci

### Modeling
Random Forest dengan hyperparameter terbaik:
```python
{
  'max_depth': None,
  'max_features': 'log2',
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 500
}
```

## 🌐 Deployment

Aplikasi telah di-deploy di **Hugging Face Spaces**:  
[![🤗 Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/[username]/suya-classification)

## 📝 Laporan Lengkap
- Laporan Akhir (.pdf)
- Presentasi Proyek (.pptx)

## 👥 Kontributor
- **Ihda Nur Maulidia H.** (A11.2023.15394) — Ketua  
- **Ahmad Fikri P.B** (A11.2023.15466)  
- **Johan Akira Fatahillah** (A11.2018.10897)  
- **Claresta Nalla S.** (A11.2023.15365)  
- **Masayu Octa Faradisa** (A11.2023.15374)

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---
