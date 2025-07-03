# 🎓 Sistem Klasifikasi Aspirasi Mahasiswa (SUYA) dengan Random Forest  
📢 **NEW MODEL UPDATE!**: `suyafinal.ipynb` kini hadir dengan akurasi **89%**

## 🌐 Deployment Online

Aplikasi telah tersedia online melalui Hugging Face Spaces:

🔗 **[SUYA Demo Online (v4.0)](https://huggingface.co/spaces/johndenver654/demosuya4)**  
[![HuggingFace Spaces](https://img.shields.io/badge/Launch%20Demo-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/johndenver654/demosuya4)


---

## 📌 Deskripsi Proyek
Sistem klasifikasi otomatis untuk mengkategorikan aspirasi mahasiswa Fakultas Ilmu Komputer UDINUS ke dalam 5 kategori utama:
- Fasilitas
- Dosen
- Ormawa
- Pembelajaran
- Administrasi

Model awal menggunakan algoritma **Random Forest**, menghasilkan akurasi keseluruhan **82%** dan skor F1 **0.83**.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.11  
- **Library Utama**:
  - `scikit-learn==1.2.2`
  - `Sastrawi==1.0.1`
  - `gradio==3.x`
- **Infrastruktur**:
  - Google Colab
  - Hugging Face Spaces

---

## 📂 Struktur File

```
suya-classification/
├── app.py
├── suyastrong.py
├── suya_rf_model.pkl
├── suya_tfidf_vectorizer.pkl
├── suya_label_encoder.pkl
└── suya_prioritized_keywords.pkl
```

---

## 🔧 Instalasi

```bash
git clone https://github.com/[username]/suya-classification.git
cd suya-classification
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

### 🔍 Prediksi via Gradio

```bash
python app.py
```
Akses di `http://localhost:7860`

### 🎯 Training Model Baru

```bash
python suyastrong.py
```

---

## 📊 Performa Model Awal

| Kategori       | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Administrasi   | 0.86      | 0.83   | 0.85     |
| Dosen          | 0.71      | 0.80   | 0.75     |
| Fasilitas      | 0.90      | 0.90   | 0.90     |
| Ormawa         | 0.94      | 0.97   | 0.95     |
| Pembelajaran   | 0.69      | 0.60   | 0.64     |

**Akurasi Total**: **82%**


---

## 🧠 Alur Pemrosesan Teks

### Cleaning
- Case folding
- Menghapus tanda baca
- Stopword removal
- Stemming *(terkontrol)*

### Feature Engineering
- TF-IDF (4000 fitur, n-gram 1–3)
- Panjang teks
- Kata kunci prioritas

### Modeling
```python
RandomForestClassifier(
  n_estimators=500,
  max_depth=None,
  max_features='log2',
  min_samples_leaf=1,
  min_samples_split=2
)
```

---

## 🌐 Deployment
[![🤗 Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/[username]/suya-classification)

---

## 📝 Laporan Proyek
- 📄 Laporan Akhir (.pdf)
- 🎞️ Presentasi Proyek (.pptx)

---

## 👥 Kontributor
- **Ihda Nur Maulidia H.** (A11.2023.15394) — Ketua  
- **Ahmad Fikri P.B** (A11.2023.15466)  
- **Johan Akira Fatahillah** (A11.2018.10897)  
- **Claresta Nalla S.** (A11.2023.15365)  
- **Masayu Octa Faradisa** (A11.2023.15374)

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

# 🚀 SUYA Classifier Pipeline v2.0 (suyafinal.ipynb)  
*Enhanced & Optimized Text Classification Model*

Versi terbaru dari model awal dengan banyak perbaikan dan optimasi.

[![Download Notebook](https://img.shields.io/badge/Download-Notebook-blue)](suyafinal.ipynb)  
[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_username/your_repo/blob/main/suyafinal.ipynb)

---

## 🔍 1. Overview

- ✅ Akurasi: **89.3%** (F1: 0.89)
- 🛠️ Hyperparameter Tuning: GridSearchCV
- 🔄 Data Augmentasi untuk kelas minoritas
- 🧠 Feature Engineering: TF-IDF + Kata Kunci + Fitur Tambahan

---

## ⚙️ 2. Struktur Pipeline

```python
Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,3))),
        ('keywords', CountVectorizer(vocabulary=flat_keywords)),
        ('additional', AdditionalFeatures(prioritized_keywords))
    ])),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced_subsample'
    ))
])
```

---

## 📊 3. Performa Model

| Kategori      | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Administrasi  | 0.88      | 0.97   | 0.92     |
| Dosen         | 0.87      | 0.87   | 0.87     |
| Fasilitas     | 0.82      | 0.93   | 0.87     |
| Ormawa        | 0.97      | 0.93   | 0.95     |
| Pembelajaran  | 0.96      | 0.77   | 0.85     |

**Akurasi Total**: **89.3%**

---

## 🧪 4. Cara Penggunaan

```python
from joblib import load

model = load('suya_classifier_v2.joblib') 
prediksi = model.predict(["dosen tidak jelas menjelaskan materi"])
```
---

**© 2025 SUYA Dev Team**
