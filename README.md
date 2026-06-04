# 🩺 Diabetes Risk Prediction App

Aplikasi prediksi risiko diabetes berbasis Machine Learning menggunakan Streamlit.

## 📁 Struktur Project

```
diabetes_project/
├── diabetes_app.py          # Halaman utama (Prediksi)
├── pages/
│   ├── model_performance.py # Evaluasi model
│   ├── data_insight.py      # EDA & visualisasi
│   └── about.py             # Tentang proyek
├── diabetes.csv             # Dataset (tambahkan sendiri)
├── model_diabetes.pkl       # Model Random Forest (generate dari notebook)
├── scaler.pkl               # StandardScaler
├── selected_features.pkl    # Fitur terpilih
├── requirements.txt
├── pyproject.toml
└── .streamlit/
    └── config.toml
```

## 🚀 Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Atau pakai Poetry:
```bash
poetry install
poetry shell
```

### 2. Siapkan model

Jalankan notebook `.ipynb` terlebih dahulu untuk menghasilkan:
- `model_diabetes.pkl`
- `scaler.pkl`
- `selected_features.pkl`

### 3. Jalankan aplikasi

```bash
streamlit run diabetes_app.py
```

## 📊 Halaman Aplikasi

| Halaman | Deskripsi |
|---|---|
| 🏠 Prediksi Risiko | Input data kesehatan & lihat hasil prediksi |
| 📊 Performa Model | Perbandingan Logistic Regression vs Random Forest |
| 🔍 Data Insight | EDA: distribusi, faktor risiko, korelasi |
| ℹ️ Tentang Proyek | Deskripsi dataset, model, dan pipeline |

## 🤖 Model

- **Best Model:** Random Forest (200 estimators)
- **Dataset:** CDC BRFSS Diabetes Health Indicators
- **Fitur:** 21 variabel kesehatan & demografi
