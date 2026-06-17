# 🩺 Smart Diabetes Risk Prediction System

Aplikasi berbasis Streamlit untuk memprediksi risiko diabetes menggunakan Machine Learning serta memberikan rekomendasi kesehatan personal menggunakan Large Language Model (LLM) Gemini.

---

## 📖 Deskripsi

Diabetes merupakan salah satu penyakit kronis yang terus mengalami peningkatan jumlah kasus setiap tahunnya. Deteksi risiko diabetes sejak dini dapat membantu masyarakat meningkatkan kesadaran terhadap kondisi kesehatan dan mendorong penerapan gaya hidup yang lebih sehat.

Smart Diabetes Risk Prediction System dikembangkan untuk membantu pengguna memperoleh estimasi risiko diabetes berdasarkan data kesehatan dan gaya hidup yang dimasukkan. Sistem memanfaatkan algoritma LightGBM sebagai model prediksi utama dan Gemini AI untuk menghasilkan rekomendasi kesehatan personal sesuai kondisi pengguna.

---

## ✨ Fitur Utama

### 🩺 Prediksi Risiko

Halaman utama untuk melakukan prediksi risiko diabetes berdasarkan data kesehatan pengguna.

### 📊 Performa Model

Menampilkan hasil evaluasi dan perbandingan performa lima model Machine Learning.

### 📈 Data Insight

Menyediakan visualisasi dan eksplorasi dataset diabetes.

### ℹ️ About

Menampilkan informasi aplikasi, dataset, fitur, serta panduan penggunaan.

---

## 🗂 Dataset

Dataset yang digunakan berasal dari:

**Diabetes Health Indicators Dataset**

Sumber asli data berasal dari:

**Behavioral Risk Factor Surveillance System (BRFSS) 2015**
oleh Centers for Disease Control and Prevention (CDC).

🔗 **Link Dataset:**
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv

Karakteristik dataset:

| Informasi             | Nilai                            |
| --------------------- | -------------------------------- |
| Jumlah Data           | 70.692                           |
| Jumlah Fitur Awal     | 21                               |
| Jumlah Fitur Terpilih | 11                               |
| Target                | Diabetes_binary                  |
| Kelas Target          | 0 = Tidak Diabetes, 1 = Diabetes |

---

## 🔍 Fitur yang Digunakan Model

Model LightGBM menggunakan 11 fitur hasil seleksi fitur Chi-Square:

1. PhysHlth
2. BMI
3. MentHlth
4. Age
5. GenHlth
6. HighBP
7. DiffWalk
8. HighChol
9. HeartDiseaseorAttack
10. Smoker
11. HvyAlcoholConsump

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/esajun/diabetes-prediction.git
cd diabetes-risk-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Buat file `.env`

```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

## ▶️ Menjalankan Aplikasi

```bash
streamlit run diabetes_app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

---

## 🚀 Live Demo

🔗 **Try the App:**
https://diabetes-prediction-app-8.streamlit.app/


---


## ⚠️ Disclaimer

Hasil prediksi yang diberikan oleh aplikasi ini hanya bersifat estimasi berdasarkan model Machine Learning dan tidak dapat digunakan sebagai diagnosis medis resmi.

Pengguna tetap disarankan untuk berkonsultasi dengan dokter atau tenaga kesehatan profesional untuk mendapatkan pemeriksaan dan diagnosis yang tepat.

---

## ❤️ Tujuan Pengembangan

Project ini dikembangkan sebagai bagian dari implementasi Machine Learning dan AI pada bidang kesehatan untuk membantu meningkatkan kesadaran masyarakat terhadap risiko diabetes melalui teknologi prediktif yang mudah diakses.


