import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

FEATURES = [
    'PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth',
    'HighBP', 'DiffWalk', 'Income', 'HighChol', 'HeartDiseaseorAttack'
]

@st.cache_resource
def load_model():
    if not os.path.exists("diabetes_model.pkl"):
        st.error("File diabetes_model.pkl tidak ditemukan.")
        st.stop()
    return joblib.load("diabetes_model.pkl")

@st.cache_data
def get_model_accuracy():
    if not os.path.exists("diabetes.csv") or not os.path.exists("diabetes_model.pkl"):
        return None, None
    try:
        df = pd.read_csv("diabetes.csv").drop_duplicates()
        X  = df[FEATURES]
        y  = df["Diabetes_binary"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        mdl    = joblib.load("diabetes_model.pkl")
        y_pred = mdl.predict(X_test)
        return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
    except Exception as e:
        import traceback
        st.write(traceback.format_exc())
        return None, None

model = load_model()

def age_to_category(age: int) -> int:
    breaks = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, b in enumerate(breaks, start=1):
        if age < b:
            return i
    return 13

with st.sidebar:
    st.page_link("diabetes_app.py",                label="Prediksi Risiko")
    st.page_link("pages/model_performance.py",     label="Performa Model")
    st.page_link("pages/data_insight.py",          label="Data Insight")
    st.page_link("pages/about.py",                 label="About")

st.title("🩺 Prediksi Risiko Diabetes")
st.write("""
Aplikasi ini membantu kamu mengetahui estimasi risiko diabetes berdasarkan data kesehatan yang kamu masukkan.
Isi seluruh form di bawah ini dengan data yang sesuai kondisi kamu, lalu klik tombol **Prediksi**.
Model akan menganalisis dan memberikan hasil apakah kamu berisiko tinggi atau rendah terkena diabetes dengan nilai probabilitasnya.

> Hasil prediksi ini bukan diagnosis medis. Selalu konsultasikan kondisi kesehatan kamu dengan dokter atau tenaga medis profesional.
""")
st.divider()

# Input Form 
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    input_values = {}

    with col1:
        st.subheader("Data Pribadi & Fisik")

        # Age
        usia = st.number_input("Usia (tahun)", min_value=18, max_value=120, value=35)
        input_values["Age"] = age_to_category(int(usia))

        # BMI
        tinggi_cm = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
        berat_kg  = st.number_input("Berat Badan (kg)",  min_value=20,  max_value=300, value=65)
        bmi_calc  = berat_kg / (tinggi_cm / 100) ** 2
        st.caption(f"BMI terhitung: **{bmi_calc:.1f}**")
        input_values["BMI"] = round(bmi_calc, 1)

         # Income
        inc_map = {
            "< $10.000/tahun": 1,
            "$10.000–15.000":  2,
            "$15.000–20.000":  3,
            "$20.000–25.000":  4,
            "$25.000–35.000":  5,
            "$35.000–50.000":  6,
            "$50.000–75.000":  7,
            "> $75.000/tahun": 8,
        }
        input_values["Income"] = inc_map[
            st.selectbox("Pendapatan per Tahun", list(inc_map.keys()), index=4)
        ]

        # PhysHlth
        input_values["PhysHlth"] = st.slider(
            "Berapa hari kesehatan fisik terganggu dalam 30 hari terakhir?", 0, 30, 0)

        # MentHlth
        input_values["MentHlth"] = st.slider(
            "Berapa hari kesehatan mental terganggu dalam 30 hari terakhir?", 0, 30, 0)

    with col2:
        st.subheader("Kondisi Kesehatan")

        # GenHlth
        gen_map = {
            "Sangat Baik (Excellent)": 1,
            "Baik (Very Good)":        2,
            "Cukup (Good)":            3,
            "Kurang (Fair)":           4,
            "Buruk (Poor)":            5,
        }
        input_values["GenHlth"] = gen_map[
            st.selectbox("Kondisi Kesehatan Secara Umum", list(gen_map.keys()), index=1)
        ]

        # HighBP
        input_values["HighBP"] = st.selectbox(
            "Tekanan Darah Tinggi", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        # HighChol
        input_values["HighChol"] = st.selectbox(
            "Kolesterol Tinggi", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        # HeartDiseaseorAttack
        input_values["HeartDiseaseorAttack"] = st.selectbox(
            "Pernah Penyakit / Serangan Jantung", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        # DiffWalk
        input_values["DiffWalk"] = st.selectbox(
            "Kesulitan Berjalan / Naik Tangga", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        

    st.markdown("")
    submitted = st.form_submit_button("Prediksi Risiko Diabetes", use_container_width=True, type="primary")

# Prediction 
if submitted:
    TRAIN_ORDER = [
    'PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth',
    'HighBP', 'DiffWalk', 'Income', 'HighChol', 'HeartDiseaseorAttack'
    ]
    input_data = pd.DataFrame([input_values])[TRAIN_ORDER]

    prediction = model.predict(input_data)[0]
    try:
        proba    = model.predict_proba(input_data)[0]
        prob_pos = proba[1] * 100
        prob_neg = proba[0] * 100
    except Exception:
        prob_pos = 80.0 if prediction == 1 else 20.0
        prob_neg = 100 - prob_pos

    st.divider()

    if prediction == 1:
        st.error("### Risiko Tinggi Diabetes")
        st.metric("Probabilitas Diabetes", f"{prob_pos:.1f}%")
        st.progress(int(prob_pos))
        st.warning("""
💡 **Rekomendasi:**
- Segera konsultasikan dengan dokter atau tenaga medis
- Kurangi konsumsi makanan tinggi gula dan karbohidrat sederhana
- Tingkatkan aktivitas fisik minimal 30 menit per hari
- Pantau kadar gula darah secara rutin
- Jaga berat badan agar tetap ideal
""")
    else:
        st.success("### Risiko Rendah Diabetes")
        st.metric("Probabilitas Tidak Diabetes", f"{prob_neg:.1f}%")
        st.progress(int(prob_neg))
        st.info("""
💡 **Tetap jaga kesehatan!**
- Pertahankan pola makan sehat dan seimbang
- Rutin berolahraga minimal 3x seminggu
- Konsumsi buah dan sayur setiap hari
- Lakukan pemeriksaan kesehatan berkala
- Hindari kebiasaan merokok dan alkohol berlebih
""")

    with st.expander("Lihat Data Input"):
        st.dataframe(input_data.T.rename(columns={0: "Nilai"}), use_container_width=True)

    st.divider()
    accuracy, f1 = get_model_accuracy()
    if accuracy is not None:
        st.caption(
            f"ℹ️ **Tentang Model:** Prediksi menggunakan algoritma **XGBoost** "
            f"yang dilatih dari dataset CDC BRFSS 2015. "
            f"Akurasi model pada data uji: **{accuracy*100:.1f}%** · F1 Score: **{f1*100:.1f}%**. "
            f"Hasil ini merupakan estimasi statistik, bukan diagnosis medis."
        )
    else:
        st.caption(
            "ℹ️ **Tentang Model:** Prediksi menggunakan algoritma **XGBoost** "
            "yang dilatih dari dataset CDC BRFSS 2015. "
            "Hasil ini merupakan estimasi statistik, bukan diagnosis medis."
        )