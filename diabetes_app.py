import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from google import genai
from dotenv import load_dotenv

st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

FEATURES = [
    'PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth',
    'HighBP', 'DiffWalk', 'HighChol', 'HeartDiseaseorAttack',
    'Smoker', 'HvyAlcoholConsump'
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
    except Exception:
        return None, None

model = load_model()

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

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
        
        # Smoker
        input_values["Smoker"] = st.selectbox(
            "Pernah merokok >= 100 batang seumur hidup?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        # HvyAlcoholConsump
        input_values["HvyAlcoholConsump"] = st.selectbox(
            "Konsumsi alkohol berlebih?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")        

    st.markdown("")
    submitted = st.form_submit_button("Prediksi Risiko Diabetes", use_container_width=True, type="primary")

if submitted:
    input_data = pd.DataFrame([input_values])[FEATURES]

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
    else:
        st.success("### Risiko Rendah Diabetes")
        st.metric("Probabilitas Tidak Diabetes", f"{prob_neg:.1f}%")
        st.progress(int(prob_neg))

        # ===== Gemini Recommendation =====

        prompt = f"""
        Anda adalah asisten kesehatan.

        Data pengguna:
        - Usia asli: {usia} tahun
        - Usia kategori model: {input_values['Age']}
        - BMI: {input_values['BMI']}
        - Kesehatan fisik terganggu: {input_values['PhysHlth']} hari
        - Kesehatan mental terganggu: {input_values['MentHlth']} hari
        - Kondisi kesehatan umum: {input_values['GenHlth']}
        - Tekanan darah tinggi: {'Ya' if input_values['HighBP'] else 'Tidak'}
        - Kolesterol tinggi: {'Ya' if input_values['HighChol'] else 'Tidak'}
        - Riwayat penyakit jantung: {'Ya' if input_values['HeartDiseaseorAttack'] else 'Tidak'}
        - Kesulitan berjalan: {'Ya' if input_values['DiffWalk'] else 'Tidak'}
        - Perokok: {'Ya' if input_values['Smoker'] else 'Tidak'}
        - Konsumsi alkohol berlebih: {'Ya' if input_values['HvyAlcoholConsump'] else 'Tidak'}

        Hasil model:
        - Risiko diabetes: {'Tinggi' if prediction == 1 else 'Rendah'}
        - Probabilitas diabetes: {prob_pos:.1f}%

        Berikan:
        1. Ringkasan kondisi pengguna
        2. Faktor risiko utama yang terlihat
        3. Saran gaya hidup yang sesuai kondisi pengguna
        4. Disclaimer bahwa ini bukan diagnosis medis

        Gunakan bahasa Indonesia yang sederhana.
        Maksimal 40 kata.
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("🤖 Rekomendasi Personal")
            st.write(response.text)

        except Exception as e:
            st.warning(f"Gagal menghasilkan rekomendasi AI: {e}")

    with st.expander("Lihat Data Input"):
        st.dataframe(input_data.T.rename(columns={0: "Nilai"}), use_container_width=True)

    st.divider()
    accuracy, f1 = get_model_accuracy()
    if accuracy is not None:
        st.caption(
            f"ℹ️ **Tentang Model:** Prediksi menggunakan algoritma **LightGBM** "
            f"yang dilatih dari dataset CDC BRFSS 2015. "
            f"Akurasi model pada data uji: **{accuracy*100:.1f}%** · F1 Score: **{f1*100:.1f}%**. "
            f"Hasil ini merupakan estimasi statistik, bukan diagnosis medis."
        )
    else:
        st.caption(
            "ℹ️ **Tentang Model:** Prediksi menggunakan algoritma **LightGBM** "
            "yang dilatih dari dataset CDC BRFSS 2015. "
            "Hasil ini merupakan estimasi statistik, bukan diagnosis medis."
        )