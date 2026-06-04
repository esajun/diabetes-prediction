import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model_diabetes.pkl"):
        st.error("File model_diabetes.pkl tidak ditemukan.")
        st.stop()
    model  = joblib.load("model_diabetes.pkl")
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, scaler

model, scaler = load_model()

ALL_FEATURES = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
    "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 DiabetesRisk")
    st.caption("ML-Powered Prediction")
    st.divider()
    st.page_link("diabetes_app.py",                label="🏠 Prediksi Risiko")
    st.page_link("pages/model_performance.py",     label="📊 Performa Model")
    st.page_link("pages/data_insight.py",          label="🔍 Data Insight")
    st.page_link("pages/about.py",                 label="ℹ️ Tentang Proyek")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Prediksi Risiko Diabetes")
st.write("Isi data kesehatan di bawah ini, lalu klik **Prediksi** untuk melihat estimasi risiko diabetes.")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Kondisi Medis")
        HighBP    = st.selectbox("Tekanan Darah Tinggi",    [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        HighChol  = st.selectbox("Kolesterol Tinggi",       [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        CholCheck = st.selectbox("Cek Kolesterol (5 thn)",  [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        Stroke    = st.selectbox("Pernah Stroke",            [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        HeartDiseaseorAttack = st.selectbox("Penyakit Jantung", [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        BMI = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)

    with col2:
        st.subheader("Gaya Hidup")
        Smoker            = st.selectbox("Perokok",                  [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        PhysActivity      = st.selectbox("Aktif Fisik (30 hari)",    [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        Fruits            = st.selectbox("Konsumsi Buah/hari",       [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        Veggies           = st.selectbox("Konsumsi Sayur/hari",      [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        HvyAlcoholConsump = st.selectbox("Alkohol Berlebih",         [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        DiffWalk          = st.selectbox("Kesulitan Berjalan",       [0,1], format_func=lambda x: "Ya" if x else "Tidak")

    with col3:
        st.subheader("Demografi & Umum")
        AnyHealthcare = st.selectbox("Punya Asuransi Kesehatan",     [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        NoDocbcCost   = st.selectbox("Tidak ke Dokter krn Biaya",    [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        GenHlth  = st.slider("Kesehatan Umum (1=Excellent, 5=Poor)", 1, 5, 3)
        MentHlth = st.slider("Hari Kes. Mental Buruk (30hr)",        0, 30, 0)
        PhysHlth = st.slider("Hari Kes. Fisik Buruk (30hr)",         0, 30, 0)
        Sex       = st.selectbox("Jenis Kelamin",  [0,1], format_func=lambda x: "Pria" if x else "Wanita")
        Age       = st.slider("Kategori Usia (1=18-24 s/d 13=80+)", 1, 13, 5)
        Education = st.slider("Tingkat Pendidikan (1–6)", 1, 6, 4)
        Income    = st.slider("Tingkat Pendapatan (1–8)", 1, 8, 5)

    st.markdown("")
    submitted = st.form_submit_button("🔍 Prediksi Risiko Diabetes", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame([{
        "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck,
        "BMI": BMI, "Smoker": Smoker, "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack, "PhysActivity": PhysActivity,
        "Fruits": Fruits, "Veggies": Veggies, "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare, "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth, "MentHlth": MentHlth, "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk, "Sex": Sex, "Age": Age,
        "Education": Education, "Income": Income
    }])
    for col in ALL_FEATURES:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[ALL_FEATURES]

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
        st.error(f"### ⚠️ Risiko Tinggi Diabetes")
        st.metric("Probabilitas Diabetes", f"{prob_pos:.1f}%")
        st.progress(int(prob_pos))
        st.warning("💡 Segera konsultasikan dengan tenaga medis. Perhatikan pola makan, tingkatkan aktivitas fisik, dan pantau kadar gula darah secara rutin.")
    else:
        st.success(f"### ✅ Risiko Rendah Diabetes")
        st.metric("Probabilitas Tidak Diabetes", f"{prob_neg:.1f}%")
        st.progress(int(prob_neg))
        st.info("💡 Tetap jaga kesehatan! Pertahankan gaya hidup sehat dengan olahraga rutin dan pola makan seimbang.")

    with st.expander("📋 Lihat Data Input"):
        st.dataframe(input_data.T.rename(columns={0: "Nilai"}), use_container_width=True)
