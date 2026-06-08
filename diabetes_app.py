import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

# Custom CSS dengan animasi dan styling modern
st.markdown("""
<style>
    /* Hiding sidebar nav */
    [data-testid='stSidebarNav'] {display: none;}
    
    /* Gradient background */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainContainer"] {
        background: linear-gradient(135deg, #013c58 0%, #00537a 100%) !important;
    }
    
    /* Custom color scheme */
    :root {
        --primary-color: #00537a;
        --secondary-color: #013c58;
        --accent-color: #f5a201;
        --accent-light: #ffd35b;
        --text-on-accent: #013c58;
        --danger-color: #ef4444;
    }
    
    /* Title styling dengan animasi fade-in */
    h1 {
        font-size: 2.8em !important;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
        font-weight: 800 !important;
        animation: slideDown 0.6s ease-out;
        margin-bottom: 10px !important;
    }
    
    /* Subheader styling */
    h2 {
        color: #333 !important;
        border-bottom: 3px solid #667eea !important;
        padding-bottom: 10px !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #555 !important;
        font-weight: 600 !important;
    }
    
    /* Animasi keyframes */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Form styling */
    .stForm {
        background: #ffd35b !important;
        color: #013c58 !important;
        border-radius: 15px !important;
        padding: 30px !important;
        box-shadow: 0 10px 30px rgba(1,60,88,0.25) !important;
        animation: fadeIn 0.8s ease-out;
        border: none !important;
    }
    
    /* Input fields styling */
    .stNumberInput input, .stSelectbox select, .stSlider {
        border-radius: 8px !important;
        border: 2px solid #013c58 !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
        background: rgba(255,255,255,0.95) !important;
        color: #013c58 !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Button styling dengan hover effect */
    .stButton button {
        background: linear-gradient(135deg, #013c58 0%, #00537a 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(1, 60, 88, 0.35) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #00537a 0%, #013c58 100%) !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 12px !important;
        text-align: center !important;
        box-shadow: 0 5px 15px rgba(1,60,88,0.2) !important;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Divider styling */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
        margin: 30px 0 !important;
    }
    
    /* Caption and text styling */
    .caption {
        color: #666 !important;
        font-size: 0.95em !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: #ffd35b !important;
        color: #013c58 !important;
        border-radius: 10px !important;
        border-left: 5px solid #00537a !important;
        animation: slideDown 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

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
    st.page_link("diabetes_app.py",                label="🩺 Prediksi Risiko")
    st.page_link("pages/model_performance.py",     label="📊 Performa Model")
    st.page_link("pages/data_insight.py",          label="🔍 Data Insight")
    st.page_link("pages/about.py",                 label="ℹ️ About")

# Header dengan background container
st.markdown("""
<div style="
    background: linear-gradient(135deg, #013c58 0%, #00537a 100%);
    padding: 40px 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(1,60,88,0.22);
    animation: slideDown 0.6s ease-out;
">
    <h1 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🩺 Prediksi Risiko Diabetes</h1>
    <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.9); font-size: 1.1em;">
        Aplikasi AI untuk estimasi risiko diabetes berdasarkan data kesehatan Anda
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: #ffd35b;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #00537a;
    margin-bottom: 20px;
">
    <p>✓ Isi form di bawah dengan data kesehatan yang akurat</p>
    <p>✓ Model AI akan menganalisis risiko diabetes Anda</p>
    <p>✓ Dapatkan estimasi probabilitas dengan rekomendasi</p>
    <p style="margin-bottom: 0; color: #ef4444;"><strong>⚠️ Disclaimer:</strong> Hasil prediksi ini bukan diagnosis medis. Selalu konsultasikan dengan dokter profesional.</p>
</div>
""", unsafe_allow_html=True)

# Input Form dengan styling enhanced
st.markdown("<div style='margin-top: 30px;'><h2>📋 Formulir Pengisian Data</h2></div>", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    input_values = {}

    with col1:
        st.markdown("<h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>👤 Data Pribadi & Fisik</h3>", unsafe_allow_html=True)

        # Age
        usia = st.number_input("📅 Usia (tahun)", min_value=18, max_value=120, value=35)
        input_values["Age"] = age_to_category(int(usia))

        # BMI
        tinggi_cm = st.number_input("📏 Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
        berat_kg  = st.number_input("⚖️ Berat Badan (kg)",  min_value=20,  max_value=300, value=65)
        bmi_calc  = berat_kg / (tinggi_cm / 100) ** 2
        
        # BMI Color indicator
        if bmi_calc < 18.5:
            bmi_status = "🔵 Kurus"
            bmi_color = "#3b82f6"
        elif bmi_calc < 25:
            bmi_status = "🟢 Normal"
            bmi_color = "#10b981"
        elif bmi_calc < 30:
            bmi_status = "🟡 Berlebih"
            bmi_color = "#f59e0b"
        else:
            bmi_status = "🔴 Obese"
            bmi_color = "#ef4444"
        
        st.markdown(f"<p style='color: {bmi_color}; font-weight: bold; font-size: 1.1em;'>BMI: {bmi_calc:.1f} - {bmi_status}</p>", unsafe_allow_html=True)
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
            st.selectbox("💰 Pendapatan per Tahun", list(inc_map.keys()), index=4)
        ]

        # PhysHlth
        phys_days = st.slider("🏃 Hari kesehatan fisik terganggu (30 hari terakhir)", 0, 30, 0)
        input_values["PhysHlth"] = phys_days

        # MentHlth
        ment_days = st.slider("🧠 Hari kesehatan mental terganggu (30 hari terakhir)", 0, 30, 0)
        input_values["MentHlth"] = ment_days

    with col2:
        st.markdown("<h3 style='color: #764ba2; border-bottom: 2px solid #764ba2; padding-bottom: 10px;'>❤️ Kondisi Kesehatan</h3>", unsafe_allow_html=True)

        # GenHlth
        gen_map = {
            "Sangat Baik (Excellent)": 1,
            "Baik (Very Good)":        2,
            "Cukup (Good)":            3,
            "Kurang (Fair)":           4,
            "Buruk (Poor)":            5,
        }
        input_values["GenHlth"] = gen_map[
            st.selectbox("🏥 Kondisi Kesehatan Secara Umum", list(gen_map.keys()), index=1)
        ]

        # HighBP
        input_values["HighBP"] = st.selectbox(
            "🩸 Tekanan Darah Tinggi", [0, 1], format_func=lambda x: "✅ Ya" if x else "❌ Tidak")

        # HighChol
        input_values["HighChol"] = st.selectbox(
            "🧬 Kolesterol Tinggi", [0, 1], format_func=lambda x: "✅ Ya" if x else "❌ Tidak")

        # HeartDiseaseorAttack
        input_values["HeartDiseaseorAttack"] = st.selectbox(
            "💔 Pernah Penyakit / Serangan Jantung", [0, 1], format_func=lambda x: "✅ Ya" if x else "❌ Tidak")

        # DiffWalk
        input_values["DiffWalk"] = st.selectbox(
            "🚶 Kesulitan Berjalan / Naik Tangga", [0, 1], format_func=lambda x: "✅ Ya" if x else "❌ Tidak")

    st.markdown("")
    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        submitted = st.form_submit_button("🔍 Analisis Risiko Diabetes", use_container_width=True, type="primary")

# Prediction dengan hasil yang lebih menarik
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

    st.markdown("""
<div style='
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 15px;
    margin-top: 40px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    animation: slideDown 0.6s ease-out;
'>
    <h2 style='color: white; margin-top: 0;'>📊 Hasil Analisis Risiko Diabetes</h2>
</div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        # High Risk
        st.markdown(f"""
<div style='
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
    animation: pulse 2s ease-in-out infinite;
'>
    <h3 style='color: white; margin-top: 0;'>⚠️ RISIKO TINGGI DIABETES</h3>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.1em; margin: 15px 0;'>
        Model AI memprediksi Anda memiliki risiko tinggi terhadap diabetes
    </p>
    <div style='background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin-top: 15px;'>
        <p style='color: white; margin: 0; font-size: 1.3em; font-weight: bold;'>
            Probabilitas: <span style='font-size: 1.5em;'>{prob_pos:.1f}%</span>
        </p>
    </div>
</div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(int(prob_pos) / 100)
        
        st.markdown("""
<div style='
    background: #fef3c7;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #ef4444;
    margin-bottom: 20px;
'>
    <h4 style='color: #92400e; margin-top: 0;'>💡 Rekomendasi Penting:</h4>
    <ul style='color: #78350f; line-height: 1.8;'>
        <li><strong>Segera konsultasikan dengan dokter</strong> untuk pemeriksaan lebih lanjut</li>
        <li><strong>Atur pola makan:</strong> Kurangi gula, karbohidrat sederhana, dan lemak jenuh</li>
        <li><strong>Tingkatkan aktivitas fisik:</strong> Minimal 30 menit per hari, 5 hari seminggu</li>
        <li><strong>Pantau gula darah:</strong> Lakukan pemeriksaan rutin</li>
        <li><strong>Kurangi berat badan:</strong> Jika BMI di atas normal</li>
        <li><strong>Kelola stress:</strong> Meditasi, yoga, atau aktivitas yang menyenangkan</li>
    </ul>
</div>
        """, unsafe_allow_html=True)
        
    else:
        # Low Risk
        st.markdown(f"""
<div style='
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    animation: slideDown 0.6s ease-out;
'>
    <h3 style='color: white; margin-top: 0;'>✅ RISIKO RENDAH DIABETES</h3>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.1em; margin: 15px 0;'>
        Model AI memprediksi Anda memiliki risiko rendah terhadap diabetes
    </p>
    <div style='background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin-top: 15px;'>
        <p style='color: white; margin: 0; font-size: 1.3em; font-weight: bold;'>
            Probabilitas Tidak Diabetes: <span style='font-size: 1.5em;'>{prob_neg:.1f}%</span>
        </p>
    </div>
</div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(int(prob_neg) / 100)
        
        st.markdown("""
<div style='
    background: #d1fae5;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #10b981;
    margin-bottom: 20px;
'>
    <h4 style='color: #065f46; margin-top: 0;'>💡 Tetap Jaga Kesehatan:</h4>
    <ul style='color: #047857; line-height: 1.8;'>
        <li><strong>Pertahankan pola makan sehat</strong> dengan nutrisi seimbang</li>
        <li><strong>Olahraga teratur:</strong> Minimal 3x seminggu untuk menjaga kondisi</li>
        <li><strong>Konsumsi buah dan sayur</strong> setiap hari</li>
        <li><strong>Pemeriksaan kesehatan berkala</strong> untuk deteksi dini</li>
        <li><strong>Hindari kebiasaan buruk:</strong> Merokok dan alkohol berlebih</li>
        <li><strong>Istirahat cukup:</strong> 7-8 jam tidur setiap malam</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

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