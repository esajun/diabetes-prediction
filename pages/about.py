import streamlit as st

st.set_page_config(page_title="About", layout="wide")

# Custom CSS
st.markdown("""
<style>
    [data-testid='stSidebarNav'] {display: none;}

    .main {
        background: #f8fafc;
    }

    .about-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
        border-left: 6px solid #3b82f6;
    }

    .about-card h2,
    .about-card h3 {
        color: #0f172a;
    }

    .about-card p,
    .about-card li {
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",                label="🩺 Prediksi Risiko")
    st.page_link("pages/model_performance.py",     label="📊 Performa Model")
    st.page_link("pages/data_insight.py",          label="🔍 Data Insight")
    st.page_link("pages/about.py",                 label="ℹ️ About")

# Header
st.markdown("""
<div style="
    background: #0f172a;
    padding: 32px 24px;
    border-radius: 18px;
    margin-bottom: 26px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.1);
">
    <h1 style="margin: 0; color: white;">ℹ️ Tentang Aplikasi</h1>
    <p style="margin: 10px 0 0 0; color: #cbd5e1; font-size: 1em;">
        Dapatkan wawasan kesehatan dengan prediksi risiko diabetes dan inspirasi gaya hidup sehat.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <h2 style="margin-top: 0; color: #013c58;">📋 Tentang Aplikasi Prediksi Diabetes</h2>
    <p style="font-size: 1.05em; line-height: 1.8; color: #444;">
        Aplikasi ini dibuat untuk membantu Anda memahami potensi risiko diabetes dengan cara yang lebih personal dan positif.
        Dengan dukungan model <strong>XGBoost</strong>, aplikasi ini menganalisis data kesehatan Anda untuk memberikan gambaran yang jelas
        tentang kondisi Anda saat ini.
    </p>
    <p style="font-size: 1.05em; line-height: 1.8; color: #444;">
        Isi informasi seperti usia, berat badan, tinggi badan, tekanan darah, kolesterol, dan kebiasaan harian Anda.
        Hasilnya akan membantu Anda melihat area yang bisa ditingkatkan demi hidup lebih sehat, sekaligus memberi rekomendasi
        sederhana untuk mendukung perjalanan kesehatan Anda.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="about-card">
        <h3 style="margin-top: 0; color: #00537a;">🔬 Dataset & Model</h3>
        <ul style="color: #555; line-height: 2;">
            <li><strong>Dataset:</strong> diabetes.csv dengan 70+ ribu data penderita dan non-penderita diabetes</li>
            <li><strong>Target:</strong> Kolom <code>Diabetes_binary</code> (0 = Tidak, 1 = Ya)</li>
            <li><strong>Fitur:</strong> 10 faktor kesehatan yang paling berpengaruh</li>
            <li><strong>Model:</strong> XGBoost dengan akurasi tinggi</li>
            <li><strong>Performa:</strong> Akurasi, Precision, Recall, F1-Score, dan ROC-AUC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="about-card">
        <h3 style="margin-top: 0; color: #00537a;">✨ Fitur Utama</h3>
        <ul style="color: #555; line-height: 2;">
            <li>🩺 <strong>Prediksi Risiko</strong> - Hitung risiko diabetes Anda secara real-time</li>
            <li>📊 <strong>Performa Model</strong> - Bandingkan 4 algoritma machine learning</li>
            <li>🔍 <strong>Data Insight</strong> - Eksplorasi data dan visualisasi pola penting</li>
            <li>📈 <strong>Statistik Detail</strong> - Analisis mendalam tentang dataset</li>
            <li>💡 <strong>Rekomendasi</strong> - Saran kesehatan berdasarkan hasil prediksi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <h3 style="margin-top: 0; color: #00537a;">🎯 Cara Menggunakan Aplikasi</h3>
    <ol style="color: #555; line-height: 2.2; font-size: 1.05em;">
        <li><strong>Buka Halaman Prediksi Risiko</strong> - Dari menu sidebar, klik "🩺 Prediksi Risiko"</li>
        <li><strong>Isi Formulir</strong> - Masukkan data kesehatan Anda dengan seakurat mungkin di kedua kolom</li>
        <li><strong>Hitung Risiko</strong> - Klik tombol "🔍 Analisis Risiko Diabetes"</li>
        <li><strong>Baca Hasil</strong> - Lihat estimasi probabilitas dan rekomendasi kesehatan</li>
        <li><strong>Eksplorasi Data</strong> - Kunjungi halaman Data Insight untuk analisis lebih lanjut</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="about-card" style="border-left-color: #f5a201;">
        <h3 style="margin-top: 0; color: #013c58;">⚠️ Disclaimer Penting</h3>
        <p style="color: #555; line-height: 1.8;">
            <strong>Hasil prediksi ini hanya bersifat estimasi dan BUKAN diagnosis medis resmi.</strong> 
            Aplikasi ini tidak dapat menggantikan konsultasi dengan dokter atau tenaga medis profesional.
        </p>
        <p style="color: #555; line-height: 1.8;">
            Jika Anda memiliki keluhan kesehatan, merasa khawatir tentang risiko diabetes, atau memiliki gejala 
            yang mencurigakan, <strong>segera konsultasikan dengan dokter atau tenaga medis profesional</strong> untuk diagnosis dan penanganan yang tepat.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="about-card" style="border-left-color: #ffd35b;">
        <h3 style="margin-top: 0; color: #00537a;">💪 Tips Kesehatan Umum</h3>
        <ul style="color: #555; line-height: 2;">
            <li>🥗 <strong>Pola Makan Sehat</strong> - Konsumsi buah, sayur, dan makanan berserat tinggi</li>
            <li>🏃 <strong>Olahraga Teratur</strong> - Minimal 30 menit per hari, 5 hari seminggu</li>
            <li>⏰ <strong>Istirahat Cukup</strong> - Tidur 7-8 jam setiap malam</li>
            <li>💧 <strong>Minum Air</strong> - Penuhi kebutuhan hidrasi harian Anda</li>
            <li>🚭 <strong>Hindari Rokok & Alkohol</strong> - Jaga kesehatan organ vital</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <h3 style="margin-top: 0; color: #013c58;">📚 Informasi Lebih Lanjut</h3>
    <p style="color: #555; line-height: 1.8;">
        Untuk memahami faktor-faktor risiko diabetes lebih mendalam, kunjungi halaman <strong>📊 Performa Model</strong> 
        untuk melihat metrik evaluasi dan membandingkan berbagai algoritma machine learning yang tersedia.
    </p>
    <p style="color: #555; line-height: 1.8;">
        Halaman <strong>🔍 Data Insight</strong> menyediakan visualisasi data, statistik deskriptif, dan analisis korelasi 
        yang membantu Anda memahami pola dan hubungan antar faktor kesehatan dalam dataset.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: #e2e8f0;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #cbd5e1;
    text-align: center;
">
    <h3 style="margin-top: 0; color: #0f172a;">❤️ Kesehatan adalah Investasi Terbaik</h3>
    <p style="color: #475569; font-size: 1.05em;">
        Mulai dari sekarang, ambil langkah proaktif untuk menjaga kesehatan Anda.
        Gunakan aplikasi ini sebagai alat edukasi dan motivasi untuk gaya hidup yang lebih sehat.
    </p>
</div>
""", unsafe_allow_html=True)
