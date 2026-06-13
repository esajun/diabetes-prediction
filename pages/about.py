import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",            label="Prediksi Risiko")
    st.page_link("pages/model_performance.py", label="Performa Model")
    st.page_link("pages/data_insight.py",      label="Data Insight")
    st.page_link("pages/about.py",             label="About")

st.title("ℹ️ Tentang Aplikasi")
st.write("Dapatkan wawasan kesehatan dengan prediksi risiko diabetes dan inspirasi gaya hidup sehat.")

st.header("📋 Tentang Aplikasi Prediksi Diabetes")
st.write(
    "Aplikasi ini dibuat untuk membantu Anda memahami potensi risiko diabetes dengan cara yang lebih personal dan positif. "
    "Dengan dukungan model **LightGBM**, aplikasi ini menganalisis data kesehatan Anda untuk memberikan gambaran yang jelas "
    "tentang kondisi Anda saat ini."
)
st.write(
    "Isi informasi seperti usia, berat badan, tinggi badan, tekanan darah, kolesterol, dan kebiasaan harian Anda. "
    "Hasilnya akan membantu Anda melihat area yang bisa ditingkatkan demi hidup lebih sehat, sekaligus memberi rekomendasi "
    "sederhana untuk mendukung perjalanan kesehatan Anda."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔬 Dataset & Model")
    st.markdown(
        "- **Dataset:** diabetes.csv dengan 70+ ribu data penderita dan non-penderita diabetes\n"
        "- **Target:** Kolom `Diabetes_binary` (0 = Tidak, 1 = Ya)\n"
        "- **Fitur:** 11 faktor kesehatan yang paling berpengaruh\n"
        "- **Model:** LightGBM dengan akurasi tinggi\n"
        "- **Performa:** Akurasi, Precision, Recall, F1-Score, dan ROC-AUC"
    )

with col2:
    st.subheader("✨ Fitur Utama")
    st.markdown(
        "- 🩺 **Prediksi Risiko** - Hitung risiko diabetes Anda secara real-time\n"
        "- 📊 **Performa Model** - Bandingkan 5 model machine learning\n"
        "- 🔍 **Data Insight** - Eksplorasi data dan visualisasi pola penting\n"
        "- 📈 **Statistik Detail** - Analisis mendalam tentang dataset\n"
        "- 💡 **Rekomendasi** - Saran kesehatan berdasarkan hasil prediksi"
    )

st.markdown("---")

st.subheader("🎯 Cara Menggunakan Aplikasi")
st.markdown(
    "1. **Buka Halaman Prediksi Risiko** - Dari menu sidebar, klik 🩺 Prediksi Risiko\n"
    "2. **Isi Formulir** - Masukkan data kesehatan Anda dengan seakurat mungkin di kedua kolom\n"
    "3. **Hitung Risiko** - Klik tombol 🔍 Analisis Risiko Diabetes\n"
    "4. **Baca Hasil** - Lihat estimasi probabilitas dan rekomendasi kesehatan\n"
    "5. **Eksplorasi Data** - Kunjungi halaman Data Insight untuk analisis lebih lanjut"
)

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("⚠️ Disclaimer Penting")
    st.write(
        "**Hasil prediksi ini hanya bersifat estimasi dan BUKAN diagnosis medis resmi.** "
        "Aplikasi ini tidak dapat menggantikan konsultasi dengan dokter atau tenaga medis profesional."
    )
    st.write(
        "Jika Anda memiliki keluhan kesehatan, merasa khawatir tentang risiko diabetes, atau memiliki gejala yang mencurigakan, "
        "**segera konsultasikan dengan dokter atau tenaga medis profesional** untuk diagnosis dan penanganan yang tepat."
    )

with col2:
    st.subheader("💪 Tips Kesehatan Umum")
    st.markdown(
        "- 🥗 **Pola Makan Sehat** - Konsumsi buah, sayur, dan makanan berserat tinggi\n"
        "- 🏃 **Olahraga Teratur** - Minimal 30 menit per hari, 5 hari seminggu\n"
        "- ⏰ **Istirahat Cukup** - Tidur 7-8 jam setiap malam\n"
        "- 💧 **Minum Air** - Penuhi kebutuhan hidrasi harian Anda\n"
        "- 🚭 **Hindari Rokok & Alkohol** - Jaga kesehatan organ vital"
    )

st.markdown("---")

st.subheader("📚 Informasi Lebih Lanjut")
st.write(
    "Untuk memahami faktor-faktor risiko diabetes lebih mendalam, kunjungi halaman **📊 Performa Model** "
    "untuk melihat metrik evaluasi dan membandingkan berbagai algoritma machine learning yang tersedia."
)
st.write(
    "Halaman **🔍 Data Insight** menyediakan visualisasi data, statistik deskriptif, dan analisis korelasi "
    "yang membantu Anda memahami pola dan hubungan antar faktor kesehatan dalam dataset."
)

st.markdown("---")

st.subheader("❤️ Kesehatan adalah Investasi Terbaik")
st.write(
    "Mulai dari sekarang, ambil langkah proaktif untuk menjaga kesehatan Anda. "
    "Gunakan aplikasi ini sebagai alat edukasi dan motivasi untuk gaya hidup yang lebih sehat."
)
