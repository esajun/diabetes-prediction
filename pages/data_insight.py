import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Data Insight", layout="wide")
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",            label="Prediksi Risiko")
    st.page_link("pages/model_performance.py", label="Performa Model")
    st.page_link("pages/data_insight.py",      label="Data Insight")
    st.page_link("pages/about.py",             label="About")

st.title("🔍 Data Insight")
st.write("Eksplorasi dan visualisasi dataset diabetes.")
st.divider()

@st.cache_data
def load_data():
    if os.path.exists("diabetes.csv"):
        df = pd.read_csv("diabetes.csv").drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        return df
    return None

df = load_data()

if df is None:
    st.warning("⚠️ File `diabetes.csv` tidak ditemukan.")
    st.stop()

if "Diabetes_binary" not in df.columns:
    st.error("❌ Kolom `Diabetes_binary` tidak ditemukan pada dataset.")
    st.stop()

FEATURES = [
    'PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth',
    'HighBP', 'DiffWalk', 'Income', 'HighChol', 'HeartDiseaseorAttack'
]

diabetes_count = (df["Diabetes_binary"] == 1).sum()
non_diabetes_count = (df["Diabetes_binary"] == 0).sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Sampel", f"{len(df):,}")
c2.metric("Total Fitur", f"{df.shape[1]}")
c3.metric(
    "Diabetes",
    f"{diabetes_count:,}",
    f"{diabetes_count/len(df)*100:.1f}%"
)
c4.metric(
    "Non-Diabetes",
    f"{non_diabetes_count:,}",
    f"{non_diabetes_count/len(df)*100:.1f}%"
)

st.divider()

st.subheader("📖 Penjelasan 10 Fitur yang Digunakan Model")
st.write("""
| Fitur | Penjelasan |
|---|---|
| PhysHlth | Jumlah hari kesehatan fisik terganggu (0-30) |
| BMI | Body Mass Index |
| MentHlth | Jumlah hari kesehatan mental terganggu (0-30) |
| Age | Kategori usia (1-13) |
| GenHlth | Kondisi kesehatan umum (1-5) |
| HighBP | Tekanan darah tinggi (0=Tidak, 1=Ya) |
| DiffWalk | Kesulitan berjalan (0=Tidak, 1=Ya) |
| Income | Tingkat pendapatan (1-8) |
| HighChol | Kolesterol tinggi (0=Tidak, 1=Ya) |
| HeartDiseaseorAttack | Riwayat penyakit jantung (0=Tidak, 1=Ya) |
""")

st.divider()

with st.expander("📋 Lihat Sampel Data Awal (10 baris pertama)"):
    st.dataframe(df.head(10), use_container_width=True, height=300)

with st.expander("📈 Lihat Statistik Deskriptif"):
    st.dataframe(df.describe(), use_container_width=True)

with st.expander("📖 Fitur Dataset"):
    st.write("""
| Fitur | Penjelasan |
|---|---|
| **Diabetes_binary** | Target: 0 = Tidak diabetes, 1 = Prediabetes/Diabetes |
| **HighBP** | Tekanan darah tinggi: 0 = Tidak, 1 = Ya |
| **HighChol** | Kolesterol tinggi: 0 = Tidak, 1 = Ya |
| **CholCheck** | Cek kolesterol dalam 5 tahun terakhir: 0 = Tidak, 1 = Ya |
| **BMI** | Body Mass Index |
| **Smoker** | Pernah merokok >= 100 batang seumur hidup: 0 = Tidak, 1 = Ya |
| **Stroke** | Pernah terkena stroke: 0 = Tidak, 1 = Ya |
| **HeartDiseaseorAttack** | Pernah penyakit jantung koroner/serangan jantung: 0 = Tidak, 1 = Ya |
| **PhysActivity** | Aktif fisik dalam 30 hari terakhir (di luar pekerjaan): 0 = Tidak, 1 = Ya |
| **Fruits** | Konsumsi buah >= 1x sehari: 0 = Tidak, 1 = Ya |
| **Veggies** | Konsumsi sayur >= 1x sehari: 0 = Tidak, 1 = Ya |
| **HvyAlcoholConsump** | Konsumsi alkohol berlebih (pria >=14/minggu, wanita >=7/minggu): 0 = Tidak, 1 = Ya |
| **AnyHealthcare** | Punya asuransi/jaminan kesehatan: 0 = Tidak, 1 = Ya |
| **NoDocbcCost** | Tidak ke dokter karena biaya dalam 12 bulan terakhir: 0 = Tidak, 1 = Ya |
| **GenHlth** | Kesehatan umum: 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor |
| **MentHlth** | Hari kesehatan mental buruk dalam 30 hari terakhir (0-30) |
| **PhysHlth** | Hari sakit/cedera fisik dalam 30 hari terakhir (0-30) |
| **DiffWalk** | Kesulitan berjalan/naik tangga: 0 = Tidak, 1 = Ya |
| **Sex** | Jenis kelamin: 0 = Perempuan, 1 = Laki-laki |
| **Age** | Kategori usia: 1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+ tahun |
| **Education** | Tingkat pendidikan: 1 = Tidak sekolah ... 6 = Sarjana ke atas |
| **Income** | Tingkat pendapatan: 1=<$10rb, 2=$10-15rb, 3=$15-20rb, 4=$20-25rb, 5=$25-35rb, 6=$35-50rb, 7=$50-75rb, 8=>$75rb per tahun |
""")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Distribusi", "🌡️ Heatmap Korelasi", "💡 Insight"])

with tab1:
    st.subheader("Distribusi Semua Fitur Prediksi")
    for i in range(0, len(FEATURES), 2):
        cols = st.columns(2)
        for col, feature in zip(cols, FEATURES[i:i+2]):
            with col:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    if feature in ["HighBP", "DiffWalk", "HighChol", "HeartDiseaseorAttack", "GenHlth", "Income", "Age"]:
                        sns.countplot(x=feature, hue="Diabetes_binary", data=df,
                                      palette=["#2563EB", "#EF4444"], ax=ax)
                        ax.set_ylabel("Jumlah")
                        ax.legend(title="Diabetes", labels=["Tidak (0)", "Ya (1)"])
                    else:
                        sns.histplot(data=df, x=feature, hue="Diabetes_binary", bins=30,
                                     kde=True, palette=["#2563EB", "#EF4444"], ax=ax, alpha=0.6)
                        ax.set_ylabel("Frekuensi")
                        ax.legend(title="Diabetes", labels=["Tidak (0)", "Ya (1)"])
                    ax.set_title(f"Distribusi {feature}", fontsize=11)
                    ax.set_xlabel(feature)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error pada {feature}: {e}")

with tab2:
    st.subheader("Heatmap Korelasi Semua Fitur")
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Heatmap Korelasi Fitur Numerik")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error: {e}")

with tab3:
    st.subheader("💡 Key Insights")
    try:
        not_diab = (df["Diabetes_binary"] == 0).sum()
        diab     = (df["Diabetes_binary"] == 1).sum()

        st.write("**📊 Distribusi Target:**")
        st.write(f"- Tidak Diabetes: {not_diab:,} ({not_diab/len(df)*100:.1f}%)")
        st.write(f"- Diabetes: {diab:,} ({diab/len(df)*100:.1f}%)")

        st.write("**📌 Rata-rata Fitur: Diabetes vs Tidak Diabetes:**")
        stats = df.groupby("Diabetes_binary")[FEATURES].mean().T
        stats.columns = ["Tidak Diabetes", "Diabetes"]
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        corr = df.corr(numeric_only=True)
        top_corr = corr["Diabetes_binary"].abs().drop("Diabetes_binary").sort_values(ascending=False).head(5)
        st.write("**🔗 Top 5 Fitur Berkorelasi dengan Diabetes:**")
        for i, (feat, val) in enumerate(top_corr.items(), 1):
            st.write(f"{i}. `{feat}`: {val:.3f}")
    except Exception as e:
        st.warning(f"Gagal menghitung insight: {e}")

st.divider()
with st.expander("📄 Lihat Tabel Data Lengkap"):
    st.write(f"Total baris: {len(df)}")
    st.dataframe(df, use_container_width=True, height=400)