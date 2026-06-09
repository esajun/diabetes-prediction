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
st.write("Eksplorasi dan visualisasi dataset diabetes secara detail.")
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

# ── Ringkasan ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sampel",        f"{df.shape[0]:,}")
c2.metric("Total Fitur",         f"{df.shape[1]}")
c3.metric("Prevalensi Diabetes", f"{df['Diabetes_binary'].mean()*100:.1f}%")
c4.metric("Missing Data",        f"{df.isnull().sum().sum()}")

st.divider()

with st.expander("📋 Lihat Sampel Data Awal (10 baris pertama)"):
    st.dataframe(df.head(10), use_container_width=True, height=300)

with st.expander("📈 Lihat Statistik Deskriptif"):
    st.dataframe(df.describe(), use_container_width=True)

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

