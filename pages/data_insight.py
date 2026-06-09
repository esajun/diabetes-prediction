import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Data Insight", layout="wide")

# Custom CSS untuk Data Insight page
st.markdown("""
<style>
    [data-testid='stSidebarNav'] {display: none;}
    
    /* Page styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling untuk metrics */
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    h1, h2 {
        animation: slideDown 0.6s ease-out;
    }
    
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    animation: slideDown 0.6s ease-out;
">
    <h1 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🔍 Data Insight</h1>
    <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.9); font-size: 1em;">
        Eksplorasi dan visualisasi dataset diabetes secara detail
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists("diabetes.csv"):
        df = pd.read_csv("diabetes.csv").drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        return df
    return None

df = load_data()

if df is None:
    st.warning("❌ File `diabetes.csv` tidak ditemukan. Upload file untuk melihat visualisasi.")
    st.stop()

# Pastikan kolom target ada
if "Diabetes_binary" not in df.columns:
    st.error("❌ Kolom `Diabetes_binary` tidak ditemukan pada dataset.")
    st.stop()

# Summary Statistics dengan styling
st.markdown("<h2 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>📈 Ringkasan Data</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    ">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Total Sampel</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{df.shape[0]:,}</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    ">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Total Fitur</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{df.shape[1]}</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    perc = df["Diabetes_binary"].mean() * 100
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    ">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Diabetes %</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{perc:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    ">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Missing Data</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{df.isnull().sum().sum()}</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sampel Data dengan expander
with st.expander("📋 Lihat Sampel Data Awal (10 baris pertama)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True, height=300)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h2>📊 Statistik Deskriptif</h2>", unsafe_allow_html=True)
with st.expander("📈 Lihat Statistik Detail (Expand)", expanded=False):
    st.dataframe(df.describe(), use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h2>🎨 Visualisasi Data</h2>", unsafe_allow_html=True)

FEATURES = [
    'PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth',
    'HighBP', 'DiffWalk', 'Income', 'HighChol', 'HeartDiseaseorAttack'
]

# Visualisations dengan styling
tab1, tab2, tab3 = st.tabs(["📊 Distribusi", "🔥 Heatmap Korelasi", "💡 Insight"])

with tab1:
    st.markdown("<h4>Distribusi Semua Fitur Prediksi</h4>", unsafe_allow_html=True)
    for i in range(0, len(FEATURES), 2):
        cols = st.columns(2)
        for col, feature in zip(cols, FEATURES[i:i+2]):
            with col:
                st.markdown(f"<h5>{feature}</h5>", unsafe_allow_html=True)
                try:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    # Untuk fitur kategorikal gunakan countplot dengan pemisahan berdasarkan target biner
                    if feature in ["HighBP", "DiffWalk", "HighChol", "HeartDiseaseorAttack", "GenHlth", "Income", "Age"]:
                        sns.countplot(x=feature, hue='Diabetes_binary', data=df, palette=["#667eea", "#f59e0b"], ax=ax)
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Jumlah")
                        ax.legend(title='Diabetes', labels=['Tidak (0)', 'Ya (1)'])
                    else:
                        # Untuk fitur numerik gunakan histplot terpisah menurut target biner
                        sns.histplot(data=df, x=feature, hue='Diabetes_binary', bins=30, kde=True, palette=["#667eea", "#f59e0b"], ax=ax, alpha=0.6)
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Frekuensi")
                        ax.legend(title='Diabetes', labels=['Tidak (0)', 'Ya (1)'])
                    ax.set_title(f"Distribusi {feature}", fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error pada {feature}: {e}")

with tab2:
    st.markdown("<h4 style='text-align: center;'>Heatmap Korelasi Fitur Numerik</h4>", unsafe_allow_html=True)
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Korelasi'})
        ax.set_title("Heatmap Korelasi Semua Fitur Numerik", fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error: {e}")

with tab3:
    st.markdown("<h3>💡 Key Insights</h3>", unsafe_allow_html=True)
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        corr = df[num_cols].corr()
        
        # Diabetes distribution
        not_diab = (df["Diabetes_binary"] == 0).sum()
        diab = (df["Diabetes_binary"] == 1).sum()
        st.write(f"**📊 Distribusi Target:**")
        st.write(f"- Tidak Diabetes: {not_diab:,} ({not_diab/len(df)*100:.1f}%)")
        st.write(f"- Diabetes: {diab:,} ({diab/len(df)*100:.1f}%)")

        # Feature stats by diabetes
        st.write("**📌 Perbandingan Rata-rata Fitur antara Diabetes dan Tidak Diabetes:**")
        stats = df.groupby("Diabetes_binary")[FEATURES].mean().T
        stats.columns = ["Tidak Diabetes", "Diabetes"]
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        # Top correlations
        if "Diabetes_binary" in corr.columns:
            top_corr = corr["Diabetes_binary"].abs().sort_values(ascending=False)
            # Filter out Diabetes_binary itself and take top 5
            top_corr = top_corr[top_corr.index != "Diabetes_binary"].head(5)
            st.write(f"**🔗 Fitur dengan Korelasi Tertinggi terhadap Diabetes:**")
            for idx, (feat, val) in enumerate(top_corr.items(), 1):
                st.write(f"{idx}. `{feat}`: {val:.3f}")
    except Exception as e:
        st.warning(f"Gagal menghitung insight: {e}")

st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("📄 Lihat Tabel Data Lengkap", expanded=False):
    st.markdown(f"Total baris: {len(df)}")
    st.dataframe(df, use_container_width=True, height=400)

