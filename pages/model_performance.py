import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Performa Model", layout="wide")

# Custom CSS untuk Model Performance page
st.markdown("""
<style>
    [data-testid='stSidebarNav'] {display: none;}
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric card styling */
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
    
    h1, h2 {
        animation: slideDown 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",            label="🩺 Prediksi Risiko")
    st.page_link("pages/model_performance.py", label="📊 Performa Model")
    st.page_link("pages/data_insight.py",      label="🔍 Data Insight")
    st.page_link("pages/about.py",             label="ℹ️ About")

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
    <h1 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📊 Performa Model</h1>
    <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.9); font-size: 1em;">
        Evaluasi lengkap dan perbandingan 4 model machine learning untuk prediksi diabetes
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='background: #d1fae5; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;'>✨ Perbandingan: <strong>Logistic Regression</strong> vs <strong>Decision Tree</strong> vs <strong>Random Forest</strong> vs <strong>XGBoost</strong></p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

FEATURES = [
    'PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth',
    'HighBP', 'DiffWalk', 'Income', 'HighChol', 'HeartDiseaseorAttack'
]

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv").drop_duplicates() if os.path.exists("diabetes.csv") else None

@st.cache_resource
def train_all_models(dummy):
    df = load_data()
    if df is None:
        return None, None, None, None, None, None

    X = df[FEATURES]
    y = df["Diabetes_binary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
        "XGBoost":             joblib.load("diabetes_model.pkl") if os.path.exists("diabetes_model.pkl") else XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    preds   = {}
    for name, m in models.items():
        if name != "XGBoost":
            m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1 Score":  f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
        }
        preds[name] = y_pred

    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"}), preds, y_test

df = load_data()

if df is None:
    st.warning("File `diabetes.csv` tidak ditemukan.")
    st.stop()

if not os.path.exists("diabetes_model.pkl"):
    st.warning("File `diabetes_model.pkl` tidak ditemukan.")
    st.stop()

hasil, preds, y_test = train_all_models("run")

# Best Model: XGBoost dengan styling
st.markdown("""
<div style="
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
">
    <h2 style="color: white; margin-top: 0;">🏆 Model Terbaik: XGBoost</h2>
    <p style="color: rgba(255,255,255,0.9); margin: 0;">Model yang digunakan untuk prediksi pada halaman Prediksi Risiko</p>
</div>
""", unsafe_allow_html=True)

xgb_row = hasil[hasil["Model"] == "XGBoost"].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)

metric_style = """
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
"""

with c1:
    st.markdown(f"""
    <div style="{metric_style}">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Accuracy</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{xgb_row['Accuracy']:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div style="{metric_style}">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Precision</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{xgb_row['Precision']:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="{metric_style}">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Recall</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{xgb_row['Recall']:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div style="{metric_style}">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">F1 Score</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{xgb_row['F1 Score']:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div style="{metric_style}">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">ROC-AUC</p>
        <h3 style="margin: 10px 0 0 0; font-size: 2em;">{xgb_row['ROC-AUC']:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h2>📈 Analisis Perbandingan Model</h2>", unsafe_allow_html=True)

# Tabel & Grafik 
col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown("<h3>📊 Tabel Perbandingan Metrik</h3>", unsafe_allow_html=True)
    st.dataframe(
        hasil.set_index("Model").style.format("{:.4f}").highlight_max(axis=0, color="#d1fae5"),
        use_container_width=True
    )

with col_b:
    st.markdown("<h3>📈 Grafik Perbandingan</h3>", unsafe_allow_html=True)
    metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
    x = np.arange(len(hasil["Model"]))
    w = 0.25
    colors = ["#667eea", "#764ba2", "#10b981"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, hasil[metric], w, label=metric, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels(hasil["Model"], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("<hr>", unsafe_allow_html=True)

# Confusion Matrix 4 model dengan styling
st.markdown("<h2>🔢 Confusion Matrix - Perbandingan 4 Model</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; margin-bottom: 20px;'>Matriks kesalahan untuk setiap model (True Negative | False Positive || False Negative | True Positive)</p>", unsafe_allow_html=True)

cols = st.columns(4)
cmaps = ["Blues", "RdYlGn_r", "Oranges", "Purples"]
model_names = list(preds.keys())

for col, (name, y_pred), cmap in zip(cols, preds.items(), cmaps):
    with col:
        st.markdown(f"<h4 style='text-align: center; color: #667eea;'>{name}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3.5, 3))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False, 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax.set_xlabel("Prediksi", fontsize=9, fontweight='bold')
        ax.set_ylabel("Aktual", fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h2>🎯 Fitur Input Model</h2>", unsafe_allow_html=True)
st.markdown("""
<div style='
    background: #f3f4f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
'>
    <p style='color: #333; margin: 0; line-height: 1.8;'>
        <strong>Total 10 fitur yang digunakan untuk prediksi:</strong><br>
""" + ", ".join([f"<code>{feat}</code>" for feat in FEATURES]) + """
    </p>
</div>
""", unsafe_allow_html=True)