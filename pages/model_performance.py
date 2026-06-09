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
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",            label="Prediksi Risiko")
    st.page_link("pages/model_performance.py", label="Performa Model")
    st.page_link("pages/data_insight.py",      label="Data Insight")
    st.page_link("pages/about.py",             label="About")

st.title("📊 Performa Model")
st.write("Evaluasi dan perbandingan 4 model: Logistic Regression, Decision Tree, Random Forest, dan XGBoost.")
st.divider()

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

# Best Model: XGBoost 
st.subheader("XGBoost (Best Model)")
xgb_row = hasil[hasil["Model"] == "XGBoost"].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{xgb_row['Accuracy']:.4f}")
c2.metric("Precision", f"{xgb_row['Precision']:.4f}")
c3.metric("Recall",    f"{xgb_row['Recall']:.4f}")
c4.metric("F1 Score",  f"{xgb_row['F1 Score']:.4f}")
c5.metric("ROC-AUC",   f"{xgb_row['ROC-AUC']:.4f}")

st.divider()

# Tabel & Grafik 
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Tabel Perbandingan")
    st.dataframe(hasil.set_index("Model").style.format("{:.4f}").highlight_max(axis=0, color="#1a4535"), use_container_width=True)

with col_b:
    st.subheader("Grafik Perbandingan Metrik")
    metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
    x = np.arange(len(hasil["Model"]))
    w = 0.25
    colors = ["#2563EB", "#8B5CF6", "#10B981"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, hasil[metric], w, label=metric, color=color, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(hasil["Model"], rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# Confusion Matrix 4 model 
st.subheader("Confusion Matrix")
cols = st.columns(4)
cmaps = ["Blues", "Oranges", "Greens", "Purples"]
for col, (name, y_pred), cmap in zip(cols, preds.items(), cmaps):
    with col:
        st.write(name)
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                    cmap=cmap, ax=ax, cbar=False)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.subheader("Fitur yang Digunakan")
st.code(", ".join(FEATURES))