import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Data Insight", layout="wide")
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",                label="Prediksi Risiko")
    st.page_link("pages/model_performance.py",     label="Performa Model")
    st.page_link("pages/data_insight.py",          label="Data Insight")
    st.page_link("pages/about.py",                 label="About")

st.title("🔍 Data Insight")
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
    st.warning("File `diabetes.csv` tidak ditemukan. Upload file untuk melihat visualisasi.")
    st.stop()

