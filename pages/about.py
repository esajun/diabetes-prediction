import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("diabetes_app.py",                label="Prediksi Risiko")
    st.page_link("pages/model_performance.py",     label="Performa Model")
    st.page_link("pages/data_insight.py",          label="Data Insight")
    st.page_link("pages/about.py",                 label="About")

st.title("ℹ️ About")
st.divider()

