import streamlit as st

from utils.eda import show_eda
from utils.forecasting import show_forecasting

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title(
    "Sales Forecasting"
)

menu = st.sidebar.radio(
    "Menu",
    [
        "📊 Data Analysis",
        "📈 Forecasting All Item"
    ]
)

st.sidebar.markdown("---")

with open(
    "assets/template.xlsx",
    "rb"
) as file:

    st.sidebar.download_button(
        label="📥 Download Template",
        data=file,
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Forecasting System

    External Factors:
    - USD/IDR
    - JPY/IDR
    - Motor Production Domestic
    - Motor Production Export
    """
)

# ==========================
# MAIN PAGE
# ==========================

if menu == "📊 Data Analysis":

    show_eda()

elif menu == "📈 Forecasting All Item":

    show_forecasting()
