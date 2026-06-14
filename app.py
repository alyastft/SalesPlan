import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting System")

st.markdown("""
### PT Kayaba Indonesia

Forecasting penjualan shock absorber menggunakan:

- Machine Learning
- Deep Learning
- Time Series Forecasting

### External Factors

- Kurs USD/IDR
- Kurs JPY/IDR
- Produksi Motor Domestik
- Produksi Motor Ekspor

### Menu

📊 Data Analysis

📈 Forecasting All Items
""")

with open(
    "assets/template.xlsx",
    "rb"
) as file:

    st.download_button(
        label="📥 Download Template",
        data=file,
        file_name="template.xlsx"
    )

st.info(
    "Gunakan menu sidebar di sebelah kiri."
)

import streamlit as st

page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Data Analysis",
        "Forecasting All Items"
    ]
)
from pages.data_analysis import show_data_analysis

if page == "Home":
    show_home()

elif page == "Data Analysis":
    show_data_analysis()

elif page == "Forecasting All Items":
    show_forecasting()
