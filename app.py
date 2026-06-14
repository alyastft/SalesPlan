import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting System")

st.markdown("""
Sistem forecasting penjualan produk shock absorber PT. Kayaba Indonesia
menggunakan pendekatan:

- Machine Learning
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Extra Trees
    - Gradient Boosting
    - ElasticNet

- Deep Learning
    - BiLSTM

- Time Series
    - Prophet

---

### Menu

📊 Data Analysis

Menampilkan:

- Dataset Overview
- Missing Value Analysis
- Statistical Summary
- Trend Analysis
- Seasonality Analysis
- Product Classification
- External Factor Analysis
- Correlation Analysis

---

📈 Forecasting All Items

Menampilkan:

- Pemilihan model forecasting untuk setiap item
- Forecast seluruh item sekaligus
- Forecast per tahun
- Total forecast
- Export hasil forecast

---

### External Factors

Forecasting mempertimbangkan:

- Kurs USD/IDR
- Kurs JPY/IDR
- Produksi Motor Domestik
- Produksi Motor Ekspor
- Historical Sales
- Trend
- Seasonality
""")

st.info(
    "Pilih halaman melalui sidebar di sebelah kiri."
)

with open(
    "assets/template.xlsx",
    "rb"
) as file:

    st.download_button(
        label="📥 Download Template Excel",
        data=file,
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
