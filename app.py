import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import preprocess_data
from utils.eda import show_eda
from utils.classification import classify_items
from utils.forecasting import forecast_item
from utils.helper import RECOMMENDED_MODELS


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# HOME PAGE
# =====================================================

def home():

    st.title("📈 Sales Forecasting Dashboard")

    st.markdown("""
    ### Welcome

    Aplikasi ini digunakan untuk melakukan analisis dan forecasting
    penjualan produk berdasarkan data historical.

    ### Fitur Aplikasi

    #### 1. Data Analysis
    - Upload dataset penjualan
    - Melihat preview data
    - Exploratory Data Analysis (EDA)
    - Analisis tren penjualan
    - Klasifikasi produk menjadi:
        - Stable
        - Declining
        - Volatile
        - Intermittent
        - Discontinued

    #### 2. Forecasting
    Melakukan prediksi penjualan produk menggunakan berbagai metode:
    - XGBoost
    - LightGBM
    - CatBoost
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - ElasticNet
    - Prophet
    - BiLSTM

    Model yang direkomendasikan akan menyesuaikan karakteristik produk.
    """)

    st.info(
        "Silakan gunakan menu sidebar untuk berpindah halaman."
    )


# =====================================================
# FORECAST PAGE
# =====================================================

def forecasting_page():

    import io

    st.title("🔮 Batch Sales Forecasting")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"],
        key="forecast_file"
    )

    if uploaded_file is None:
        st.info("Upload dataset terlebih dahulu")
        return


    # ======================
    # LOAD DATA
    # ======================

    if uploaded_file.name.endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded_file)
    else:
        raw_df = pd.read_csv(uploaded_file)


    df = preprocess_data(raw_df)


    st.success(
        f"Dataset berhasil dimuat dengan {df['Model'].nunique()} produk"
    )


    # ======================
    # MODEL SELECTION
    # ======================

    models = [
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "ElasticNet",
        "Prophet",
        "BiLSTM"
    ]


    selected_model = st.selectbox(
        "Pilih Forecast Model",
        models
    )


    periods = st.slider(
        "Forecast Horizon (Month)",
        1,
        36,
        12
    )


    # ======================
    # RUN FORECAST
    # ======================

    if st.button("Generate All Forecast"):


        all_forecast = []


        progress = st.progress(0)

        products = df["Model"].unique()


        for i, product in enumerate(products):

            item_df = df[
                df["Model"] == product
            ].copy()


            result = forecast_item(
                item_df,
                selected_model,
                periods
            )


            if not result.empty:

                result["Model"] = product


                result["KYB No"] = (
                    item_df["KYB No"]
                    .iloc[0]
                )


                all_forecast.append(
                    result
                )


            progress.progress(
                (i + 1) / len(products)
            )


        if len(all_forecast) == 0:

            st.error(
                "Forecast gagal dibuat"
            )

            return


        forecast_df = pd.concat(
            all_forecast,
            ignore_index=True
        )


        # ======================
        # PREVIEW
        # ======================

        st.subheader(
            "Forecast Preview"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )


        # ======================
        # SUMMARY
        # ======================

        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Jumlah Produk",
            forecast_df["Model"].nunique()
        )


        col2.metric(
            "Total Forecast",
            f"{forecast_df['Forecast'].sum():,.0f}"
        )


        col3.metric(
            "Jumlah Data Forecast",
            len(forecast_df)
        )


        # ======================
        # MONTHLY FORECAST
        # ======================

        st.subheader(
            "Total Forecast per Month"
        )


        monthly = (
            forecast_df
            .groupby("Forecast Date")["Forecast"]
            .sum()
            .reset_index()
        )


        fig = px.line(
            monthly,
            x="Forecast Date",
            y="Forecast",
            markers=True
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ======================
        # EXPORT EXCEL
        # ======================

        output = io.BytesIO()


        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            forecast_df.to_excel(
                writer,
                index=False,
                sheet_name="Forecast"
            )


        output.seek(0)


        st.download_button(
            "📥 Download Forecast Excel",
            data=output,
            file_name=f"forecast_{selected_model}.xlsx",
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        )

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")


page = st.sidebar.radio(
    "Choose Page",
    [
        "Home",
        "Data Analysis",
        "Forecast"
    ]
)


# =====================================================
# ROUTER
# =====================================================

if page == "Home":

    home()


elif page == "Data Analysis":

    show_eda()


elif page == "Forecast":

    forecasting_page()
