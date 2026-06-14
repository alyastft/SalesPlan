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

    st.title("🔮 Sales Forecasting")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"],
        key="forecast_file"
    )

    if uploaded_file is None:

        st.info("Upload dataset terlebih dahulu")

        return


    # ==============================
    # LOAD DATA
    # ==============================

    if uploaded_file.name.endswith(".xlsx"):

        raw_df = pd.read_excel(uploaded_file)

    else:

        raw_df = pd.read_csv(uploaded_file)


    df = preprocess_data(raw_df)


    # ==============================
    # SELECT PRODUCT
    # ==============================

    products = sorted(
        df["Model"].unique()
    )

    selected_product = st.selectbox(
        "Pilih Product",
        products
    )


    item_df = df[
        df["Model"] == selected_product
    ].copy()


    st.subheader("Historical Sales")

    fig_history = px.line(
        item_df,
        x="ds",
        y="y",
        markers=True,
        title=f"Sales History - {selected_product}"
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )


    # ==============================
    # CLASSIFICATION
    # ==============================

    classification = classify_items(df)

    category = classification.loc[
        classification["Model"] == selected_product,
        "Category"
    ].values[0]


    st.subheader("Product Category")

    st.success(
        f"{selected_product} termasuk kategori: {category}"
    )


    # ==============================
    # MODEL SELECTION
    # ==============================

    recommended_models = RECOMMENDED_MODELS.get(
        category,
        []
    )

    st.subheader("Choose Forecast Model")

    model = st.selectbox(
        "Model",
        recommended_models
    )


    periods = st.slider(
        "Forecast Horizon (Month)",
        min_value=1,
        max_value=36,
        value=12
    )


    # ==============================
    # FORECAST BUTTON
    # ==============================

    if st.button("Generate Forecast"):


        with st.spinner(
            "Running forecasting model..."
        ):

            forecast_df = forecast_item(
                item_df,
                method=model,
                periods=periods
            )


        if forecast_df.empty:

            st.error(
                "Data tidak cukup untuk melakukan forecasting"
            )

            return


        # ==========================
        # RESULT TABLE
        # ==========================

        st.subheader(
            "Forecast Result"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )


        # ==========================
        # VISUALIZATION
        # ==========================

        history = item_df[
            ["ds", "y"]
        ].copy()

        history.columns = [
            "Date",
            "Sales"
        ]


        future = forecast_df.copy()

        future.columns = [
            "Date",
            "Sales"
        ]


        history["Type"] = "Historical"

        future["Type"] = "Forecast"


        plot_df = pd.concat(
            [
                history,
                future
            ]
        )


        fig = px.line(
            plot_df,
            x="Date",
            y="Sales",
            color="Type",
            markers=True,
            title="Historical vs Forecast"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        total_forecast = (
            forecast_df["Forecast"]
            .sum()
        )


        st.metric(
            "Total Forecast Sales",
            f"{total_forecast:,.0f}"
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
