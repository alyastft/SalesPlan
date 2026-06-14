import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import preprocess_data
from utils.eda import show_eda
from utils.classification import classify_items
from utils.forecasting import (
    forecast_item,
    evaluate_model
)
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

    st.title("🔮 Multi Product Forecasting")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"],
        key="forecast"
    )

    if uploaded_file is None:
        st.info("Upload dataset terlebih dahulu")
        return


    # LOAD DATA
    if uploaded_file.name.endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded_file)
    else:
        raw_df = pd.read_csv(uploaded_file)


    df = preprocess_data(raw_df)


    # CLASSIFICATION
    classify_df = classify_items(df)


    st.subheader(
        "Model Selection Per Product"
    )

    selected_models = {}


    # SELECT MODEL UNTUK SETIAP ITEM
    for idx, row in classify_df.iterrows():

        model_name = row["Model"]
        category = row["Category"]

        recommendations = RECOMMENDED_MODELS.get(
            category,
            ["Random Forest"]
        )

        col1, col2, col3 = st.columns(
            [3, 2, 3]
        )

        with col1:
            st.write(
                f"**{model_name}**"
            )

        with col2:
            st.write(
                category
            )

        with col3:

            selected_models[model_name] = st.selectbox(
                "Model",
                recommendations,
                key=f"model_{model_name}",
                label_visibility="collapsed"
            )


    st.divider()


    periods = st.slider(
        "Forecast Horizon (Month)",
        1,
        36,
        12
    )


    # GENERATE FORECAST
    if st.button(
        "Generate Forecast Semua Produk"
    ):

        progress = st.progress(0)

        results = []


        products = classify_df["Model"].tolist()


        for i, product in enumerate(products):

            item_df = df[
                df["Model"] == product
            ].copy()


            method = selected_models[
                product
            ]
            
            
            # Hitung akurasi model
            mape = evaluate_model(
                item_df,
                method
            )
            
            
            # Forecast masa depan
            forecast = forecast_item(
                item_df,
                method,
                periods
            )


            if not forecast.empty:

                forecast["Model"] = product

                forecast["KYB No"] = (
                    item_df["KYB No"]
                    .iloc[0]
                )

                forecast["Category"] = (
                    classify_df.loc[
                        classify_df["Model"] == product,
                        "Category"
                    ].values[0]
                )

                forecast["Method"] = method
                forecast["MAPE (%)"] = mape


                results.append(
                    forecast
                )


            progress.progress(
                (i + 1) / len(products)
            )


        if len(results) == 0:

            st.error(
                "Tidak ada hasil forecast"
            )

            return


        final_forecast = pd.concat(
            results,
            ignore_index=True
        )


        # PREVIEW
        st.subheader(
            "Forecast Result"
        )

        st.subheader(
            "Total Forecast per Month"
        )
        
        monthly_forecast = (
            final_forecast
            .groupby("Forecast Date")["Forecast"]
            .sum()
            .reset_index()
        )
        
        fig = px.line(
            monthly_forecast,
            x="Forecast Date",
            y="Forecast",
            markers=True,
            title="Total Sales Forecast"
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        
        st.subheader(
            "Historical vs Forecast Per Product"
        )
        
        selected_chart_product = st.selectbox(
            "Choose Product",
            final_forecast["Model"].unique()
        )
        
        
        history = (
            df[
                df["Model"] == selected_chart_product
            ][["ds", "y"]]
            .copy()
        )
        
        history.columns = [
            "Date",
            "Sales"
        ]
        
        history["Type"] = "Historical"
        
        
        future = (
            final_forecast[
                final_forecast["Model"]
                == selected_chart_product
            ][["Forecast Date", "Forecast"]]
            .copy()
        )
        
        future.columns = [
            "Date",
            "Sales"
        ]
        
        future["Type"] = "Forecast"
        
        
        plot_df = pd.concat(
            [
                history,
                future
            ]
        )
        
        
        fig_item = px.line(
            plot_df,
            x="Date",
            y="Sales",
            color="Type",
            markers=True,
            title=f"Sales Forecast - {selected_chart_product}"
        )
        
        
        st.plotly_chart(
            fig_item,
            use_container_width=True
        )

        # SUMMARY
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Product",
            final_forecast["Model"].nunique()
        )

        c2.metric(
            "Total Forecast Sales",
            f"{final_forecast['Forecast'].sum():,.0f}"
        )

        c3.metric(
            "Total Rows",
            len(final_forecast)
        )


        final_forecast["Forecast"] = (
            final_forecast["Forecast"]
            .astype(int)
        )
        final_forecast["MAPE (%)"] = (
            pd.to_numeric(
                final_forecast["MAPE (%)"],
                errors="coerce"
            )
            .round(2)
        )

        final_forecast = final_forecast[
            [
                "Model",
                "KYB No",
                "Category",
                "Method",
                "MAPE (%)",
                "Forecast Date",
                "Forecast"
            ]
        ]
        # EXPORT EXCEL
        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            final_forecast.to_excel(
                writer,
                index=False,
                sheet_name="Forecast"
            )


        output.seek(0)


        st.download_button(
            "📥 Download Forecast Excel",
            data=output,
            file_name="forecast_all_products.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
