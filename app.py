import streamlit as st
import pandas as pd
import plotly.express as px
import io

from utils.preprocessing import preprocess_data
from utils.eda import show_eda
from utils.classification import classify_items
from utils.forecasting import (
    forecast_item,
    evaluate_model
)
from utils.helper import RECOMMENDED_MODELS


# PAGE CONFIG

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# HOME PAGE

def home():

    st.title("📈 Sales Forecasting Dashboard")

    st.markdown("""
    ## Welcome

    Aplikasi ini digunakan untuk melakukan analisis
    dan forecasting penjualan produk berdasarkan data historical.

    ### Menu aplikasi

    ### 📊 Data Analysis
    - Upload dataset
    - Preview data
    - Exploratory Data Analysis (EDA)
    - Analisis tren penjualan
    - Klasifikasi produk:
        - Stable
        - Declining
        - Volatile
        - Intermittent
        - Discontinued

    ### 🔮 Forecasting
    - Forecast seluruh produk sekaligus
    - Setiap produk dapat memilih model berbeda
    - Model mengikuti rekomendasi kategori produk
    - Perhitungan MAPE setiap produk
    - Export hasil forecast ke Excel
    """)


    st.info(
        "Gunakan menu sidebar untuk berpindah halaman."
    )


# FORECAST PAGE

def forecasting_page():

    st.title(
        "🔮 Multi Product Forecasting"
    )


    # Menyimpan hasil agar tidak hilang saat Streamlit rerun
    if "forecast_result" not in st.session_state:

        st.session_state.forecast_result = None


    # Upload Dataset

    uploaded_file = st.file_uploader(
        "Upload Sales Dataset",
        type=[
            "csv"
        ],
        key="forecast_file"
    )


    if uploaded_file is None:

        st.info(
            "Silakan upload dataset terlebih dahulu"
        )

        return


    # Load Dataset
    
    try:

        if uploaded_file.name.endswith(".xlsx"):

            raw_df = pd.read_excel(
                uploaded_file
            )

        else:

            raw_df = pd.read_csv(
                uploaded_file
            )


    except Exception as e:

        st.error(
            f"Gagal membaca file: {e}"
        )

        return


    # Preprocessing
    
    try:

        df = preprocess_data(
            raw_df
        )

    except Exception as e:

        st.error(
            f"Preprocessing gagal: {e}"
        )

        return


    st.success(
        f"Dataset berhasil dimuat: "
        f"{df['Model'].nunique()} produk "
        f"dengan {len(df)} data sales"
    )


    # Classification
    
    try:

        classify_df = classify_items(
            df
        )

    except Exception as e:

        st.error(
            f"Klasifikasi gagal: {e}"
        )

        return


    st.subheader(
        "📌 Model Selection Per Product"
    )


    st.caption(
        "Pilih model forecasting untuk setiap produk "
        "berdasarkan rekomendasi kategori."
    )


    selected_models = {}


    # Select Model Setiap Item
    
    for _, row in classify_df.iterrows():

        product = row["Model"]

        category = row["Category"]


        recommendations = RECOMMENDED_MODELS.get(
            category,
            ["Random Forest"]
        )


        col1, col2, col3 = st.columns(
            [
                4,
                2,
                4
            ]
        )


        with col1:

            st.write(
                f"**{product}**"
            )


        with col2:

            st.write(
                category
            )


        with col3:

            selected_models[product] = st.selectbox(
                label="Model",
                options=recommendations,
                key=f"model_{product}",
                label_visibility="collapsed"
            )


    st.divider()


    periods = st.slider(
        "Forecast Horizon (Month)",
        min_value=1,
        max_value=36,
        value=12
    )

    # GENERATE FORECAST
    
    if st.button(
        "🚀 Generate Forecast Semua Produk",
        type="primary"
    ):

        results = []

        products = classify_df["Model"].tolist()

        total_products = len(products)


        # Progress bar
        progress_bar = st.progress(0)

        status_text = st.empty()


        for i, product in enumerate(products):

            item_df = (
                df[
                    df["Model"] == product
                ]
                .copy()
            )


            method = selected_models[product]


            status_text.info(
                f"""
                Processing {i+1}/{total_products}

                Product: {product}

                Model: {method}
                """
            )


            try:

                # Hitung MAPE (Backtesting)
               
                mape = evaluate_model(
                    item_df,
                    method
                )


                # Forecast Future
               
                forecast = forecast_item(
                    item_df,
                    method,
                    periods
                )


                # Jika model tidak menghasilkan output
                if forecast.empty:

                    st.warning(
                        f"{product} tidak memiliki hasil forecast."
                    )

                    continue


                # Tambahkan informasi produk
               
                forecast["Model"] = product


                forecast["KYB No"] = (
                    item_df["KYB No"]
                    .iloc[0]
                )


                forecast["Category"] = (
                    classify_df.loc[
                        classify_df["Model"] == product,
                        "Category"
                    ]
                    .iloc[0]
                )


                forecast["Method"] = method


                forecast["MAPE (%)"] = mape


                # Hilangkan decimal forecast
                forecast["Forecast"] = (
                    forecast["Forecast"]
                    .fillna(0)
                    .round()
                    .astype(int)
                )


                results.append(
                    forecast
                )


            except Exception as e:

                st.error(
                    f"""
                    Forecast gagal:

                    Product: {product}

                    Model: {method}

                    Error:
                    {e}
                    """
                )


            # Update progress
            progress_bar.progress(
                (i + 1)
                / total_products
            )


        # Selesai Forecast
        
        status_text.success(
            "✅ Forecast seluruh produk selesai."
        )


        if len(results) == 0:

            st.error(
                """
                Tidak ada forecast yang berhasil dibuat.

                Periksa pesan error di atas.
                """
            )

            return


        # Gabungkan semua hasil
        
        final_forecast = pd.concat(
            results,
            ignore_index=True
        )


        # Rapikan MAPE
        final_forecast["MAPE (%)"] = (
            pd.to_numeric(
                final_forecast["MAPE (%)"],
                errors="coerce"
            )
            .round(2)
        )


        # Susun urutan kolom
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


        # Simpan ke session agar tidak hilang
        st.session_state.forecast_result = (
            final_forecast
        )


    # DISPLAY FORECAST RESULT
    
    if st.session_state.forecast_result is not None:

        final_forecast = (
            st.session_state.forecast_result
        )


        st.divider()

        st.subheader(
            "📋 Forecast Result"
        )


        st.dataframe(
            final_forecast,
            use_container_width=True
        )


        # SUMMARY
    
        st.subheader(
            "📊 Forecast Summary"
        )


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
            "Total Forecast Rows",
            len(final_forecast)
        )


        # TOTAL FORECAST PER MONTH
    
        st.subheader(
            "📈 Total Forecast Per Month"
        )


        monthly_forecast = (
            final_forecast
            .groupby("Forecast Date")["Forecast"]
            .sum()
            .reset_index()
        )


        fig_total = px.line(
            monthly_forecast,
            x="Forecast Date",
            y="Forecast",
            markers=True,
            title="Total Sales Forecast Per Month"
        )


        st.plotly_chart(
            fig_total,
            use_container_width=True
        )


        # HISTORICAL VS FORECAST
    
        st.subheader(
            "📉 Historical vs Forecast Per Product"
        )


        selected_product = st.selectbox(
            "Choose Product",
            final_forecast["Model"].unique(),
            key="forecast_chart_product"
        )


        history = (
            df[
                df["Model"] == selected_product
            ][
                ["ds", "y"]
            ]
            .copy()
        )


        history.columns = [
            "Date",
            "Sales"
        ]


        history["Type"] = (
            "Historical"
        )


        future = (
            final_forecast[
                final_forecast["Model"]
                == selected_product
            ][
                [
                    "Forecast Date",
                    "Forecast"
                ]
            ]
            .copy()
        )


        future.columns = [
            "Date",
            "Sales"
        ]


        future["Type"] = (
            "Forecast"
        )


        chart_df = pd.concat(
            [
                history,
                future
            ],
            ignore_index=True
        )


        fig_product = px.line(
            chart_df,
            x="Date",
            y="Sales",
            color="Type",
            markers=True,
            title=f"Sales Trend - {selected_product}"
        )


        st.plotly_chart(
            fig_product,
            use_container_width=True
        )


        # DOWNLOAD EXCEL
    
        st.subheader(
            "📥 Export Forecast"
        )


        output = io.BytesIO()


        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            final_forecast.to_excel(
                writer,
                index=False,
                sheet_name="Forecast Result"
            )


        output.seek(0)


        st.download_button(
            label="Download Forecast Excel",
            data=output,
            file_name="forecast_all_products.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title(
    "Navigation"
)


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
