import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import preprocess_data
from utils.classification import classify_items


def show_eda():

    st.title("📊 Data Analysis & EDA")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"]
    )

    if uploaded_file is None:

        st.info(
            "Upload dataset terlebih dahulu"
        )

        return

    # ==========================
    # LOAD DATA
    # ==========================

    if uploaded_file.name.endswith(".xlsx"):

        raw_df = pd.read_excel(
            uploaded_file
        )

    else:

        raw_df = pd.read_csv(
            uploaded_file
        )

    st.subheader(
        "Raw Dataset"
    )

    st.dataframe(
        raw_df,
        use_container_width=True
    )

    # ==========================
    # PREPROCESSING
    # ==========================

    df = preprocess_data(
        raw_df
    )

    st.subheader(
        "Processed Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # ==========================
    # SUMMARY
    # ==========================

    st.subheader(
        "Dataset Summary"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        len(df)
    )

    c2.metric(
        "Products",
        df["Model"].nunique()
    )

    c3.metric(
        "Total Sales",
        f"{df['y'].sum():,.0f}"
    )

    c4.metric(
        "Average Sales",
        f"{df['y'].mean():,.0f}"
    )

    # ==========================
    # DESCRIPTIVE STATS
    # ==========================

    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(
        df["y"]
        .describe()
        .to_frame()
    )

    # ==========================
    # MISSING VALUES
    # ==========================

    st.subheader(
        "Missing Values"
    )

    missing_df = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing_df
    )

    # ==========================
    # MONTHLY SALES TREND
    # ==========================

    st.subheader(
        "Monthly Sales Trend"
    )

    monthly_sales = (
        df.groupby("ds")["y"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="ds",
        y="y",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================
    # TOP PRODUCT
    # ==========================

    st.subheader(
        "Top 10 Product"
    )

    top_product = (
        df.groupby("Model")["y"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        top_product,
        x="Model",
        y="y"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ==========================
    # CLASSIFICATION
    # ==========================

    st.subheader(
        "Product Classification"
    )

    classification_df = classify_items(
        df
    )

    st.dataframe(
        classification_df,
        use_container_width=True
    )

    # ==========================
    # CATEGORY DISTRIBUTION
    # ==========================

    st.subheader(
        "Category Distribution"
    )

    category_count = (
        classification_df[
            "Category"
        ]
        .value_counts()
    )

    fig3 = px.pie(
        values=category_count.values,
        names=category_count.index
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # ==========================
    # TREND PER ITEM
    # ==========================

    st.subheader(
        "Trend Per Product"
    )

    selected_product = st.selectbox(
        "Choose Product",
        sorted(
            df["Model"].unique()
        )
    )

    item_df = df[
        df["Model"]
        == selected_product
    ]

    fig4 = px.line(
        item_df,
        x="ds",
        y="y",
        markers=True,
        title=selected_product
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ==========================
    # EXTERNAL FACTORS
    # ==========================

    st.subheader(
        "External Factors"
    )

    try:

        usd = pd.read_excel(
            "assets/KursUSD.xlsx"
        )

        jpy = pd.read_excel(
            "assets/KursJPY.xlsx"
        )

        motor = pd.read_excel(
            "assets/ProduksiSepedaMotorSeluruh.xlsx"
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "USD",
                "JPY",
                "Motor Production"
            ]
        )

        with tab1:

            st.dataframe(
                usd,
                use_container_width=True
            )

            usd.columns = [
                "Tanggal",
                "USD"
            ]

            fig_usd = px.line(
                usd,
                x="Tanggal",
                y="USD"
            )

            st.plotly_chart(
                fig_usd,
                use_container_width=True
            )

        with tab2:

            st.dataframe(
                jpy,
                use_container_width=True
            )

            jpy.columns = [
                "Tanggal",
                "JPY"
            ]

            fig_jpy = px.line(
                jpy,
                x="Tanggal",
                y="JPY"
            )

            st.plotly_chart(
                fig_jpy,
                use_container_width=True
            )

        with tab3:

            st.dataframe(
                motor,
                use_container_width=True
            )

            fig_motor = go.Figure()

            fig_motor.add_trace(
                go.Scatter(
                    x=motor["Bulan"],
                    y=motor["Domestik"],
                    name="Domestik"
                )
            )

            fig_motor.add_trace(
                go.Scatter(
                    x=motor["Bulan"],
                    y=motor["Ekspor"],
                    name="Ekspor"
                )
            )

            st.plotly_chart(
                fig_motor,
                use_container_width=True
            )

    except:

        st.warning(
            "File faktor eksternal belum ditemukan"
        )

    # ==========================
    # EXPORT PRODUCT
    # ==========================

    st.subheader(
        "Export Product Analysis"
    )

    export_df = raw_df[
        raw_df["KYB No"]
        .astype(str)
        .str.endswith("-E")
    ]

    st.metric(
        "Jumlah Product Export",
        export_df["Model"].nunique()
    )

    st.dataframe(
        export_df.head()
    )
