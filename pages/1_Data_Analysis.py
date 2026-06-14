import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import preprocess_data
from utils.classification import classify_items

st.set_page_config(
    page_title="Data Analysis",
    layout="wide"
)

st.title("Data Analysis & Exploratory Data Analysis")

uploaded_file = st.file_uploader(
    "Upload Excel/CSV File",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df = preprocess_data(df)

    st.success("Dataset Loaded")

    # ==================================
    # DATASET OVERVIEW
    # ==================================

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            len(df)
        )

    with col2:
        st.metric(
            "Total Products",
            df["Model"].nunique()
        )

    with col3:
        st.metric(
            "Start Period",
            df["ds"].min().strftime("%b-%Y")
        )

    with col4:
        st.metric(
            "End Period",
            df["ds"].max().strftime("%b-%Y")
        )

    st.dataframe(df.head())

    # ==================================
    # MISSING VALUE
    # ==================================

    st.header("Missing Value Check")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_df)

    # ==================================
    # STATISTICAL SUMMARY
    # ==================================

    st.header("Statistical Summary")

    st.dataframe(df.describe())

    # ==================================
    # MONTHLY SALES TREND
    # ==================================

    st.header("Monthly Sales Trend")

    monthly_sales = (
        df.groupby("ds")["y"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="ds",
        y="y",
        title="Overall Monthly Sales Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================
    # SEASONALITY
    # ==================================

    st.header("Seasonality Analysis")

    seasonality = (
        df.assign(
            Month=df["ds"].dt.month_name()
        )
        .groupby("Month")["y"]
        .mean()
        .reset_index()
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    seasonality["Month"] = pd.Categorical(
        seasonality["Month"],
        categories=month_order,
        ordered=True
    )

    seasonality = seasonality.sort_values(
        "Month"
    )

    fig = px.bar(
        seasonality,
        x="Month",
        y="y",
        title="Average Sales per Month"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================
    # TOP PRODUCTS
    # ==================================

    st.header("Top 20 Best Selling Products")

    top20 = (
        df.groupby("Model")["y"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(20)
        .reset_index()
    )

    fig = px.bar(
        top20,
        x="Model",
        y="y",
        title="Top 20 Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(top20)

    # ==================================
    # TREND CLASSIFICATION
    # ==================================

    st.header("Trend Classification")

    classification_df = classify_items(df)

    st.dataframe(
        classification_df,
        use_container_width=True
    )

    # ==================================
    # EXTERNAL FACTORS
    # ==================================

    st.header("External Factors")

    usd = pd.read_excel(
        "assets/KursUSD.xlsx"
    )

    jpy = pd.read_excel(
        "assets/KursJPY.xlsx"
    )

    motor = pd.read_excel(
        "assets/ProduksiSepedaMotorSeluruh.xlsx"
    )

    usd["Tanggal"] = pd.to_datetime(
        usd["Tanggal"]
    )

    jpy["Tanggal"] = pd.to_datetime(
        jpy["Tanggal"]
    )

    motor["Bulan"] = pd.to_datetime(
        motor["Bulan"]
    )

    st.subheader("USD Exchange Rate")

    fig = px.line(
        usd,
        x="Tanggal",
        y="Kurs USD/IDR"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("JPY Exchange Rate")

    fig = px.line(
        jpy,
        x="Tanggal",
        y="Kurs JPY/IDR"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Motorcycle Production")

    fig = px.line(
        motor,
        x="Bulan",
        y=["Domestik", "Ekspor"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================
    # CORRELATION ANALYSIS
    # ==================================

    st.header("Correlation Analysis")

    sales_monthly = (
        df.groupby("ds")["y"]
        .sum()
        .reset_index()
    )

    sales_monthly.columns = [
        "ds",
        "Sales"
    ]

    usd["ds"] = (
        usd["Tanggal"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    jpy["ds"] = (
        jpy["Tanggal"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    motor["ds"] = (
        motor["Bulan"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    corr_df = (
        sales_monthly
        .merge(
            usd[["ds", "Kurs USD/IDR"]],
            on="ds",
            how="left"
        )
        .merge(
            jpy[["ds", "Kurs JPY/IDR"]],
            on="ds",
            how="left"
        )
        .merge(
            motor[
                [
                    "ds",
                    "Domestik",
                    "Ekspor"
                ]
            ],
            on="ds",
            how="left"
        )
    )

    corr_matrix = corr_df.drop(
        columns=["ds"]
    ).corr()

    st.dataframe(
        corr_matrix.round(3)
    )

else:

    st.info(
        "Please upload file first"
    )
