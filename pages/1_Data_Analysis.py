import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import preprocess_data
from utils.classification import classify_items

st.set_page_config(layout="wide")

def show_data_analyssis():
    st.title("📊 Data Analysis")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["xlsx","csv"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")
    st.dataframe(df)

    df = preprocess_data(df)

    st.subheader("Processed Dataset")
    st.dataframe(df)

    st.subheader("Dataset Summary")

    c1,c2,c3,c4 = st.columns(4)

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

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing"
    ]

    st.dataframe(missing)

    st.subheader("Monthly Sales Trend")

    monthly = (
        df.groupby("ds")["y"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="ds",
        y="y"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Product Classification"
    )

    classification_df = classify_items(df)

    st.dataframe(classification_df)

    st.subheader(
        "Category Distribution"
    )

    category_count = (
        classification_df["Category"]
        .value_counts()
    )

    fig2 = px.pie(
        values=category_count.values,
        names=category_count.index
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader(
        "External Factors"
    )

    usd = pd.read_excel(
        "assets/KursUSD.xlsx"
    )

    jpy = pd.read_excel(
        "assets/KursJPY.xlsx"
    )

    motor = pd.read_excel(
        "assets/ProduksiSepedaMotorSeluruh.xlsx"
    )

    tab1,tab2,tab3 = st.tabs(
        [
            "USD",
            "JPY",
            "Motor Production"
        ]
    )

    with tab1:
        st.dataframe(usd)

    with tab2:
        st.dataframe(jpy)

    with tab3:
        st.dataframe(motor)

else:

    st.info(
        "Upload file terlebih dahulu"
    )

