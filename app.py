import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_data
from utils.classification import classify_items

st.set_page_config(
    page_title='Forecasting System',
    layout='wide'
)

st.title("Sales Forecasting System")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    try:

        df = pd.read_excel(uploaded_file)

        st.subheader("Raw Dataset")
        st.dataframe(df.head())

        df = preprocess_data(df)

        st.subheader("Processed Dataset")
        st.dataframe(df.head())

        classification_df = classify_items(df)

        st.subheader("Classification Result")
        st.dataframe(classification_df)

    except Exception as e:

        st.error(f'ERROR: {e}')

else:

    st.info("Please upload Excel file")
