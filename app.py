import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_data

st.set_page_config(
    page_title='Forecasting System',
    layout='wide'
)

st.title('Sales Forecasting System')

uploaded_file = st.file_uploader(
    'Upload Excel/CSV File',
    type=['xlsx', 'csv']
)

if uploaded_file is not None:

    if uploaded_file.name.endswith('.xlsx'):

        df = pd.read_excel(uploaded_file)

    else:

        df = pd.read_csv(uploaded_file)

    st.write('Raw Dataset')
    st.dataframe(df.head())

    # preprocessing
    df = preprocess_data(df)

    st.write('Processed Dataset')
    st.dataframe(df.head())
