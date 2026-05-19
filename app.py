import streamlit as st
import pandas as pd


st.set_page_config(
    page_title='Forecasting System',
    layout='wide'
)

st.title('Sales Forecasting System')

st.write('Upload dataset to start forecasting.')

uploaded_file = st.file_uploader(
    'Upload Excel/CSV File',
    type=['xlsx', 'csv']
)

if uploaded_file is not None:

    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    else:
        df = pd.read_csv(uploaded_file)

    st.success('Dataset uploaded successfully')

    st.dataframe(df.head())
