import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_data
from utils.classification import classify_items
from utils.helper import RECOMMENDED_MODELS

st.set_page_config(
    page_title='Forecasting System',
    layout='wide'
)

st.title("Sales Forecasting")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.write("Raw Data")
    st.dataframe(df.head())

    df = preprocess_data(df)

    st.write("Processed Dataset")
    st.dataframe(df.head())

    classification_df = classify_items(df)

    st.write("Trend Classification")
    st.dataframe(classification_df)

    selected_model = st.selectbox(
        'Choose Item',
        classification_df['Model']
    )

    selected_row = classification_df[
        classification_df['Model'] == selected_model
    ].iloc[0]

    category = selected_row['Category']

    st.subheader('Trend Analysis')

    st.write(f'Category: {category}')

    st.write(f'''
    Recommended Models:
    {", ".join(RECOMMENDED_MODELS[category])}
    ''')

    selected_method = st.selectbox(
        'Choose Forecasting Method',
        RECOMMENDED_MODELS[category]
    )

else:

    st.info("Please upload file first")
