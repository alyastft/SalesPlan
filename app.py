import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_data
from utils.classification import classify_items
from utils.helper import RECOMMENDED_MODELS

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


    # classification
    classification_df = classify_items(df)
    st.write('Trend Classification')
    st.dataframe(classification_df)
    
    # Dropdown item
    selected_model = st.selectbox(
        'Choose Item',
        classification_df['Model']
    )
    
    # Take Category
    selected_row = classification_df[
        classification_df['Model'] == selected_model
    ].iloc[0]
    
    category = selected_row['Category']
    
    # Insight
    st.subheader('Trend Analysis')
    
    st.write(f'Category: {category}')
    
    st.write(f'''
    Recommended Models:
    {", ".join(RECOMMENDED_MODELS[category])}
    ''')
