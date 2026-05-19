import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Sales Forecasting")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=['xlsx', 'csv']
)

if uploaded_file is not None:

    # READ FILE
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write(df.head())

    # PILIH MODEL
    selected_model = st.selectbox(
        "Choose Model",
        df['Model'].unique()
    )

    # FILTER
    item_df = df[df['Model'] == selected_model]

    # PLOT
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(item_df['ds'], item_df['y'])

    ax.set_title(selected_model)

    st.pyplot(fig)
