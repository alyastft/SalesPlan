import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.preprocessing import preprocess_data
from utils.classification import classify_items
from utils.helper import RECOMMENDED_MODELS
from utils.forecasting import forecast_item

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

    df = preprocess_data(df)

    st.write('Processed Dataset')
    st.dataframe(df.head())

    classification_df = classify_items(df)

    st.subheader('Trend Classification')
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

    st.write(
        f"Recommended Models: "
        f"{', '.join(RECOMMENDED_MODELS[category])}"
    )

    selected_method = st.selectbox(
        'Choose Forecasting Method',
        RECOMMENDED_MODELS[category]
    )

    item_df = df[
        df['Model'] == selected_model
    ].copy()

    item_df = item_df.sort_values('ds')

    if st.button('Run Forecast'):

        with st.spinner('Forecasting...'):

            forecast_df = forecast_item(
                item_df,
                selected_method,
                periods=36
            )

        if forecast_df.empty:

            st.warning(
                'Data tidak cukup untuk forecasting'
            )

            st.stop()

        st.success('Forecast Completed')

        display_df = forecast_df.copy()

        display_df['Forecast Date'] = (
            pd.to_datetime(
                display_df['Forecast Date']
            )
            .dt.strftime('%m-%Y')
        )

        st.subheader('Forecast Result')
        st.dataframe(display_df)

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=item_df['ds'],

                y=item_df['y'],

                mode='lines+markers',

                name='Actual'

            )

        )

        fig.add_trace(

            go.Scatter(

                x=forecast_df['Forecast Date'],

                y=forecast_df['Forecast'],

                mode='lines+markers',

                name='Forecast'

            )

        )

        fig.update_layout(

            title=f'{selected_model} Forecast',

            xaxis_title='Date',

            yaxis_title='Sales',

            hovermode='x unified',

            height=600

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader('Forecast Summary')

        total_forecast = (
            forecast_df['Forecast']
            .sum()
        )

        avg_forecast = (
            forecast_df['Forecast']
            .mean()
        )

        max_forecast = (
            forecast_df['Forecast']
            .max()
        )

        min_forecast = (
            forecast_df['Forecast']
            .min()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                'Total Forecast',
                f'{total_forecast:,.0f}'
            )

            st.metric(
                'Average Forecast',
                f'{avg_forecast:,.0f}'
            )

        with col2:

            st.metric(
                'Highest Forecast',
                f'{max_forecast:,.0f}'
            )

            st.metric(
                'Lowest Forecast',
                f'{min_forecast:,.0f}'
            )

else:

    st.info('Please upload file first')
