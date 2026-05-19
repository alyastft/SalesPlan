import streamlit as st
import pandas as pd
import numpy as np
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

    st.subheader('Raw Dataset')
    st.dataframe(df.head())

    df = preprocess_data(df)

    st.subheader('Processed Dataset')
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

        forecast_df['Forecast Date'] = pd.to_datetime(
            forecast_df['Forecast Date']
        )

        forecast_df['Forecast Month'] = (
            forecast_df['Forecast Date']
            .dt.strftime('%m-%Y')
        )

        forecast_df['Year'] = (
            forecast_df['Forecast Date']
            .dt.year
        )

        st.subheader('Forecast Result')

        display_df = forecast_df.copy()

        display_df['Forecast'] = (
            display_df['Forecast']
            .round(0)
            .astype(int)
        )

        st.dataframe(
            display_df[
                ['Forecast Month', 'Forecast']
            ]
        )

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

        actual = item_df['y'].tail(
            min(12, len(item_df))
        ).values

        pred = forecast_df['Forecast'].head(
            len(actual)
        ).values

        if len(actual) == len(pred):

            actual = np.array(actual)

            pred = np.array(pred)

            mask = actual != 0

            actual = actual[mask]

            pred = pred[mask]

            if len(actual) > 0:

                mape = np.mean(
                    np.abs(
                        (actual - pred) / actual
                    )
                ) * 100

            else:

                mape = 0

        else:

            mape = 0

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

        yearly_summary = (
            forecast_df
            .groupby('Year')['Forecast']
            .sum()
            .reset_index()
        )

        yearly_summary['Forecast'] = (
            yearly_summary['Forecast']
            .round(0)
            .astype(int)
        )

        st.subheader('Forecast Summary')

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                'Average Forecast',
                f'{avg_forecast:,.0f}'
            )

            st.metric(
                'Highest Forecast',
                f'{max_forecast:,.0f}'
            )

        with col2:

            st.metric(
                'Lowest Forecast',
                f'{min_forecast:,.0f}'
            )

            st.metric(
                'MAPE',
                f'{mape:.2f}%'
            )

        st.subheader('Forecast Per Year')

        st.dataframe(yearly_summary)

        st.metric(
            'Total Forecast',
            f'{total_forecast:,.0f}'
        )

else:

    st.info('Please upload file first')
