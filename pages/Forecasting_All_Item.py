import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocessing import preprocess_data
from utils.classification import classify_items
from utils.forecasting import forecast_item
from utils.helper import RECOMMENDED_MODELS

st.set_page_config(
    page_title="Forecast All Item",
    layout="wide"
)

st.title("📈 Forecasting All Items")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["xlsx", "csv"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df = preprocess_data(df)

    classification_df = classify_items(df)

    st.subheader(
        "Recommended Forecasting Method"
    )

    selected_models = {}

    for _, row in classification_df.iterrows():

        item_name = row["Model"]

        category = row["Category"]

        recommendation = (
            RECOMMENDED_MODELS[category]
        )

        selected_models[item_name] = st.selectbox(
            f"{item_name}",
            recommendation,
            key=item_name
        )

    if st.button(
        "🚀 Run Forecast All Items"
    ):

        all_forecasts = []

        progress = st.progress(0)

        total_item = len(
            classification_df
        )

        for idx, row in enumerate(
            classification_df.iterrows()
        ):

            row = row[1]

            model_name = row["Model"]

            method = (
                selected_models[
                    model_name
                ]
            )

            item_df = df[
                df["Model"] == model_name
            ].copy()

            item_df = item_df.sort_values(
                "ds"
            )

            try:

                forecast_df = forecast_item(
                    item_df=item_df,
                    method=method,
                    periods=36
                )

                if not forecast_df.empty:

                    forecast_df[
                        "Model"
                    ] = model_name

                    forecast_df[
                        "Method"
                    ] = method

                    forecast_df[
                        "Forecast"
                    ] = (
                        forecast_df[
                            "Forecast"
                        ]
                        .round()
                        .astype(int)
                    )

                    all_forecasts.append(
                        forecast_df
                    )

            except Exception as e:

                st.warning(
                    f"{model_name} gagal: {e}"
                )

            progress.progress(
                (idx + 1)
                / total_item
            )

        if len(all_forecasts) > 0:

            result = pd.concat(
                all_forecasts,
                ignore_index=True
            )

            result[
                "Forecast Month"
            ] = pd.to_datetime(
                result[
                    "Forecast Date"
                ]
            ).dt.strftime(
                "%m-%Y"
            )

            result = result[
                [
                    "Model",
                    "Method",
                    "Forecast Month",
                    "Forecast"
                ]
            ]

            st.success(
                "Forecast selesai"
            )

            st.subheader(
                "Forecast Result"
            )

            st.dataframe(
                result,
                use_container_width=True
            )

            summary = (
                result
                .groupby("Model")
                ["Forecast"]
                .sum()
                .reset_index()
            )

            summary.columns = [
                "Model",
                "Total Forecast 36 Months"
            ]

            st.subheader(
                "Forecast Summary"
            )

            st.dataframe(
                summary,
                use_container_width=True
            )

            excel_buffer = (
                pd.ExcelWriter(
                    "forecast_all.xlsx",
                    engine="openpyxl"
                )
            )

            result.to_excel(
                excel_buffer,
                sheet_name="Forecast",
                index=False
            )

            summary.to_excel(
                excel_buffer,
                sheet_name="Summary",
                index=False
            )

            excel_buffer.close()

            with open(
                "forecast_all.xlsx",
                "rb"
            ) as file:

                st.download_button(
                    label="📥 Download Forecast",
                    data=file,
                    file_name="forecast_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:

    st.info(
        "Upload dataset terlebih dahulu"
    )
