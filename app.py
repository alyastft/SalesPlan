import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import preprocess_data
from utils.eda import show_eda
from utils.classification import classify_items
from utils.forecasting import (
    forecast_item,
    evaluate_model
)
from utils.helper import RECOMMENDED_MODELS


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# SESSION STATE INIT
# =====================================================

if "final_forecast" not in st.session_state:
    st.session_state["final_forecast"] = None

if "history_df" not in st.session_state:
    st.session_state["history_df"] = None

if "col_date" not in st.session_state:
    st.session_state["col_date"] = None

if "col_sales" not in st.session_state:
    st.session_state["col_sales"] = None


# =====================================================
# HELPER: AUTO-DETECT COLUMN NAMES
# =====================================================

def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Deteksi otomatis nama kolom tanggal dan sales dari dataframe.
    Kembalikan (col_date, col_sales).
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Kandidat nama kolom tanggal
    date_candidates = [
        "ds",
        "date",
        "period",
        "bulan",
        "month",
        "tanggal",
        "time",
        "week",
        "year_month"
    ]
    col_date = None
    for cand in date_candidates:
        if cand in cols_lower:
            col_date = cols_lower[cand]
            break
    # Fallback: cari kolom dengan dtype datetime
    if col_date is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                col_date = c
                break
    # Fallback terakhir: kolom pertama
    if col_date is None:
        col_date = df.columns[0]

    # Kandidat nama kolom sales / qty
    sales_candidates = [
        "y",
        "sales",
        "qty",
        "quantity",
        "penjualan",
        "volume",
        "amount",
        "nilai",
        "demand",
        "units"
    ]
    col_sales = None
    for cand in sales_candidates:
        if cand in cols_lower:
            col_sales = cols_lower[cand]
            break
    # Fallback: cari kolom numerik pertama selain kolom date dan Model/KYB
    if col_sales is None:
        skip = {col_date.lower(), "model", "kyb no", "category", "method"}
        for c in df.columns:
            if c.lower() in skip:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                col_sales = c
                break
    # Fallback terakhir
    if col_sales is None:
        col_sales = df.columns[-1]

    return col_date, col_sales


# Lakukan hal yang sama untuk kolom forecast output
def detect_forecast_col(df: pd.DataFrame) -> str:
    """Deteksi nama kolom hasil forecast (prediksi)."""
    candidates = ["forecast", "prediction", "prediksi", "yhat", "pred", "value"]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # Fallback: kolom numerik selain date/model/category
    skip = {"model", "kyb no", "category", "method"}
    for c in df.columns:
        if c.lower() in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]


# =====================================================
# HELPER: MAPE CALCULATION
# =====================================================

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Hitung MAPE (Mean Absolute Percentage Error).
    Nilai aktual 0 diabaikan agar tidak terjadi division-by-zero.
    """
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100


def get_mape_label(mape: float) -> str:
    """Kategori akurasi berdasarkan nilai MAPE."""
    if np.isnan(mape):
        return "N/A"
    if mape < 10:
        return "Sangat Baik"
    if mape < 20:
        return "Baik"
    if mape < 50:
        return "Cukup"
    return "Kurang"


def mape_color(mape: float) -> str:
    label = get_mape_label(mape)
    return {
        "Sangat Baik": "🟢",
        "Baik": "🔵",
        "Cukup": "🟡",
        "Kurang": "🔴",
        "N/A": "⚪",
    }[label]


# =====================================================
# HOME PAGE
# =====================================================

def home():

    st.title("📈 Sales Forecasting Dashboard")

    st.markdown("""
    ### Welcome

    Aplikasi ini digunakan untuk melakukan analisis dan forecasting
    penjualan produk berdasarkan data historical.

    ### Fitur Aplikasi

    #### 1. Data Analysis
    - Upload dataset penjualan
    - Melihat preview data
    - Exploratory Data Analysis (EDA)
    - Analisis tren penjualan
    - Klasifikasi produk menjadi:
        - Stable
        - Declining
        - Volatile
        - Intermittent
        - Discontinued

    #### 2. Forecasting
    Melakukan prediksi penjualan produk menggunakan berbagai metode:
    - XGBoost
    - LightGBM
    - CatBoost
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - ElasticNet
    - Prophet
    - BiLSTM

    Model yang direkomendasikan akan menyesuaikan karakteristik produk.

    #### 3. MAPE Analysis
    - Evaluasi akurasi forecast per produk
    - Grafik perbandingan Forecast vs Actual per item
    - Ringkasan akurasi keseluruhan
    """)

    st.info(
        "Silakan gunakan menu sidebar untuk berpindah halaman."
    )


# =====================================================
# FORECAST PAGE
# =====================================================

def forecasting_page():

    st.title("🔮 Multi Product Forecasting")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"],
        key="forecast"
    )

    if uploaded_file is None:
        st.info("Upload dataset terlebih dahulu")
        return

    # LOAD DATA
    if uploaded_file.name.endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded_file)
    else:
        raw_df = pd.read_csv(uploaded_file)

    df = preprocess_data(raw_df)
    st.write(df.columns.tolist())

    st.write(df.dtypes)

    # Deteksi kolom secara otomatis
    col_date, col_sales = detect_columns(df)
    st.session_state["col_date"] = col_date
    st.session_state["col_sales"] = col_sales

    # Simpan history ke session state untuk dipakai di halaman MAPE
    st.session_state["history_df"] = df

    # CLASSIFICATION
    classify_df = classify_items(df)

    st.subheader("Model Selection Per Product")

    selected_models = {}

    # SELECT MODEL UNTUK SETIAP ITEM
    for idx, row in classify_df.iterrows():

        model_name = row["Model"]
        category = row["Category"]

        recommendations = RECOMMENDED_MODELS.get(
            category,
            ["Random Forest"]
        )

        col1, col2, col3 = st.columns([3, 2, 3])

        with col1:
            st.write(f"**{model_name}**")

        with col2:
            st.write(category)

        with col3:
            selected_models[model_name] = st.selectbox(
                "Model",
                recommendations,
                key=f"model_{model_name}",
                label_visibility="collapsed"
            )

    st.divider()

    periods = st.slider(
        "Forecast Horizon (Month)",
        1,
        36,
        12
    )

    # GENERATE FORECAST
    if st.button("Generate Forecast Semua Produk"):

        progress = st.progress(0)
        results = []
        products = classify_df["Model"].tolist()

        for i, product in enumerate(products):

            item_df = df[df["Model"] == product].copy()
            method = selected_models[product]
            forecast = forecast_item(
                item_df,
                method,
                periods
            )

            mape = evaluate_model(
                item_df,
                method,
                test_period=12
            )

            forecast["MAPE"] = mape

            if not forecast.empty:
                forecast["Model"] = product
                forecast["KYB No"] = item_df["KYB No"].iloc[0]
                forecast["Category"] = (
                    classify_df.loc[
                        classify_df["Model"] == product, "Category"
                    ].values[0]
                )
                forecast["Method"] = method
                forecast["MAPE"] = mape
                results.append(forecast)

            progress.progress((i + 1) / len(products))

        if len(results) == 0:
            st.error("Tidak ada hasil forecast")
            return

        final_forecast = pd.concat(results, ignore_index=True)

        # Simpan ke session state agar bisa diakses halaman MAPE
        st.session_state["final_forecast"] = final_forecast

        # ── PREVIEW ──────────────────────────────────────────────
        st.subheader("Forecast Result")
        st.dataframe(final_forecast, use_container_width=True)

        # ── SUMMARY METRICS ───────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Product", final_forecast["Model"].nunique())
        c2.metric(
            "Total Forecast Sales",
            f"{final_forecast['Forecast'].sum():,.0f}"
        )
        c3.metric("Total Rows", len(final_forecast))

        st.divider()

        # ── GRAFIK FORECAST VS ACTUAL PER ITEM ───────────────────
        st.subheader("📊 Grafik Forecast vs Actual per Produk")

        # Gunakan nama kolom yang terdeteksi
        fc_date_col = detect_forecast_col.__module__ and col_date  # reuse col_date
        fc_date_col = "Forecast Date"
        fc_val_col = "Forecast"

        # Debug info (lipat jika tidak dibutuhkan)
        with st.expander("ℹ️ Info Kolom yang Terdeteksi", expanded=False):
            st.write(f"**Kolom Tanggal (historis):** `{col_date}`")
            st.write(f"**Kolom Sales (historis):** `{col_sales}`")
            st.write(f"**Kolom Forecast:** `{fc_val_col}`")
            st.write(f"**Kolom Tanggal (forecast):** `{fc_date_col}`")
            st.write("**Semua kolom df:**", df.columns.tolist())
            st.write("**Semua kolom forecast:**", final_forecast.columns.tolist())

        for product in final_forecast["Model"].unique():

            # Data aktual
            act_raw = df[df["Model"] == product][[col_date, col_sales]].copy()
            act_raw = act_raw.rename(columns={col_date: "Date", col_sales: "Nilai"})
            act_raw["Date"] = pd.to_datetime(act_raw["Date"])
            act_raw["Tipe"] = "Actual"

            # Data forecast
            fc_raw = final_forecast[final_forecast["Model"] == product][
                [fc_date_col, fc_val_col]
            ].copy()
            fc_raw = fc_raw.rename(columns={fc_date_col: "Date", fc_val_col: "Nilai"})
            fc_raw["Date"] = pd.to_datetime(fc_raw["Date"])
            fc_raw["Tipe"] = "Forecast"

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=act_raw["Date"],
                y=act_raw["Nilai"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#4C9BE8", width=2),
                marker=dict(size=4),
            ))

            fig.add_trace(go.Scatter(
                x=fc_raw["Date"],
                y=fc_raw["Nilai"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#F97316", width=2, dash="dash"),
                marker=dict(size=4),
            ))

            # Garis vertikal pemisah actual / forecast
            if not act_raw.empty and not fc_raw.empty:
                split_date = act_raw["Date"].max()
                fig.add_vline(
                    x=split_date,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Forecast Start",
                    annotation_position="top right",
                )

            fig.update_layout(
                title=f"{product}",
                xaxis_title="Tanggal",
                yaxis_title=col_sales,
                legend=dict(orientation="h", y=1.12),
                height=350,
                margin=dict(t=60, b=40, l=40, r=20),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

        # ── EXPORT EXCEL ──────────────────────────────────────────
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            final_forecast.to_excel(
                writer, index=False, sheet_name="Forecast"
            )
        output.seek(0)

        st.download_button(
            "📥 Download Forecast Excel",
            data=output,
            file_name="forecast_all_products.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
        )


# =====================================================
# MAPE PAGE
# =====================================================

def mape_page():

    st.title("📐 MAPE Analysis — Akurasi Forecast")

    final_forecast: pd.DataFrame | None = st.session_state.get("final_forecast")
    history_df: pd.DataFrame | None = st.session_state.get("history_df")
    col_date: str | None = st.session_state.get("col_date")
    col_sales: str | None = st.session_state.get("col_sales")

    if final_forecast is None or history_df is None:
        st.warning(
            "Belum ada hasil forecast. "
            "Silakan jalankan forecasting terlebih dahulu di halaman **Forecast**."
        )
        return

    # Fallback deteksi kolom jika session state kosong
    if col_date is None or col_sales is None:
        col_date, col_sales = detect_columns(history_df)

    fc_val_col = detect_forecast_col(final_forecast)

    # Pastikan kolom tanggal di forecast ada — gunakan col_date jika tersedia
    fc_date_col = "Forecast Date"
    fc_val_col = "Forecast"

    # ── HITUNG MAPE PER PRODUK ────────────────────────────────────
    # Gabungkan forecast dengan data aktual pada periode yang overlap
    mape_rows = []

    for product in final_forecast["Model"].unique():

        fc = final_forecast[final_forecast["Model"] == product].copy()
        fc["_date_"] = pd.to_datetime(fc[fc_date_col])

        act = history_df[history_df["Model"] == product][
            [col_date, col_sales]
        ].copy()
        act["_date_"] = pd.to_datetime(act[col_date])

        # Inner join pada tanggal yang sama
        mape_val = fc["MAPE"].iloc[0]

        for product in final_forecast["Model"].unique():

        fc = final_forecast[
            final_forecast["Model"] == product
        ]
    
        mape_val = fc["MAPE"].iloc[0]
    
        category = fc["Category"].iloc[0]
        method = fc["Method"].iloc[0]
    
        mape_rows.append({
    
            "Produk": product,
    
            "Kategori": category,
    
            "Metode": method,
    
            "MAPE (%)": round(mape_val, 2),
    
            "Akurasi": get_mape_label(mape_val),
    
            "": mape_color(mape_val)
    
        })

    mape_df = pd.DataFrame(mape_rows)

    # ── SUMMARY CARDS ──────────────────────────────────────────────
    valid = mape_df["MAPE (%)"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Rata-rata MAPE",
        f"{valid.mean():.2f} %" if len(valid) else "N/A",
    )
    c2.metric(
        "MAPE Terbaik",
        f"{valid.min():.2f} %" if len(valid) else "N/A",
    )
    c3.metric(
        "MAPE Terburuk",
        f"{valid.max():.2f} %" if len(valid) else "N/A",
    )
    c4.metric(
        "Produk Dievaluasi",
        len(mape_df),
    )

    st.divider()

    # ── TABEL MAPE ────────────────────────────────────────────────
    st.subheader("Tabel MAPE per Produk")

    st.dataframe(
        mape_df,
        use_container_width=True,
        column_config={
            "MAPE (%)": st.column_config.NumberColumn(
                "MAPE (%)", format="%.2f %%"
            ),
            "": st.column_config.TextColumn("Status", width="small"),
        },
    )

    st.divider()

    # ── BAR CHART MAPE ────────────────────────────────────────────
    st.subheader("📊 Bar Chart MAPE per Produk")

    plot_df = mape_df.dropna(subset=["MAPE (%)"]).sort_values(
        "MAPE (%)", ascending=True
    )

    if plot_df.empty:
        st.info(
            "Tidak ada data overlap antara forecast dan aktual "
            "untuk menghitung MAPE."
        )
    else:
        color_map = {
            "Sangat Baik": "#22c55e",
            "Baik": "#3b82f6",
            "Cukup": "#f59e0b",
            "Kurang": "#ef4444",
        }
        plot_df["_color"] = plot_df["Akurasi"].map(color_map)

        fig_bar = go.Figure(
            go.Bar(
                x=plot_df["MAPE (%)"],
                y=plot_df["Produk"],
                orientation="h",
                marker_color=plot_df["_color"],
                text=plot_df["MAPE (%)"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>"
                ),
            )
        )

        fig_bar.update_layout(
            xaxis_title="MAPE (%)",
            yaxis_title="",
            height=max(300, len(plot_df) * 38),
            margin=dict(t=20, b=40, l=10, r=60),
            xaxis=dict(range=[0, plot_df["MAPE (%)"].max() * 1.25]),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── GRAFIK FORECAST VS ACTUAL PER ITEM ───────────────────────
    st.subheader("📈 Forecast vs Actual per Produk")

    products = final_forecast["Model"].unique().tolist()
    selected_product = st.selectbox(
        "Pilih Produk",
        products,
        key="mape_product_select",
    )

    if selected_product:

        fc = final_forecast[
            final_forecast["Model"] == selected_product
        ].copy()
        fc["_date_"] = pd.to_datetime(fc[fc_date_col])

        act = history_df[
            history_df["Model"] == selected_product
        ][[col_date, col_sales]].copy()
        act["_date_"] = pd.to_datetime(act[col_date])

        fig2 = go.Figure()

        # Actual
        fig2.add_trace(go.Scatter(
            x=act["_date_"],
            y=act[col_sales],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#4C9BE8", width=2.5),
            marker=dict(size=5),
        ))

        # Forecast
        fig2.add_trace(go.Scatter(
            x=fc["_date_"],
            y=fc[fc_val_col],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#F97316", width=2.5, dash="dash"),
            marker=dict(size=5),
        ))

        # Garis vertikal pemisah
        if not act.empty:
            split_date = act["_date_"].max()
            fig2.add_vline(
                x=split_date,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top right",
            )

        # Overlay titik overlap
        merged2 = fc.merge(act[["_date_", col_sales]], on="_date_", how="inner")
        if not merged2.empty:
            fig2.add_trace(go.Scatter(
                x=merged2["_date_"],
                y=merged2[col_sales],
                mode="markers",
                name="Actual (overlap)",
                marker=dict(
                    color="#4C9BE8", size=8, symbol="circle-open",
                    line=dict(width=2, color="#4C9BE8"),
                ),
            ))

        # MAPE annotation
        row = mape_df[mape_df["Produk"] == selected_product]
        mape_val = row["MAPE (%)"].values[0] if not row.empty else np.nan
        akurasi = row["Akurasi"].values[0] if not row.empty else "N/A"

        subtitle = (
            f"MAPE: {mape_val:.2f}% — {akurasi}"
            if not np.isnan(mape_val)
            else "MAPE: N/A (tidak ada data overlap)"
        )

        fig2.update_layout(
            title=dict(
                text=f"{selected_product}<br>"
                     f"<sup style='color:gray'>{subtitle}</sup>",
            ),
            xaxis_title="Tanggal",
            yaxis_title=col_sales,
            legend=dict(orientation="h", y=1.15),
            height=420,
            margin=dict(t=80, b=40, l=40, r=20),
            hovermode="x unified",
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ── EXPORT MAPE ───────────────────────────────────────────────
    st.divider()

    export_df = mape_df.drop(columns=["_color"], errors="ignore").drop(
        columns=[""], errors="ignore"
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="MAPE")
    output.seek(0)

    st.download_button(
        "📥 Download MAPE Excel",
        data=output,
        file_name="mape_analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
    )


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    [
        "Home",
        "Data Analysis",
        "Forecast",
        "MAPE Analysis",
    ]
)

# =====================================================
# ROUTER
# =====================================================

if page == "Home":
    home()

elif page == "Data Analysis":
    show_eda()

elif page == "Forecast":
    forecasting_page()

elif page == "MAPE Analysis":
    mape_page()
