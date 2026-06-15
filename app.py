import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.preprocessing import preprocess_data
from utils.eda import show_eda
from utils.classification import classify_items
from utils.forecasting import (
    forecast_item,
    evaluate_model,
)
from utils.helper import RECOMMENDED_MODELS
from utils.security import (
    init_session,
    check_session_expiry,
    enforce_state_whitelist,
    validate_upload,
    validate_dataframe,
    drop_pii_columns,
    safe_dataframe_display,
    secure_download_button,
    render_security_sidebar,
    audit_log,
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)


# =====================================================
# SECURITY INIT — jalankan sebelum apapun
# =====================================================

init_session()

if check_session_expiry():
    st.warning("⏱️ Sesi Anda telah kedaluwarsa. Data dihapus otomatis.")
    st.stop()

enforce_state_whitelist()


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

    date_candidates = [
        "ds", "date", "period", "bulan", "month",
        "tanggal", "time", "week", "year_month",
    ]
    col_date = None
    for cand in date_candidates:
        if cand in cols_lower:
            col_date = cols_lower[cand]
            break
    if col_date is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                col_date = c
                break
    if col_date is None:
        col_date = df.columns[0]

    sales_candidates = [
        "y", "sales", "qty", "quantity", "penjualan",
        "volume", "amount", "nilai", "demand", "units",
    ]
    col_sales = None
    for cand in sales_candidates:
        if cand in cols_lower:
            col_sales = cols_lower[cand]
            break
    if col_sales is None:
        skip = {col_date.lower(), "model", "kyb no", "category", "method"}
        for c in df.columns:
            if c.lower() in skip:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                col_sales = c
                break
    if col_sales is None:
        col_sales = df.columns[-1]

    return col_date, col_sales


def detect_forecast_col(df: pd.DataFrame) -> str:
    """Deteksi nama kolom hasil forecast (prediksi)."""
    candidates = ["forecast", "prediction", "prediksi", "yhat", "pred", "value"]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    skip = {"model", "kyb no", "category", "method"}
    for c in df.columns:
        if c.lower() in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]


# =====================================================
# HELPER: MAPE
# =====================================================

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100


def get_mape_label(mape: float) -> str:
    if pd.isna(mape):
        return "N/A"
    if mape < 10:
        return "Sangat Baik"
    if mape < 20:
        return "Baik"
    if mape < 50:
        return "Cukup"
    return "Kurang"


def mape_color(mape: float) -> str:
    return {
        "Sangat Baik": "🟢",
        "Baik": "🔵",
        "Cukup": "🟡",
        "Kurang": "🔴",
        "N/A": "⚪",
    }[get_mape_label(mape)]


# =====================================================
# HELPER: GRAFIK TOTAL BULANAN ACTUAL + FORECAST
# =====================================================

def plot_monthly_total(
    history_df: pd.DataFrame,
    final_forecast: pd.DataFrame,
    col_date: str,
    col_sales: str,
    date_start: str = "2024-01-01",
    date_end: str   = "2027-12-01",
) -> None:
    """
    Tampilkan grafik total penjualan bulanan (semua item digabung):
    - Actual  : dari history_df, difilter sesuai rentang tanggal
    - Forecast: dari final_forecast kolom 'Forecast Date' & 'Forecast'
    - Area abu-abu: zona forecast (setelah bulan terakhir aktual)
    """

    st.subheader("📅 Total Penjualan per Bulan — Semua Produk")

    # ── Rentang filter ────────────────────────────────────────────
    date_range_start = pd.Timestamp(date_start)
    date_range_end   = pd.Timestamp(date_end)

    # ── Actual: agregasi per bulan ────────────────────────────────
    act = history_df[[col_date, col_sales]].copy()
    act["Month"] = pd.to_datetime(act[col_date]).dt.to_period("M").dt.to_timestamp()
    act_monthly = (
        act.groupby("Month")[col_sales]
        .sum()
        .reset_index()
        .rename(columns={col_sales: "Total"})
    )
    act_monthly = act_monthly[
        (act_monthly["Month"] >= date_range_start)
        & (act_monthly["Month"] <= date_range_end)
    ].sort_values("Month")

    # ── Forecast: agregasi per bulan ──────────────────────────────
    fc = final_forecast[["Forecast Date", "Forecast"]].copy()
    fc["Month"] = pd.to_datetime(fc["Forecast Date"]).dt.to_period("M").dt.to_timestamp()
    fc_monthly = (
        fc.groupby("Month")["Forecast"]
        .sum()
        .reset_index()
        .rename(columns={"Forecast": "Total"})
    )
    fc_monthly = fc_monthly[
        (fc_monthly["Month"] >= date_range_start)
        & (fc_monthly["Month"] <= date_range_end)
    ].sort_values("Month")

    if act_monthly.empty and fc_monthly.empty:
        st.info("Tidak ada data dalam rentang Januari 2024 – Desember 2027.")
        return

    # ── Batas pemisah actual / forecast ──────────────────────────
    split_date = act_monthly["Month"].max() if not act_monthly.empty else None

    # ── Summary cards ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Actual",        f"{act_monthly['Total'].sum():,.0f}"
              if not act_monthly.empty else "—")
    c2.metric("Total Forecast",      f"{fc_monthly['Total'].sum():,.0f}"
              if not fc_monthly.empty else "—")
    c3.metric("Rata-rata Actual/bln",
              f"{act_monthly['Total'].mean():,.0f}"
              if not act_monthly.empty else "—")
    c4.metric("Rata-rata Forecast/bln",
              f"{fc_monthly['Total'].mean():,.0f}"
              if not fc_monthly.empty else "—")

    # ── Plotly figure ─────────────────────────────────────────────
    fig = go.Figure()

    # Area forecast (background shading)
    if split_date is not None and not fc_monthly.empty:
        fig.add_vrect(
            x0=split_date,
            x1=date_range_end,
            fillcolor="rgba(249, 115, 22, 0.06)",
            layer="below",
            line_width=0,
        )

    # Trace actual — area + line
    if not act_monthly.empty:
        fig.add_trace(go.Scatter(
            x=act_monthly["Month"],
            y=act_monthly["Total"],
            mode="lines+markers",
            name="Actual",
            fill="tozeroy",
            fillcolor="rgba(76, 155, 232, 0.12)",
            line=dict(color="#4C9BE8", width=2.5),
            marker=dict(size=5, color="#4C9BE8"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))

    # Trace forecast — dashed line + markers
    if not fc_monthly.empty:
        fig.add_trace(go.Scatter(
            x=fc_monthly["Month"],
            y=fc_monthly["Total"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#F97316", width=2.5, dash="dash"),
            marker=dict(size=5, color="#F97316"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
        ))

    # Garis vertikal pemisah
    if split_date is not None:
        fig.add_vline(
            x=split_date,
            line_dash="dot",
            line_color="gray",
            line_width=1.5,
            annotation_text="↑ Mulai Forecast",
            annotation_position="top right",
            annotation_font=dict(size=11, color="gray"),
        )

    # Anotasi range tahun di sumbu X
    fig.update_layout(
        title=dict(
            text="Total Penjualan Bulanan — Semua Produk "
                 "<span style='font-size:13px;color:gray'>"
                 "(Jan 2024 – Des 2027)</span>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Bulan",
            tickformat="%b %Y",
            dtick="M3",                    # tick tiap 3 bulan
            tickangle=-30,
            range=[date_range_start, date_range_end + pd.DateOffset(months=1)],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title="Total Qty / Sales",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        height=440,
        margin=dict(t=80, b=60, l=60, r=20),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Tabel ringkasan per tahun ─────────────────────────────────
    with st.expander("📋 Ringkasan per Tahun", expanded=False):

        all_monthly = pd.concat([
            act_monthly.assign(Tipe="Actual"),
            fc_monthly.assign(Tipe="Forecast"),
        ], ignore_index=True)

        all_monthly["Tahun"] = all_monthly["Month"].dt.year

        pivot = (
            all_monthly
            .groupby(["Tahun", "Tipe"])["Total"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        # Pastikan kolom ada meski salah satu kosong
        for col in ["Actual", "Forecast"]:
            if col not in pivot.columns:
                pivot[col] = 0

        pivot["Total"] = pivot["Actual"] + pivot["Forecast"]
        pivot["Actual"]   = pivot["Actual"].apply(lambda v: f"{v:,.0f}")
        pivot["Forecast"] = pivot["Forecast"].apply(lambda v: f"{v:,.0f}")
        pivot["Total"]    = pivot["Total"].apply(lambda v: f"{v:,.0f}")

        st.dataframe(
            pivot.rename(columns={"Tahun": "Tahun"}),
            use_container_width=True,
            hide_index=True,
        )


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

    st.info("Silakan gunakan menu sidebar untuk berpindah halaman.")


# =====================================================
# FORECAST PAGE
# =====================================================

def forecasting_page():
    st.title("🔮 Multi Product Forecasting")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"],
        key="forecast",
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"],
        key="forecast",
    )
    
    # ── TEMPORARY DEBUG — hapus setelah masalah selesai ──
    st.write("DEBUG uploaded_file:", uploaded_file)
    st.write("DEBUG session keys:", list(st.session_state.keys()))
    # ─────────────────────────────────────────────────────
    
    if uploaded_file is None:
        st.info("Upload dataset terlebih dahulu")
        return

    if uploaded_file is None:
        st.info("Upload dataset terlebih dahulu")
        return

    # ── Validasi file ─────────────────────────────────────────────
    ok, err_msg = validate_upload(uploaded_file)
    if not ok:
        st.error(f"❌ File ditolak: {err_msg}")
        audit_log("UPLOAD_REJECTED", err_msg)
        return

    # ── Load data ─────────────────────────────────────────────────
    if uploaded_file.name.endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded_file)
    else:
        raw_df = pd.read_csv(uploaded_file)

    # ── Validasi DataFrame ────────────────────────────────────────
    ok_df, err_df = validate_dataframe(raw_df)
    if not ok_df:
        st.error(f"❌ Data ditolak: {err_df}")
        audit_log("DATA_REJECTED", err_df)
        return

    # ── Sanitasi PII sebelum diproses ─────────────────────────────
    raw_df = drop_pii_columns(raw_df)
    audit_log("UPLOAD_OK", f"rows={len(raw_df)} cols={len(raw_df.columns)}")

    df = preprocess_data(raw_df)

    # ── Deteksi kolom ─────────────────────────────────────────────
    col_date, col_sales = detect_columns(df)
    st.session_state["col_date"] = col_date
    st.session_state["col_sales"] = col_sales
    st.session_state["history_df"] = df

    # ── Klasifikasi ───────────────────────────────────────────────
    classify_df = classify_items(df)

    st.subheader("Model Selection Per Product")

    selected_models: dict[str, str] = {}

    for _, row in classify_df.iterrows():
        model_name = row["Model"]
        category   = row["Category"]
        recommendations = RECOMMENDED_MODELS.get(category, ["Random Forest"])

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
                label_visibility="collapsed",
            )

    st.divider()

    periods = st.slider("Forecast Horizon (Month)", 1, 36, 12)

    # ── Generate Forecast ─────────────────────────────────────────
    if st.button("Generate Forecast Semua Produk"):

        progress = st.progress(0)
        results  = []
        products = classify_df["Model"].tolist()

        for i, product in enumerate(products):
            item_df = df[df["Model"] == product].copy()
            method  = selected_models[product]

            forecast = forecast_item(item_df, method, periods)
            mape     = evaluate_model(item_df, method, test_period=12)

            if not forecast.empty:
                forecast["Model"]    = product
                forecast["KYB No"]   = item_df["KYB No"].iloc[0]
                forecast["Category"] = (
                    classify_df.loc[
                        classify_df["Model"] == product, "Category"
                    ].values[0]
                )
                forecast["Method"] = method
                forecast["MAPE"]   = mape
                results.append(forecast)

            progress.progress((i + 1) / len(products))

        if not results:
            st.error("Tidak ada hasil forecast")
            return

        final_forecast = pd.concat(results, ignore_index=True)
        st.session_state["final_forecast"] = final_forecast

        audit_log(
            "FORECAST_GENERATED",
            f"products={final_forecast['Model'].nunique()} "
            f"rows={len(final_forecast)}",
        )

        # ── Preview ───────────────────────────────────────────────
        st.subheader("Forecast Result")
        safe_dataframe_display(final_forecast, use_container_width=True)

        # ── Summary metrics ───────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Product",        final_forecast["Model"].nunique())
        c2.metric("Total Forecast Sales", f"{final_forecast['Forecast'].sum():,.0f}")
        c3.metric("Total Rows",           len(final_forecast))

        st.divider()

        # ── Grafik Total Penjualan per Bulan (Actual + Forecast) ──
        plot_monthly_total(
            history_df=df,
            final_forecast=final_forecast,
            col_date=col_date,
            col_sales=col_sales,
        )

        st.divider()

        # ── Grafik Forecast vs Actual per produk ──────────────────
        st.subheader("📊 Grafik Forecast vs Actual per Produk")

        fc_date_col = "Forecast Date"
        fc_val_col  = "Forecast"

        with st.expander("ℹ️ Info Kolom yang Terdeteksi", expanded=False):
            st.write(f"**Kolom Tanggal (historis):** `{col_date}`")
            st.write(f"**Kolom Sales (historis):** `{col_sales}`")
            st.write(f"**Kolom Forecast:** `{fc_val_col}`")
            st.write(f"**Kolom Tanggal (forecast):** `{fc_date_col}`")
            st.write("**Semua kolom df:**",           df.columns.tolist())
            st.write("**Semua kolom forecast:**", final_forecast.columns.tolist())

        for product in final_forecast["Model"].unique():

            act_raw = df[df["Model"] == product][[col_date, col_sales]].copy()
            act_raw = act_raw.rename(columns={col_date: "Date", col_sales: "Nilai"})
            act_raw["Date"] = pd.to_datetime(act_raw["Date"])

            fc_raw = final_forecast[final_forecast["Model"] == product][
                [fc_date_col, fc_val_col]
            ].copy()
            fc_raw = fc_raw.rename(columns={fc_date_col: "Date", fc_val_col: "Nilai"})
            fc_raw["Date"] = pd.to_datetime(fc_raw["Date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=act_raw["Date"], y=act_raw["Nilai"],
                mode="lines+markers", name="Actual",
                line=dict(color="#4C9BE8", width=2), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=fc_raw["Date"], y=fc_raw["Nilai"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#F97316", width=2, dash="dash"), marker=dict(size=4),
            ))

            if not act_raw.empty and not fc_raw.empty:
                fig.add_vline(
                    x=act_raw["Date"].max(),
                    line_dash="dot", line_color="gray",
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

        # ── Export Excel ──────────────────────────────────────────
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            final_forecast.to_excel(writer, index=False, sheet_name="Forecast")
        output.seek(0)

        secure_download_button(
            label="📥 Download Forecast Excel",
            data=output,
            file_name="forecast_all_products.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
            action_name="EXPORT_FORECAST",
        )


# =====================================================
# MAPE PAGE
# =====================================================

def mape_page():
    st.title("📐 MAPE Analysis — Akurasi Forecast")

    final_forecast: pd.DataFrame | None = st.session_state.get("final_forecast")
    history_df:     pd.DataFrame | None = st.session_state.get("history_df")
    col_date:       str | None          = st.session_state.get("col_date")
    col_sales:      str | None          = st.session_state.get("col_sales")

    if final_forecast is None or history_df is None:
        st.warning(
            "Belum ada hasil forecast. "
            "Silakan jalankan forecasting terlebih dahulu di halaman **Forecast**."
        )
        return

    if col_date is None or col_sales is None:
        col_date, col_sales = detect_columns(history_df)

    fc_date_col = "Forecast Date"
    fc_val_col  = "Forecast"

    # ── Hitung MAPE per produk ────────────────────────────────────
    mape_rows = []

    for product in final_forecast["Model"].unique():
        fc       = final_forecast[final_forecast["Model"] == product]
        mape_val = fc["MAPE"].iloc[0]
        category = fc["Category"].iloc[0]
        method   = fc["Method"].iloc[0]

        mape_rows.append({
            "Produk"  : product,
            "Kategori": category,
            "Metode"  : method,
            "MAPE (%)": round(mape_val, 2) if pd.notna(mape_val) else np.nan,
            "Akurasi" : get_mape_label(mape_val) if pd.notna(mape_val) else "N/A",
            ""        : mape_color(mape_val)      if pd.notna(mape_val) else "⚪",
        })

    mape_df = pd.DataFrame(mape_rows)

    # ── Summary cards ─────────────────────────────────────────────
    valid = mape_df["MAPE (%)"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata MAPE",   f"{valid.mean():.2f} %" if len(valid) else "N/A")
    c2.metric("MAPE Terbaik",     f"{valid.min():.2f} %"  if len(valid) else "N/A")
    c3.metric("MAPE Terburuk",    f"{valid.max():.2f} %"  if len(valid) else "N/A")
    c4.metric("Produk Dievaluasi", len(mape_df))

    st.divider()

    # ── Tabel MAPE ────────────────────────────────────────────────
    st.subheader("Tabel MAPE per Produk")

    safe_dataframe_display(
        mape_df,
        use_container_width=True,
        column_config={
            "MAPE (%)": st.column_config.NumberColumn("MAPE (%)", format="%.2f %%"),
            "":         st.column_config.TextColumn("Status", width="small"),
        },
    )

    st.divider()

    # ── Bar chart MAPE ────────────────────────────────────────────
    st.subheader("📊 Bar Chart MAPE per Produk")

    plot_df = mape_df.dropna(subset=["MAPE (%)"]).sort_values("MAPE (%)", ascending=True)

    if plot_df.empty:
        st.info("Tidak ada data untuk menghitung MAPE.")
    else:
        color_map = {
            "Sangat Baik": "#22c55e",
            "Baik":        "#3b82f6",
            "Cukup":       "#f59e0b",
            "Kurang":      "#ef4444",
        }
        plot_df = plot_df.copy()
        plot_df["_color"] = plot_df["Akurasi"].map(color_map)

        fig_bar = go.Figure(go.Bar(
            x=plot_df["MAPE (%)"],
            y=plot_df["Produk"],
            orientation="h",
            marker_color=plot_df["_color"],
            text=plot_df["MAPE (%)"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>",
        ))
        fig_bar.update_layout(
            xaxis_title="MAPE (%)",
            yaxis_title="",
            height=max(300, len(plot_df) * 38),
            margin=dict(t=20, b=40, l=10, r=60),
            xaxis=dict(range=[0, plot_df["MAPE (%)"].max() * 1.25]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Grafik Forecast vs Actual per produk ─────────────────────
    st.subheader("📈 Forecast vs Actual per Produk")

    selected_product = st.selectbox(
        "Pilih Produk",
        final_forecast["Model"].unique().tolist(),
        key="mape_product_select",
    )

    if selected_product:
        fc  = final_forecast[final_forecast["Model"] == selected_product].copy()
        fc["_date_"] = pd.to_datetime(fc[fc_date_col])

        act = history_df[history_df["Model"] == selected_product][
            [col_date, col_sales]
        ].copy()
        act["_date_"] = pd.to_datetime(act[col_date])

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=act["_date_"], y=act[col_sales],
            mode="lines+markers", name="Actual",
            line=dict(color="#4C9BE8", width=2.5), marker=dict(size=5),
        ))
        fig2.add_trace(go.Scatter(
            x=fc["_date_"], y=fc[fc_val_col],
            mode="lines+markers", name="Forecast",
            line=dict(color="#F97316", width=2.5, dash="dash"), marker=dict(size=5),
        ))

        if not act.empty:
            fig2.add_vline(
                x=act["_date_"].max(),
                line_dash="dot", line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top right",
            )

        merged2 = fc.merge(act[["_date_", col_sales]], on="_date_", how="inner")
        if not merged2.empty:
            fig2.add_trace(go.Scatter(
                x=merged2["_date_"], y=merged2[col_sales],
                mode="markers", name="Actual (overlap)",
                marker=dict(
                    color="#4C9BE8", size=8, symbol="circle-open",
                    line=dict(width=2, color="#4C9BE8"),
                ),
            ))

        row_mape = mape_df[mape_df["Produk"] == selected_product]
        mape_val = row_mape["MAPE (%)"].values[0] if not row_mape.empty else np.nan
        akurasi  = row_mape["Akurasi"].values[0]  if not row_mape.empty else "N/A"

        subtitle = (
            f"MAPE: {mape_val:.2f}% — {akurasi}"
            if pd.notna(mape_val)
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

    # ── Export MAPE ───────────────────────────────────────────────
    st.divider()

    export_df = mape_df.drop(columns=["_color", ""], errors="ignore")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="MAPE")
    output.seek(0)

    secure_download_button(
        label="📥 Download MAPE Excel",
        data=output,
        file_name="mape_analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        action_name="EXPORT_MAPE",
    )


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    ["Home", "Data Analysis", "Forecast", "MAPE Analysis"],
)

render_security_sidebar()

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
