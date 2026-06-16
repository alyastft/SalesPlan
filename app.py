import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.preprocessing import preprocess_data
from utils.eda import show_eda
from utils.classification import classify_items
from utils.forecasting import forecast_item, evaluate_model
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

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: rgba(255,255,255,0.8); margin: 0.25rem 0 0; }

    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title { font-weight: 700; color: #1e3a5f; margin-bottom: 0.5rem; }

    .status-ok   { color: #16a34a; font-weight: 600; }
    .status-warn { color: #d97706; font-weight: 600; }
    .status-err  { color: #dc2626; font-weight: 600; }

    .stProgress > div > div { background: #2d6a9f !important; }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SECURITY INIT
# =====================================================

init_session()

if check_session_expiry():
    st.warning("⏱️ Sesi Anda telah kedaluwarsa. Data dihapus otomatis.")
    st.stop()

enforce_state_whitelist()

# =====================================================
# SESSION STATE INIT
# =====================================================

for key in ["final_forecast", "history_df", "col_date", "col_sales"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =====================================================
# HELPER: AUTO-DETECT COLUMN NAMES
# =====================================================

def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Deteksi otomatis kolom tanggal dan kolom sales."""
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


# =====================================================
# HELPER: MAPE
# =====================================================

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


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
        "Baik":        "🔵",
        "Cukup":       "🟡",
        "Kurang":      "🔴",
        "N/A":         "⚪",
    }[get_mape_label(mape)]


# =====================================================
# HELPER: GRAFIK TOTAL BULANAN
# =====================================================

def plot_monthly_total(
    history_df: pd.DataFrame,
    final_forecast: pd.DataFrame,
    col_date: str,
    col_sales: str,
    date_start: str = "2024-01-01",
    date_end: str   = "2027-12-01",
) -> None:
    st.subheader("📅 Total Penjualan per Bulan — Semua Produk")

    date_range_start = pd.Timestamp(date_start)
    date_range_end   = pd.Timestamp(date_end)

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
        st.info("Tidak ada data dalam rentang yang dipilih.")
        return

    split_date = act_monthly["Month"].max() if not act_monthly.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Actual",           f"{act_monthly['Total'].sum():,.0f}"  if not act_monthly.empty else "—")
    c2.metric("Total Forecast",         f"{fc_monthly['Total'].sum():,.0f}"   if not fc_monthly.empty else "—")
    c3.metric("Rata-rata Actual/bln",   f"{act_monthly['Total'].mean():,.0f}" if not act_monthly.empty else "—")
    c4.metric("Rata-rata Forecast/bln", f"{fc_monthly['Total'].mean():,.0f}"  if not fc_monthly.empty else "—")

    fig = go.Figure()

    if split_date is not None and not fc_monthly.empty:
        fig.add_vrect(
            x0=split_date, x1=date_range_end,
            fillcolor="rgba(249, 115, 22, 0.06)",
            layer="below", line_width=0,
        )

    if not act_monthly.empty:
        fig.add_trace(go.Scatter(
            x=act_monthly["Month"], y=act_monthly["Total"],
            mode="lines+markers", name="Actual",
            fill="tozeroy", fillcolor="rgba(76, 155, 232, 0.12)",
            line=dict(color="#4C9BE8", width=2.5), marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))

    if not fc_monthly.empty:
        fig.add_trace(go.Scatter(
            x=fc_monthly["Month"], y=fc_monthly["Total"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#F97316", width=2.5, dash="dash"), marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
        ))

    if split_date is not None:
        fig.add_vline(
            x=split_date, line_dash="dot", line_color="gray", line_width=1.5,
            annotation_text="↑ Mulai Forecast", annotation_position="top right",
            annotation_font=dict(size=11, color="gray"),
        )

    fig.update_layout(
        title=dict(text="Total Penjualan Bulanan — Semua Produk", font=dict(size=16)),
        xaxis=dict(
            title="Bulan", tickformat="%b %Y", dtick="M3", tickangle=-30,
            range=[date_range_start, date_range_end + pd.DateOffset(months=1)],
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title="Total Qty / Sales", showgrid=True,
            gridcolor="rgba(0,0,0,0.06)", rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=440, margin=dict(t=80, b=60, l=60, r=20),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Ringkasan per Tahun", expanded=False):
        all_monthly = pd.concat([
            act_monthly.assign(Tipe="Actual"),
            fc_monthly.assign(Tipe="Forecast"),
        ], ignore_index=True)
        all_monthly["Tahun"] = all_monthly["Month"].dt.year
        pivot = (
            all_monthly.groupby(["Tahun", "Tipe"])["Total"]
            .sum().unstack(fill_value=0).reset_index()
        )
        for col in ["Actual", "Forecast"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot["Total"]    = pivot["Actual"] + pivot["Forecast"]
        pivot["Actual"]   = pivot["Actual"].apply(lambda v: f"{v:,.0f}")
        pivot["Forecast"] = pivot["Forecast"].apply(lambda v: f"{v:,.0f}")
        pivot["Total"]    = pivot["Total"].apply(lambda v: f"{v:,.0f}")
        st.dataframe(pivot, use_container_width=True, hide_index=True)


# =====================================================
# HELPER: PARSE & NORMALIZE CSV
# =====================================================

def load_and_normalize_csv(uploaded_file) -> pd.DataFrame | None:
    """
    Baca CSV, deteksi separator (koma / titik koma / tab).
    Kembalikan None jika gagal.
    """
    try:
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        sample = raw_bytes[:4096].decode("utf-8", errors="replace")
        sep = ","
        if sample.count(";") > sample.count(","):
            sep = ";"
        elif sample.count("\t") > sample.count(","):
            sep = "\t"

        df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep)
        return df

    except Exception as e:
        st.error(f"❌ Gagal membaca CSV: {e}")
        return None


# ── Mapping bulan Indonesia → Inggris ────────────────────────────────────────
_BULAN_ID_MAP: dict[str, str] = {
    "Mei": "May",
    "Agu": "Aug",
    "Okt": "Oct",
    "Des": "Dec",
}


def _parse_tanggal(value) -> pd.Timestamp:
    """
    Parse satu nilai tanggal dengan support format Indonesia.
    Contoh: 'Jan-21' → 2021-01-01, 'Mei-21' → 2021-05-01, 'Agu-23' → 2023-08-01
    """
    s = str(value).strip()
    for id_name, en_name in _BULAN_ID_MAP.items():
        s = s.replace(id_name, en_name)
    try:
        return pd.to_datetime(s, format="%b-%y")
    except ValueError:
        pass
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def normalize_to_ds_y(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan DataFrame memiliki kolom 'ds' (datetime) dan 'y' (numerik).

    FIX: Support format tanggal Indonesia (Jan-21, Mei-21, Agu-21, Okt-21, Des-21)
    menggunakan _parse_tanggal() per-elemen, bukan pd.to_datetime() langsung
    yang tidak bisa mengenali nama bulan Indonesia.
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # ── Kolom tanggal → ds ────────────────────────────────────────────
    if "ds" not in cols_lower:
        date_candidates = [
            "date", "period", "bulan", "month", "tanggal",
            "time", "week", "year_month",
        ]
        found_date = None
        for cand in date_candidates:
            if cand in cols_lower:
                found_date = cols_lower[cand]
                break
        if found_date is None:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    found_date = c
                    break
        if found_date is None:
            found_date = df.columns[0]

        df = df.rename(columns={found_date: "ds"})
    else:
        df = df.rename(columns={cols_lower["ds"]: "ds"})

    # FIX: gunakan _parse_tanggal per-elemen, bukan pd.to_datetime langsung
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = df["ds"].apply(_parse_tanggal)
    # Jika sudah datetime, tidak perlu diparse ulang

    # ── Kolom sales → y ───────────────────────────────────────────────
    cols_lower2 = {c.lower(): c for c in df.columns}
    if "y" not in cols_lower2:
        sales_candidates = [
            "sales", "qty", "quantity", "penjualan",
            "volume", "amount", "nilai", "demand", "units",
        ]
        found_sales = None
        for cand in sales_candidates:
            if cand in cols_lower2:
                found_sales = cols_lower2[cand]
                break
        if found_sales is None:
            skip = {"ds", "model", "kyb no", "category", "method"}
            for c in df.columns:
                if c.lower() in skip:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    found_sales = c
                    break
        if found_sales is None:
            found_sales = df.columns[-1]

        df = df.rename(columns={found_sales: "y"})
    else:
        df = df.rename(columns={cols_lower2["y"]: "y"})

    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)

    return df


# =====================================================
# HOME PAGE
# =====================================================

def home():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Sales Forecasting Dashboard</h1>
        <p>Analisis dan prediksi penjualan produk berbasis machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Data Analysis</div>
            <ul>
                <li>Upload dataset CSV penjualan</li>
                <li>Preview &amp; Exploratory Data Analysis (EDA)</li>
                <li>Analisis tren penjualan</li>
                <li>Klasifikasi produk: Stable, Declining, Volatile, Intermittent, Discontinued</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">🔮 Forecasting</div>
            <ul>
                <li>XGBoost, LightGBM, CatBoost</li>
                <li>Random Forest, Extra Trees, Gradient Boosting</li>
                <li>ElasticNet, Prophet, BiLSTM</li>
            </ul>
            <p style="color:#64748b;font-size:0.88rem;margin-top:0.5rem">
            Model direkomendasikan sesuai karakteristik produk.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <div class="card-title">📐 MAPE Analysis</div>
            <ul>
                <li>Evaluasi akurasi forecast per produk</li>
                <li>Grafik Forecast vs Actual per item</li>
                <li>Ringkasan akurasi keseluruhan</li>
                <li>Export hasil ke CSV</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">📄 Format CSV yang Didukung</div>
            <p>Kolom minimal yang diperlukan:</p>
            <ul>
                <li><code>ds</code> atau <code>date/tanggal/bulan</code> — kolom tanggal</li>
                <li><code>y</code> atau <code>sales/qty/penjualan</code> — kolom nilai</li>
                <li><code>Model</code> — nama/kode produk (opsional, jika multi-produk)</li>
            </ul>
            <p style="color:#64748b;font-size:0.88rem;margin-top:0.5rem">
            ✅ Support format tanggal Indonesia: Jan-21, Mei-21, Agu-21, Okt-21, Des-21
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info("📌 Gunakan menu sidebar untuk berpindah halaman.")


# =====================================================
# FORECAST PAGE
# =====================================================

def forecasting_page():
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Multi-Product Forecasting</h1>
        <p>Upload CSV → Pilih model → Generate forecast</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        key="forecast_uploader",
        help="File CSV dengan kolom tanggal dan kolom penjualan/qty.",
    )

    if uploaded_file is None:
        st.info("📂 Upload file CSV terlebih dahulu untuk memulai.")

        with st.expander("📋 Contoh format CSV", expanded=True):
            st.markdown("""
**Format 1 — Single product:**
```
ds,y
2022-01-01,150
2022-02-01,170
```

**Format 2 — Multi product (format Indonesia):**
```
Model,KYB No,Date,Year,Sales
YAMAHA ABC,12200-xxx,Jan-21,2021,2600
YAMAHA ABC,12200-xxx,Mei-21,2021,3100
```
""")
        return

    ok, err_msg = validate_upload(uploaded_file)
    if not ok:
        st.error(f"❌ File ditolak: {err_msg}")
        audit_log("UPLOAD_REJECTED", err_msg)
        return

    raw_df = load_and_normalize_csv(uploaded_file)
    if raw_df is None:
        return

    ok_df, err_df = validate_dataframe(raw_df)
    if not ok_df:
        st.error(f"❌ Data ditolak: {err_df}")
        audit_log("DATA_REJECTED", err_df)
        return

    raw_df = drop_pii_columns(raw_df)
    audit_log("UPLOAD_OK", f"rows={len(raw_df)} cols={len(raw_df.columns)}")

    try:
        raw_df = normalize_to_ds_y(raw_df)
    except Exception as e:
        st.error(f"❌ Gagal normalisasi kolom: {e}")
        return

    with st.expander("🔍 Pratinjau Data & Kolom Terdeteksi", expanded=False):
        st.write(f"**Kolom dalam file:** {raw_df.columns.tolist()}")
        st.write(f"**Jumlah baris:** {len(raw_df):,}")
        nat_count = raw_df["ds"].isna().sum()
        if nat_count > 0:
            st.warning(f"⚠️ {nat_count} baris dengan tanggal tidak valid (akan dibuang)")
        st.dataframe(raw_df.head(10), use_container_width=True)

    try:
        df = preprocess_data(raw_df)
    except Exception as e:
        st.error(f"❌ Error saat preprocessing: {e}")
        audit_log("PREPROCESS_ERROR", str(e))
        return

    if "ds" not in df.columns or "y" not in df.columns:
        st.error(
            "❌ Setelah preprocessing, kolom `ds` atau `y` tidak ditemukan. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )
        return

    col_date  = "ds"
    col_sales = "y"
    st.session_state["col_date"]   = col_date
    st.session_state["col_sales"]  = col_sales
    st.session_state["history_df"] = df

    if "Model" not in df.columns:
        df["Model"] = "Product"
        st.session_state["history_df"] = df

    try:
        classify_df = classify_items(df)
    except Exception as e:
        st.error(f"❌ Error saat klasifikasi produk: {e}")
        audit_log("CLASSIFY_ERROR", str(e))
        return

    if classify_df.empty:
        st.warning("⚠️ Tidak ada produk yang dapat diklasifikasi.")
        return

    st.success(f"✅ **{len(classify_df)} produk** terdeteksi. Pilih model lalu klik Generate Forecast.")

    st.subheader("🎛️ Pilih Model per Produk")

    selected_models: dict[str, str] = {}
    cols_header = st.columns([3, 2, 3])
    cols_header[0].markdown("**Produk**")
    cols_header[1].markdown("**Kategori**")
    cols_header[2].markdown("**Model**")
    st.divider()

    for _, row in classify_df.iterrows():
        model_name      = row["Model"]
        category        = row.get("Category", "Stable")
        recommendations = RECOMMENDED_MODELS.get(category, ["Random Forest"])

        c1, c2, c3 = st.columns([3, 2, 3])
        c1.write(f"**{model_name}**")
        c2.write(f"`{category}`")
        with c3:
            selected_models[model_name] = st.selectbox(
                "Model",
                recommendations,
                key=f"model_{model_name}",
                label_visibility="collapsed",
            )

    st.divider()
    periods = st.slider("📅 Forecast Horizon (Bulan)", 1, 36, 12, key="fc_periods")

    if st.button("🚀 Generate Forecast Semua Produk", key="btn_generate", type="primary"):

        progress_bar       = st.progress(0, text="Memulai proses forecasting...")
        status_placeholder = st.empty()
        results  = []
        products = classify_df["Model"].tolist()

        for i, product in enumerate(products):
            status_placeholder.info(f"⏳ Memproses **{product}** ({i+1}/{len(products)})…")

            item_df = df[df["Model"] == product].copy()
            method  = selected_models.get(product, "Random Forest")

            if "ds" not in item_df.columns or "y" not in item_df.columns:
                st.warning(f"⚠️ Lewati {product}: kolom ds/y tidak ditemukan.")
                progress_bar.progress((i + 1) / len(products))
                continue

            if len(item_df) < 5:
                st.warning(f"⚠️ Lewati {product}: data terlalu sedikit ({len(item_df)} baris).")
                progress_bar.progress((i + 1) / len(products))
                continue

            try:
                forecast = forecast_item(item_df, method, periods)
            except Exception as e:
                st.warning(f"⚠️ Gagal forecast **{product}**: {e}")
                progress_bar.progress((i + 1) / len(products))
                continue

            if forecast is None or forecast.empty:
                st.warning(f"⚠️ Forecast kosong untuk **{product}**.")
                progress_bar.progress((i + 1) / len(products))
                continue

            try:
                mape = evaluate_model(item_df, method, test_period=min(12, len(item_df) // 3))
            except Exception:
                mape = np.nan

            forecast["Model"]    = product
            forecast["KYB No"]   = item_df["KYB No"].iloc[0] if "KYB No" in item_df.columns else "-"
            forecast["Category"] = (
                classify_df.loc[classify_df["Model"] == product, "Category"].values[0]
                if "Category" in classify_df.columns else "Unknown"
            )
            forecast["Method"] = method
            forecast["MAPE"]   = mape if mape is not None else np.nan
            results.append(forecast)

            progress_bar.progress((i + 1) / len(products), text=f"Selesai: {product}")

        status_placeholder.empty()

        if not results:
            st.error("❌ Tidak ada hasil forecast. Periksa format data atau pilih model lain.")
            return

        final_forecast = pd.concat(results, ignore_index=True)
        st.session_state["final_forecast"] = final_forecast

        audit_log(
            "FORECAST_GENERATED",
            f"products={final_forecast['Model'].nunique()} rows={len(final_forecast)}",
        )

        st.success(f"✅ Forecast berhasil untuk **{final_forecast['Model'].nunique()}** produk!")

        st.subheader("📋 Hasil Forecast")
        safe_dataframe_display(final_forecast, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Produk",         final_forecast["Model"].nunique())
        c2.metric("Total Forecast Sales", f"{final_forecast['Forecast'].sum():,.0f}")
        c3.metric("Total Baris",          len(final_forecast))

        st.divider()

        plot_monthly_total(
            history_df=df,
            final_forecast=final_forecast,
            col_date=col_date,
            col_sales=col_sales,
        )

        st.divider()

        st.subheader("📊 Grafik Forecast vs Actual per Produk")

        for product in final_forecast["Model"].unique():
            act_raw = df[df["Model"] == product][["ds", "y"]].copy()
            act_raw["ds"] = pd.to_datetime(act_raw["ds"])

            fc_raw = final_forecast[final_forecast["Model"] == product][
                ["Forecast Date", "Forecast"]
            ].copy()
            fc_raw["Forecast Date"] = pd.to_datetime(fc_raw["Forecast Date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=act_raw["ds"], y=act_raw["y"],
                mode="lines+markers", name="Actual",
                line=dict(color="#4C9BE8", width=2), marker=dict(size=4),
                hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=fc_raw["Forecast Date"], y=fc_raw["Forecast"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#F97316", width=2, dash="dash"), marker=dict(size=4),
                hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
            ))
            if not act_raw.empty and not fc_raw.empty:
                fig.add_vline(
                    x=act_raw["ds"].max(),
                    line_dash="dot", line_color="gray",
                    annotation_text="Forecast Start",
                    annotation_position="top right",
                )
            fig.update_layout(
                title=f"📦 {product}",
                xaxis_title="Tanggal", yaxis_title="Sales / Qty",
                legend=dict(orientation="h", y=1.12),
                height=350, margin=dict(t=60, b=40, l=40, r=20),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        csv_bytes = final_forecast.to_csv(index=False).encode("utf-8")
        secure_download_button(
            label="📥 Download Forecast (CSV)",
            data=csv_bytes,
            file_name="forecast_all_products.csv",
            mime="text/csv",
            action_name="EXPORT_FORECAST",
        )


# =====================================================
# MAPE PAGE
# =====================================================

def mape_page():
    st.markdown("""
    <div class="main-header">
        <h1>📐 MAPE Analysis</h1>
        <p>Evaluasi akurasi forecast per produk</p>
    </div>
    """, unsafe_allow_html=True)

    final_forecast: pd.DataFrame | None = st.session_state.get("final_forecast")
    history_df:     pd.DataFrame | None = st.session_state.get("history_df")
    col_date:       str | None          = st.session_state.get("col_date")
    col_sales:      str | None          = st.session_state.get("col_sales")

    if final_forecast is None or history_df is None:
        st.warning(
            "⚠️ Belum ada hasil forecast. "
            "Silakan jalankan forecasting terlebih dahulu di halaman **Forecast**."
        )
        return

    if col_date is None:
        col_date = "ds"
    if col_sales is None:
        col_sales = "y"

    fc_date_col = "Forecast Date"
    fc_val_col  = "Forecast"

    mape_rows = []
    for product in final_forecast["Model"].unique():
        fc       = final_forecast[final_forecast["Model"] == product]
        mape_val = fc["MAPE"].iloc[0]
        category = fc["Category"].iloc[0] if "Category" in fc.columns else "-"
        method   = fc["Method"].iloc[0]   if "Method"   in fc.columns else "-"

        mape_rows.append({
            "Produk"  : product,
            "Kategori": category,
            "Metode"  : method,
            "MAPE (%)": round(float(mape_val), 2) if pd.notna(mape_val) else np.nan,
            "Akurasi" : get_mape_label(mape_val) if pd.notna(mape_val) else "N/A",
            "Status"  : mape_color(mape_val)      if pd.notna(mape_val) else "⚪",
        })

    mape_df = pd.DataFrame(mape_rows)
    valid   = mape_df["MAPE (%)"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata MAPE",    f"{valid.mean():.2f} %" if len(valid) else "N/A")
    c2.metric("MAPE Terbaik",      f"{valid.min():.2f} %"  if len(valid) else "N/A")
    c3.metric("MAPE Terburuk",     f"{valid.max():.2f} %"  if len(valid) else "N/A")
    c4.metric("Produk Dievaluasi", len(mape_df))

    st.divider()

    st.subheader("📋 Tabel MAPE per Produk")
    safe_dataframe_display(
        mape_df,
        use_container_width=True,
        column_config={
            "MAPE (%)": st.column_config.NumberColumn("MAPE (%)", format="%.2f %%"),
            "Status":   st.column_config.TextColumn("Status", width="small"),
        },
    )

    st.divider()

    st.subheader("📊 Bar Chart MAPE per Produk")
    plot_df = mape_df.dropna(subset=["MAPE (%)"]).sort_values("MAPE (%)", ascending=True)

    if plot_df.empty:
        st.info("Tidak ada data MAPE untuk ditampilkan.")
    else:
        color_map = {
            "Sangat Baik": "#22c55e",
            "Baik":        "#3b82f6",
            "Cukup":       "#f59e0b",
            "Kurang":      "#ef4444",
        }
        plot_df = plot_df.copy()
        plot_df["_color"] = plot_df["Akurasi"].map(color_map).fillna("#94a3b8")

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
            xaxis_title="MAPE (%)", yaxis_title="",
            height=max(300, len(plot_df) * 38),
            margin=dict(t=20, b=40, l=10, r=60),
            xaxis=dict(range=[0, plot_df["MAPE (%)"].max() * 1.3]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

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
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=fc["_date_"], y=fc[fc_val_col],
            mode="lines+markers", name="Forecast",
            line=dict(color="#F97316", width=2.5, dash="dash"), marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
        ))
        if not act.empty:
            fig2.add_vline(
                x=act["_date_"].max(), line_dash="dot", line_color="gray",
                annotation_text="Forecast Start", annotation_position="top right",
            )

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
            xaxis_title="Tanggal", yaxis_title="Sales / Qty",
            legend=dict(orientation="h", y=1.15),
            height=420, margin=dict(t=80, b=40, l=40, r=20),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    export_df  = mape_df.drop(columns=["_color", "Status"], errors="ignore")
    csv_bytes  = export_df.to_csv(index=False).encode("utf-8")
    secure_download_button(
        label="📥 Download MAPE (CSV)",
        data=csv_bytes,
        file_name="mape_analysis.csv",
        mime="text/csv",
        action_name="EXPORT_MAPE",
    )


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📈 Navigation")

page = st.sidebar.radio(
    "Pilih Halaman",
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
