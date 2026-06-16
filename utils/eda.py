"""
utils/eda.py
============
Halaman Exploratory Data Analysis (EDA).
Upload CSV → tampilkan statistik, tren, distribusi.
"""

from __future__ import annotations

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.security import validate_upload, validate_dataframe, drop_pii_columns, audit_log


def _load_csv(uploaded_file) -> pd.DataFrame | None:
    """Baca CSV dengan deteksi separator otomatis."""
    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        sample = raw[:4096].decode("utf-8", errors="replace")
        sep = ","
        if sample.count(";") > sample.count(","):
            sep = ";"
        elif sample.count("\t") > sample.count(","):
            sep = "\t"
        return pd.read_csv(io.BytesIO(raw), sep=sep)
    except Exception as e:
        st.error(f"❌ Gagal membaca CSV: {e}")
        return None


def show_eda() -> None:
    st.title("📊 Data Analysis & EDA")

    uploaded_file = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"],
        key="eda_uploader",
        help="File CSV dengan kolom tanggal dan penjualan.",
    )

    if uploaded_file is None:
        st.info("📂 Upload file CSV untuk memulai analisis.")
        return

    # Validasi
    ok, err = validate_upload(uploaded_file)
    if not ok:
        st.error(f"❌ {err}")
        return

    df = _load_csv(uploaded_file)
    if df is None:
        return

    ok_df, err_df = validate_dataframe(df)
    if not ok_df:
        st.error(f"❌ {err_df}")
        return

    df = drop_pii_columns(df)
    audit_log("EDA_UPLOAD", f"rows={len(df)} cols={len(df.columns)}")

    # ── Info dasar ────────────────────────────────────────────────────
    st.subheader("📋 Informasi Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris",  f"{len(df):,}")
    c2.metric("Jumlah Kolom",  len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())

    with st.expander("🔍 Preview Data (10 baris pertama)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📈 Statistik Deskriptif", expanded=False):
        st.dataframe(df.describe().T, use_container_width=True)

    with st.expander("❓ Missing Values per Kolom", expanded=False):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("✅ Tidak ada missing value!")
        else:
            st.dataframe(
                missing.reset_index().rename(columns={"index": "Kolom", 0: "Missing"}),
                use_container_width=True,
            )

    # ── Deteksi kolom tanggal & numerik ──────────────────────────────
    cols_lower = {c.lower(): c for c in df.columns}

    # Kolom tanggal
    date_candidates = ["ds", "date", "tanggal", "bulan", "month", "period", "time"]
    col_date = None
    for cand in date_candidates:
        if cand in cols_lower:
            col_date = cols_lower[cand]
            break
    if col_date is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                col_date = c
                break
            except Exception:
                pass

    # Kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Tren penjualan ────────────────────────────────────────────────
    if col_date and numeric_cols:
        st.subheader("📈 Tren Penjualan dari Waktu ke Waktu")

        try:
            df["_date_parsed_"] = pd.to_datetime(df[col_date], errors="coerce")
            df_valid = df.dropna(subset=["_date_parsed_"])

            if not df_valid.empty:
                y_col = st.selectbox(
                    "Pilih kolom nilai",
                    numeric_cols,
                    key="eda_y_col",
                )

                # Group by bulan jika ada kolom Model
                if "Model" in df.columns:
                    top_models = (
                        df_valid.groupby("Model")[y_col]
                        .sum()
                        .nlargest(10)
                        .index.tolist()
                    )
                    selected_models = st.multiselect(
                        "Filter Produk (kosongkan = semua, maks 10)",
                        options=top_models,
                        default=top_models[:5] if len(top_models) >= 5 else top_models,
                        key="eda_model_filter",
                    )
                    if selected_models:
                        df_plot = df_valid[df_valid["Model"].isin(selected_models)].copy()
                    else:
                        df_plot = df_valid.copy()

                    df_plot["_month_"] = df_plot["_date_parsed_"].dt.to_period("M").dt.to_timestamp()
                    monthly = (
                        df_plot.groupby(["_month_", "Model"])[y_col]
                        .sum()
                        .reset_index()
                    )
                    fig = px.line(
                        monthly, x="_month_", y=y_col, color="Model",
                        labels={"_month_": "Bulan", y_col: "Total"},
                        title=f"Tren {y_col} per Bulan per Produk",
                    )
                else:
                    df_valid["_month_"] = df_valid["_date_parsed_"].dt.to_period("M").dt.to_timestamp()
                    monthly = df_valid.groupby("_month_")[y_col].sum().reset_index()
                    fig = px.line(
                        monthly, x="_month_", y=y_col,
                        labels={"_month_": "Bulan", y_col: "Total"},
                        title=f"Tren {y_col} per Bulan",
                    )

                fig.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Gagal membuat grafik tren: {e}")

    # ── Distribusi ────────────────────────────────────────────────────
    if numeric_cols:
        st.subheader("📊 Distribusi Nilai")
        col_to_plot = st.selectbox(
            "Pilih kolom untuk distribusi",
            numeric_cols,
            key="eda_dist_col",
        )
        try:
            fig_dist = px.histogram(
                df, x=col_to_plot, nbins=50,
                title=f"Distribusi {col_to_plot}",
                color_discrete_sequence=["#4C9BE8"],
            )
            fig_dist.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Gagal membuat distribusi: {e}")

    # ── Korelasi ──────────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        with st.expander("🔗 Matriks Korelasi", expanded=False):
            try:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Matriks Korelasi Kolom Numerik",
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Gagal membuat korelasi: {e}")

    # ── Top Produk ────────────────────────────────────────────────────
    if "Model" in df.columns and numeric_cols:
        st.subheader("🏆 Top Produk berdasarkan Total Penjualan")
        try:
            y_col_top = st.selectbox(
                "Pilih kolom nilai",
                numeric_cols,
                key="eda_top_col",
            )
            top_n = st.slider("Tampilkan Top N", 5, 30, 10, key="eda_top_n")
            top_df = (
                df.groupby("Model")[y_col_top]
                .sum()
                .nlargest(top_n)
                .reset_index()
                .sort_values(y_col_top, ascending=True)
            )
            fig_top = px.bar(
                top_df, x=y_col_top, y="Model", orientation="h",
                title=f"Top {top_n} Produk — Total {y_col_top}",
                color=y_col_top, color_continuous_scale="Blues",
            )
            fig_top.update_layout(
                height=max(300, top_n * 30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_top, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Gagal membuat chart top produk: {e}")
