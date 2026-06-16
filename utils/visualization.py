"""
utils/visualization.py
======================
Fungsi visualisasi reusable untuk Sales Forecasting Dashboard.
Semua fungsi mengembalikan go.Figure atau langsung render ke Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ── Palet warna konsisten ─────────────────────────────────────────────────────
COLOR_ACTUAL   = "#4C9BE8"   # biru
COLOR_FORECAST = "#F97316"   # oranye
COLOR_GOOD     = "#22c55e"   # hijau
COLOR_WARN     = "#f59e0b"   # kuning
COLOR_BAD      = "#ef4444"   # merah
COLOR_GRID     = "rgba(0,0,0,0.06)"
BG_TRANSPARENT = "rgba(0,0,0,0)"

CATEGORY_COLORS: dict[str, str] = {
    "Stable":       "#22c55e",
    "Declining":    "#ef4444",
    "Volatile":     "#f59e0b",
    "Intermittent": "#8b5cf6",
    "Discontinued": "#6b7280",
}

MAPE_COLORS: dict[str, str] = {
    "Sangat Baik": "#22c55e",
    "Baik":        "#3b82f6",
    "Cukup":       "#f59e0b",
    "Kurang":      "#ef4444",
    "N/A":         "#94a3b8",
}


# FORECAST CHARTS

def plot_forecast_vs_actual(
    actual_df:   pd.DataFrame,
    forecast_df: pd.DataFrame,
    product:     str,
    col_date:    str = "ds",
    col_sales:   str = "y",
    mape:        float | None = None,
    height:      int = 380,
) -> go.Figure:
    """
    Grafik line Actual vs Forecast untuk satu produk.

    Parameters
    ----------
    actual_df   : DataFrame historis dengan kolom col_date dan col_sales
    forecast_df : DataFrame forecast dengan kolom 'Forecast Date' dan 'Forecast'
    product     : nama produk (untuk judul)
    mape        : nilai MAPE opsional untuk subtitle
    """
    act = actual_df.copy()
    act["_date_"] = pd.to_datetime(act[col_date])

    fc = forecast_df.copy()
    fc["_date_"] = pd.to_datetime(fc["Forecast Date"])

    fig = go.Figure()

    # Area actual
    if not act.empty:
        fig.add_trace(go.Scatter(
            x=act["_date_"], y=act[col_sales],
            mode="lines+markers", name="Actual",
            fill="tozeroy",
            fillcolor=f"rgba(76,155,232,0.08)",
            line=dict(color=COLOR_ACTUAL, width=2.5),
            marker=dict(size=5, color=COLOR_ACTUAL),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))

    # Line forecast
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["_date_"], y=fc["Forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
            marker=dict(size=5, color=COLOR_FORECAST),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
        ))

    # Garis batas actual/forecast
    if not act.empty:
        split_x = act["_date_"].max()
        fig.add_vline(
            x=split_x, line_dash="dot", line_color="gray", line_width=1.5,
            annotation_text="Forecast Start",
            annotation_position="top right",
            annotation_font=dict(size=10, color="gray"),
        )

    # Subtitle MAPE
    subtitle = ""
    if mape is not None and not np.isnan(mape):
        label = _mape_label(mape)
        subtitle = f"<sup style='color:gray'>MAPE: {mape:.2f}% — {label}</sup>"

    fig.update_layout(
        title=dict(text=f"{product}{('<br>' + subtitle) if subtitle else ''}"),
        xaxis=dict(
            title="Tanggal", tickformat="%b %Y",
            showgrid=True, gridcolor=COLOR_GRID,
        ),
        yaxis=dict(
            title="Sales / Qty", rangemode="tozero",
            showgrid=True, gridcolor=COLOR_GRID,
        ),
        legend=dict(orientation="h", y=1.12, xanchor="left", x=0),
        height=height,
        margin=dict(t=70, b=40, l=50, r=20),
        hovermode="x unified",
        plot_bgcolor=BG_TRANSPARENT,
        paper_bgcolor=BG_TRANSPARENT,
    )
    return fig


def plot_monthly_total(
    history_df:    pd.DataFrame,
    final_forecast: pd.DataFrame,
    col_date:      str = "ds",
    col_sales:     str = "y",
    date_start:    str = "2024-01-01",
    date_end:      str = "2027-12-01",
) -> None:
    """
    Tampilkan grafik total penjualan bulanan semua produk (Actual + Forecast)
    langsung ke Streamlit, lengkap dengan metrik dan tabel ringkasan tahunan.
    """
    st.subheader("📅 Total Penjualan per Bulan — Semua Produk")

    date_range_start = pd.Timestamp(date_start)
    date_range_end   = pd.Timestamp(date_end)

    # ── Actual monthly ────────────────────────────────────────────────
    act = history_df[[col_date, col_sales]].copy()
    act["Month"] = pd.to_datetime(act[col_date]).dt.to_period("M").dt.to_timestamp()
    act_monthly = (
        act.groupby("Month")[col_sales].sum()
        .reset_index().rename(columns={col_sales: "Total"})
    )
    act_monthly = act_monthly[
        (act_monthly["Month"] >= date_range_start)
        & (act_monthly["Month"] <= date_range_end)
    ].sort_values("Month")

    # ── Forecast monthly ──────────────────────────────────────────────
    fc = final_forecast[["Forecast Date", "Forecast"]].copy()
    fc["Month"] = pd.to_datetime(fc["Forecast Date"]).dt.to_period("M").dt.to_timestamp()
    fc_monthly = (
        fc.groupby("Month")["Forecast"].sum()
        .reset_index().rename(columns={"Forecast": "Total"})
    )
    fc_monthly = fc_monthly[
        (fc_monthly["Month"] >= date_range_start)
        & (fc_monthly["Month"] <= date_range_end)
    ].sort_values("Month")

    if act_monthly.empty and fc_monthly.empty:
        st.info("Tidak ada data dalam rentang yang dipilih.")
        return

    split_date = act_monthly["Month"].max() if not act_monthly.empty else None

    # ── Metrik ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Actual",           f"{act_monthly['Total'].sum():,.0f}"  if not act_monthly.empty else "—")
    c2.metric("Total Forecast",         f"{fc_monthly['Total'].sum():,.0f}"   if not fc_monthly.empty else "—")
    c3.metric("Rata-rata Actual/bln",   f"{act_monthly['Total'].mean():,.0f}" if not act_monthly.empty else "—")
    c4.metric("Rata-rata Forecast/bln", f"{fc_monthly['Total'].mean():,.0f}"  if not fc_monthly.empty else "—")

    # ── Grafik ────────────────────────────────────────────────────────
    fig = go.Figure()

    if split_date is not None and not fc_monthly.empty:
        fig.add_vrect(
            x0=split_date, x1=date_range_end,
            fillcolor="rgba(249,115,22,0.06)", layer="below", line_width=0,
        )

    if not act_monthly.empty:
        fig.add_trace(go.Scatter(
            x=act_monthly["Month"], y=act_monthly["Total"],
            mode="lines+markers", name="Actual",
            fill="tozeroy", fillcolor="rgba(76,155,232,0.12)",
            line=dict(color=COLOR_ACTUAL, width=2.5), marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))

    if not fc_monthly.empty:
        fig.add_trace(go.Scatter(
            x=fc_monthly["Month"], y=fc_monthly["Total"],
            mode="lines+markers", name="Forecast",
            line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"), marker=dict(size=5),
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
            showgrid=True, gridcolor=COLOR_GRID,
        ),
        yaxis=dict(
            title="Total Qty / Sales", showgrid=True,
            gridcolor=COLOR_GRID, rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=440, margin=dict(t=80, b=60, l=60, r=20),
        hovermode="x unified",
        plot_bgcolor=BG_TRANSPARENT, paper_bgcolor=BG_TRANSPARENT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabel ringkasan per tahun ─────────────────────────────────────
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


# MAPE CHARTS

def plot_mape_bar(mape_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart horizontal MAPE per produk, diurutkan dari terendah ke tertinggi.
    mape_df harus punya kolom: ['Produk', 'MAPE (%)', 'Akurasi']
    """
    plot_df = mape_df.dropna(subset=["MAPE (%)"]).sort_values("MAPE (%)", ascending=True).copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data MAPE.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    plot_df["_color"] = plot_df["Akurasi"].map(MAPE_COLORS).fillna("#94a3b8")

    fig = go.Figure(go.Bar(
        x=plot_df["MAPE (%)"],
        y=plot_df["Produk"],
        orientation="h",
        marker_color=plot_df["_color"],
        text=plot_df["MAPE (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="MAPE (%)", yaxis_title="",
        height=max(300, len(plot_df) * 38),
        margin=dict(t=20, b=40, l=10, r=80),
        xaxis=dict(range=[0, plot_df["MAPE (%)"].max() * 1.3]),
        plot_bgcolor=BG_TRANSPARENT,
        paper_bgcolor=BG_TRANSPARENT,
    )
    return fig


def plot_mape_gauge(mape: float, product: str) -> go.Figure:
    """Gauge chart untuk satu nilai MAPE."""
    label = _mape_label(mape) if not np.isnan(mape) else "N/A"
    color = MAPE_COLORS.get(label, "#94a3b8")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mape if not np.isnan(mape) else 0,
        title=dict(text=f"{product}<br><span style='font-size:0.8em;color:gray'>{label}</span>"),
        number=dict(suffix="%", font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color=color, thickness=0.3),
            steps=[
                dict(range=[0, 10],  color="rgba(34,197,94,0.15)"),
                dict(range=[10, 20], color="rgba(59,130,246,0.15)"),
                dict(range=[20, 50], color="rgba(245,158,11,0.15)"),
                dict(range=[50, 100],color="rgba(239,68,68,0.15)"),
            ],
            threshold=dict(
                line=dict(color="red", width=2),
                thickness=0.75,
                value=50,
            ),
        ),
    ))
    fig.update_layout(height=260, margin=dict(t=60, b=20, l=30, r=30))
    return fig


# EDA CHARTS

def plot_trend(
    df:        pd.DataFrame,
    col_date:  str,
    col_sales: str,
    col_group: str | None = None,
    title:     str = "Tren Penjualan",
    height:    int = 400,
) -> go.Figure:
    """
    Line chart tren bulanan. Jika col_group diberikan, buat multi-line per group.
    """
    df = df.copy()
    df["_month_"] = pd.to_datetime(df[col_date]).dt.to_period("M").dt.to_timestamp()

    if col_group and col_group in df.columns:
        monthly = (
            df.groupby(["_month_", col_group])[col_sales]
            .sum().reset_index()
        )
        fig = px.line(
            monthly, x="_month_", y=col_sales, color=col_group,
            labels={"_month_": "Bulan", col_sales: "Total"},
            title=title,
        )
    else:
        monthly = df.groupby("_month_")[col_sales].sum().reset_index()
        fig = px.line(
            monthly, x="_month_", y=col_sales,
            labels={"_month_": "Bulan", col_sales: "Total"},
            title=title,
        )
        fig.update_traces(line=dict(color=COLOR_ACTUAL, width=2.5))

    fig.update_layout(
        height=height,
        xaxis=dict(tickformat="%b %Y", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, rangemode="tozero"),
        plot_bgcolor=BG_TRANSPARENT, paper_bgcolor=BG_TRANSPARENT,
        hovermode="x unified",
    )
    return fig


def plot_category_pie(classify_df: pd.DataFrame) -> go.Figure:
    """Pie chart distribusi kategori produk."""
    if "Category" not in classify_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Kolom Category tidak ditemukan.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    counts = classify_df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    colors = [CATEGORY_COLORS.get(c, "#94a3b8") for c in counts["Category"]]

    fig = go.Figure(go.Pie(
        labels=counts["Category"],
        values=counts["Count"],
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Distribusi Kategori Produk",
        height=380,
        margin=dict(t=60, b=20, l=20, r=20),
        showlegend=True,
    )
    return fig


def plot_top_products(
    df:        pd.DataFrame,
    col_sales: str,
    top_n:     int = 10,
) -> go.Figure:
    """Bar chart horizontal top N produk berdasarkan total penjualan."""
    if "Model" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Kolom Model tidak ditemukan.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    top_df = (
        df.groupby("Model")[col_sales]
        .sum().nlargest(top_n).reset_index()
        .sort_values(col_sales, ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=top_df[col_sales],
        y=top_df["Model"],
        orientation="h",
        marker=dict(
            color=top_df[col_sales],
            colorscale="Blues",
            showscale=False,
        ),
        text=top_df[col_sales].apply(lambda v: f"{v:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} Produk — Total {col_sales}",
        xaxis_title="Total", yaxis_title="",
        height=max(300, top_n * 32),
        margin=dict(t=50, b=40, l=10, r=80),
        plot_bgcolor=BG_TRANSPARENT, paper_bgcolor=BG_TRANSPARENT,
    )
    return fig


def plot_distribution(df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
    """Histogram distribusi satu kolom numerik."""
    fig = px.histogram(
        df, x=col, nbins=50,
        title=title or f"Distribusi {col}",
        color_discrete_sequence=[COLOR_ACTUAL],
    )
    fig.update_layout(
        height=340,
        plot_bgcolor=BG_TRANSPARENT, paper_bgcolor=BG_TRANSPARENT,
        bargap=0.05,
    )
    return fig


def plot_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    """Heatmap korelasi antar kolom numerik."""
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matriks Korelasi",
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=420, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def plot_external_features(ext_df: pd.DataFrame) -> None:
    """Tampilkan grafik fitur eksternal (Kurs & Produksi Motor) ke Streamlit."""
    if ext_df.empty:
        st.info("Data eksternal tidak tersedia.")
        return

    st.subheader("🌐 Fitur Eksternal")

    tabs = st.tabs(["Kurs USD/IDR", "Kurs JPY/IDR", "Produksi Sepeda Motor"])

    with tabs[0]:
        if "Kurs USD/IDR" in ext_df.columns:
            fig = px.line(ext_df, x="ds", y="Kurs USD/IDR",
                          title="Kurs USD/IDR per Bulan",
                          color_discrete_sequence=[COLOR_ACTUAL])
            fig.update_layout(height=320, plot_bgcolor=BG_TRANSPARENT,
                              paper_bgcolor=BG_TRANSPARENT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data Kurs USD tidak tersedia.")

    with tabs[1]:
        if "Kurs JPY/IDR" in ext_df.columns:
            fig = px.line(ext_df, x="ds", y="Kurs JPY/IDR",
                          title="Kurs JPY/IDR per Bulan",
                          color_discrete_sequence=[COLOR_FORECAST])
            fig.update_layout(height=320, plot_bgcolor=BG_TRANSPARENT,
                              paper_bgcolor=BG_TRANSPARENT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data Kurs JPY tidak tersedia.")

    with tabs[2]:
        motor_cols = [c for c in ["Domestik", "Ekspor"] if c in ext_df.columns]
        if motor_cols:
            fig = go.Figure()
            colors = [COLOR_ACTUAL, COLOR_FORECAST]
            for i, col in enumerate(motor_cols):
                fig.add_trace(go.Scatter(
                    x=ext_df["ds"], y=ext_df[col],
                    mode="lines+markers", name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig.update_layout(
                title="Produksi Sepeda Motor per Bulan",
                height=320, plot_bgcolor=BG_TRANSPARENT,
                paper_bgcolor=BG_TRANSPARENT, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data Produksi Motor tidak tersedia.")


# HELPER INTERNAL

def _mape_label(mape: float) -> str:
    if np.isnan(mape):
        return "N/A"
    if mape < 10:
        return "Sangat Baik"
    if mape < 20:
        return "Baik"
    if mape < 50:
        return "Cukup"
    return "Kurang"
