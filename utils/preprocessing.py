"""
utils/preprocessing.py
======================
Preprocessing dataset untuk forecasting.

Asumsi input: DataFrame sudah memiliki kolom 'ds' (tanggal) dan 'y' (numerik)
setelah dinormalisasi oleh normalize_to_ds_y() di app.py.

Jika kolom 'Model' tidak ada, akan dibuat dengan nilai default 'Product'.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing lengkap sebelum klasifikasi & forecasting.

    Langkah:
    1. Pastikan kolom ds & y ada
    2. Parse ds ke datetime, y ke numerik
    3. Hapus baris dengan ds atau y NaN
    4. Tambahkan kolom Model jika belum ada
    5. Resample ke frekuensi bulanan per produk (jumlahkan y)
    6. Isi gap tanggal yang hilang dengan 0
    7. Reset index

    Returns
    -------
    DataFrame dengan kolom minimal: ds, y, Model
    """

    df = df.copy()

    # ── 1. Pastikan kolom wajib ada ───────────────────────────────────
    if "ds" not in df.columns:
        raise ValueError(
            "Kolom 'ds' tidak ditemukan. "
            "Pastikan file CSV memiliki kolom tanggal yang dapat dikenali "
            "(ds / date / tanggal / bulan / month)."
        )
    if "y" not in df.columns:
        raise ValueError(
            "Kolom 'y' tidak ditemukan. "
            "Pastikan file CSV memiliki kolom penjualan/qty yang dapat dikenali "
            "(y / sales / qty / penjualan / demand / volume)."
        )

    # ── 2. Konversi tipe ──────────────────────────────────────────────
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")

    # ── 3. Buang baris invalid ────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["ds", "y"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"preprocess_data: {n_dropped} baris dibuang karena ds/y tidak valid.",
            stacklevel=2,
        )

    if df.empty:
        raise ValueError(
            "Tidak ada data valid setelah preprocessing. "
            "Periksa format kolom tanggal dan kolom sales."
        )

    # ── 4. Tambahkan kolom Model jika belum ada ───────────────────────
    if "Model" not in df.columns:
        df["Model"] = "Product"

    df["Model"] = df["Model"].astype(str).str.strip()

    # ── 5. Normalisasi ke awal bulan (period month) ───────────────────
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()

    # ── 6. Resample: jumlahkan y per (Model, ds) ─────────────────────
    #       Pertahankan kolom tambahan lain (KYB No, Category, dst.)
    extra_cols = [
        c for c in df.columns
        if c not in {"ds", "y", "Model"}
    ]

    # Ambil nilai representatif untuk kolom extra (first per group)
    if extra_cols:
        extra_df = (
            df.groupby(["Model", "ds"])[extra_cols]
            .first()
            .reset_index()
        )

    agg_df = (
        df.groupby(["Model", "ds"])["y"]
        .sum()
        .reset_index()
    )

    if extra_cols:
        agg_df = agg_df.merge(extra_df, on=["Model", "ds"], how="left")

    # ── 7. Isi gap tanggal bulanan yang hilang dengan 0 ───────────────
    filled_parts = []

    for model_name, grp in agg_df.groupby("Model"):
        grp = grp.sort_values("ds").set_index("ds")

        # Full date range
        full_idx = pd.date_range(
            start=grp.index.min(),
            end=grp.index.max(),
            freq="MS",
        )

        grp = grp.reindex(full_idx)
        grp["y"] = grp["y"].fillna(0)
        grp["Model"] = model_name

        # Isi kolom extra yang kosong
        for col in extra_cols:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()

        grp = grp.reset_index().rename(columns={"index": "ds"})
        filled_parts.append(grp)

    if not filled_parts:
        raise ValueError("Tidak ada data setelah resample.")

    result = pd.concat(filled_parts, ignore_index=True)
    result = result.sort_values(["Model", "ds"]).reset_index(drop=True)

    return result
