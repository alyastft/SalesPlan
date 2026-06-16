"""
utils/classification.py
=======================
Klasifikasi produk berdasarkan pola historis penjualan.

Kategori:
- Stable       : Penjualan konsisten, CV rendah
- Declining    : Tren menurun signifikan
- Volatile     : Variabilitas tinggi tapi tidak banyak nol
- Intermittent : Banyak periode nol (demand jarang)
- Discontinued : Tidak ada penjualan dalam N bulan terakhir

Input: DataFrame dengan kolom 'ds', 'y', 'Model'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Threshold klasifikasi ────────────────────────────────────────────────────
ZERO_RATIO_THRESHOLD    = 0.40   # > 40% periode nol → Intermittent
CV_STABLE_THRESHOLD     = 0.50   # CV < 50% → kandidat Stable
DECLINE_THRESHOLD       = -0.10  # slope normalised < -10% → Declining
DISCONTINUED_MONTHS     = 3      # Tidak ada penjualan N bulan terakhir


def _classify_single(grp: pd.DataFrame) -> str:
    """
    Klasifikasi satu produk berdasarkan data historisnya.

    Parameter
    ---------
    grp : DataFrame satu produk, sudah di-sort ascending by ds,
          minimal kolom 'y'.

    Returns
    -------
    str  : label kategori
    """
    y = grp["y"].fillna(0).values

    if len(y) == 0:
        return "Discontinued"

    total    = y.sum()
    mean_val = y.mean()

    # ── Discontinued: tidak ada penjualan dalam N bulan terakhir ──────
    recent = y[-DISCONTINUED_MONTHS:] if len(y) >= DISCONTINUED_MONTHS else y
    if recent.sum() == 0 and total == 0:
        return "Discontinued"
    if recent.sum() == 0 and total > 0:
        return "Discontinued"

    # ── Intermittent: banyak periode nol ──────────────────────────────
    zero_ratio = (y == 0).sum() / len(y)
    if zero_ratio > ZERO_RATIO_THRESHOLD:
        return "Intermittent"

    # ── Coefficient of Variation ──────────────────────────────────────
    std_val = y.std()
    cv      = std_val / mean_val if mean_val > 0 else 0

    # ── Declining: hitung slope linear ───────────────────────────────
    if len(y) >= 4:
        t     = np.arange(len(y))
        slope = np.polyfit(t, y, 1)[0]
        # Normalkan terhadap mean agar skala-agnostik
        norm_slope = slope / mean_val if mean_val > 0 else 0
        if norm_slope < DECLINE_THRESHOLD:
            return "Declining"

    # ── Stable vs Volatile ────────────────────────────────────────────
    if cv <= CV_STABLE_THRESHOLD:
        return "Stable"

    return "Volatile"


def classify_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Klasifikasi semua produk dalam DataFrame.

    Parameter
    ---------
    df : DataFrame dengan kolom 'ds', 'y', 'Model'.

    Returns
    -------
    DataFrame dengan kolom ['Model', 'Category', 'DataPoints', 'MeanSales', 'CV', 'ZeroRatio']
    """
    if "Model" not in df.columns:
        raise ValueError("Kolom 'Model' tidak ditemukan di DataFrame.")
    if "y" not in df.columns:
        raise ValueError("Kolom 'y' tidak ditemukan di DataFrame.")

    records = []

    for model_name, grp in df.groupby("Model"):
        grp  = grp.sort_values("ds") if "ds" in grp.columns else grp
        y    = grp["y"].fillna(0).values

        category   = _classify_single(grp)
        mean_val   = float(y.mean())    if len(y) > 0 else 0.0
        std_val    = float(y.std())     if len(y) > 0 else 0.0
        cv         = std_val / mean_val if mean_val > 0 else 0.0
        zero_ratio = float((y == 0).sum() / len(y)) if len(y) > 0 else 1.0

        records.append({
            "Model":      model_name,
            "Category":   category,
            "DataPoints": len(y),
            "MeanSales":  round(mean_val, 2),
            "CV":         round(cv, 3),
            "ZeroRatio":  round(zero_ratio, 3),
        })

    return pd.DataFrame(records)
