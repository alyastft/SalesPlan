"""
utils/external_features.py
==========================
Load fitur eksternal dari file Excel di folder assets/.
Jika file tidak tersedia, kembalikan DataFrame kosong (graceful fallback).

File yang dibutuhkan:
  - assets/KursUSD.xlsx    : kolom [Tanggal, Kurs USD/IDR]
  - assets/KursJPY.xlsx    : kolom [Tanggal, Kurs JPY/IDR]
  - assets/ProduksiSepedaMotorSeluruh.xlsx : kolom [Bulan, Domestik, Ekspor]
"""

from __future__ import annotations

import os
import warnings
import pandas as pd

# Lokasi folder assets relatif terhadap root project
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

_USD_PATH   = os.path.join(_ASSETS_DIR, "KursUSD.xlsx")
_JPY_PATH   = os.path.join(_ASSETS_DIR, "KursJPY.xlsx")
_MOTOR_PATH = os.path.join(_ASSETS_DIR, "ProduksiSepedaMotorSeluruh.xlsx")


def _read_excel_safe(path: str, label: str) -> pd.DataFrame | None:
    """Baca Excel dengan error handling; return None jika gagal."""
    if not os.path.exists(path):
        warnings.warn(f"[external_features] File tidak ditemukan: {path}", stacklevel=3)
        return None
    try:
        return pd.read_excel(path)
    except Exception as e:
        warnings.warn(f"[external_features] Gagal baca {label}: {e}", stacklevel=3)
        return None


def load_external_features() -> pd.DataFrame:
    """
    Gabungkan fitur eksternal (Kurs USD, Kurs JPY, Produksi Motor)
    menjadi satu DataFrame dengan kolom 'ds' (awal bulan).

    Returns
    -------
    DataFrame dengan kolom:
        ds, Kurs USD/IDR, Kurs JPY/IDR, Domestik, Ekspor

    Jika salah satu / semua file tidak tersedia,
    DataFrame yang dikembalikan hanya berisi kolom yang berhasil dibaca.
    Jika semua gagal, kembalikan DataFrame kosong dengan kolom lengkap.
    """

    empty_cols = ["ds", "Kurs USD/IDR", "Kurs JPY/IDR", "Domestik", "Ekspor"]
    parts: list[pd.DataFrame] = []

    # ── USD ──────────────────────────────────────────────────────────
    usd_raw = _read_excel_safe(_USD_PATH, "KursUSD")
    if usd_raw is not None:
        try:
            usd = usd_raw.copy()
            # Fleksibel: cari kolom tanggal
            date_col = _find_col(usd, ["tanggal", "date", "periode"])
            val_col  = _find_col(usd, ["kurs usd", "usd/idr", "kurs", "value"])
            usd["ds"] = pd.to_datetime(usd[date_col]).dt.to_period("M").dt.to_timestamp()
            usd = (
                usd.groupby("ds")[val_col]
                .mean()
                .reset_index()
                .rename(columns={val_col: "Kurs USD/IDR"})
            )
            parts.append(usd)
        except Exception as e:
            warnings.warn(f"[external_features] Proses USD gagal: {e}", stacklevel=2)

    # ── JPY ──────────────────────────────────────────────────────────
    jpy_raw = _read_excel_safe(_JPY_PATH, "KursJPY")
    if jpy_raw is not None:
        try:
            jpy = jpy_raw.copy()
            date_col = _find_col(jpy, ["tanggal", "date", "periode"])
            val_col  = _find_col(jpy, ["kurs jpy", "jpy/idr", "kurs", "value"])
            jpy["ds"] = pd.to_datetime(jpy[date_col]).dt.to_period("M").dt.to_timestamp()
            jpy = (
                jpy.groupby("ds")[val_col]
                .mean()
                .reset_index()
                .rename(columns={val_col: "Kurs JPY/IDR"})
            )
            parts.append(jpy)
        except Exception as e:
            warnings.warn(f"[external_features] Proses JPY gagal: {e}", stacklevel=2)

    # ── Produksi Motor ────────────────────────────────────────────────
    motor_raw = _read_excel_safe(_MOTOR_PATH, "ProduksiMotor")
    if motor_raw is not None:
        try:
            motor = motor_raw.copy()
            date_col = _find_col(motor, ["bulan", "tanggal", "date", "periode", "month"])
            dom_col  = _find_col(motor, ["domestik", "domestic", "dom"])
            exp_col  = _find_col(motor, ["ekspor", "export", "exp"])
            motor["ds"] = pd.to_datetime(motor[date_col]).dt.to_period("M").dt.to_timestamp()
            motor = (
                motor.groupby("ds")[[dom_col, exp_col]]
                .sum()
                .reset_index()
                .rename(columns={dom_col: "Domestik", exp_col: "Ekspor"})
            )
            parts.append(motor)
        except Exception as e:
            warnings.warn(f"[external_features] Proses Motor gagal: {e}", stacklevel=2)

    # ── Gabungkan ─────────────────────────────────────────────────────
    if not parts:
        return pd.DataFrame(columns=empty_cols)

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on="ds", how="outer")

    result = result.sort_values("ds").reset_index(drop=True)

    # Pastikan semua kolom ada (isi NaN jika tidak ada)
    for col in empty_cols:
        if col not in result.columns:
            result[col] = float("nan")

    return result


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Cari nama kolom dari daftar kandidat (case-insensitive).
    Raise ValueError jika tidak ditemukan.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        # Exact match
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
        # Partial match
        for col_lower, col_orig in cols_lower.items():
            if cand.lower() in col_lower:
                return col_orig
    raise ValueError(
        f"Kolom tidak ditemukan. Kandidat: {candidates}. "
        f"Kolom tersedia: {list(df.columns)}"
    )


def is_external_features_available() -> bool:
    """Cek apakah semua file eksternal tersedia."""
    return all(os.path.exists(p) for p in [_USD_PATH, _JPY_PATH, _MOTOR_PATH])
