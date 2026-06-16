from __future__ import annotations
 
import pandas as pd
import numpy as np
import warnings
 
# ── Mapping nama bulan Indonesia → Inggris ────────────────────────────────────
_BULAN_ID: dict[str, str] = {
    "Mei": "May",
    "Agu": "Aug",
    "Okt": "Oct",
    "Des": "Dec",
}
 
# Kandidat nama kolom tanggal & sales (case-insensitive)
_DATE_CANDIDATES  = ["ds", "date", "tanggal", "bulan", "month", "period", "time", "periode"]
_SALES_CANDIDATES = ["y", "sales", "qty", "penjualan", "demand", "volume", "jumlah", "amount"]
 
 
def _parse_date_id(value: str) -> pd.Timestamp:
    """
    Parse tanggal dengan format fleksibel, termasuk format Indonesia.
 
    Contoh yang didukung:
    - 'Jan-21'  → 2021-01-01
    - 'Mei-21'  → 2021-05-01
    - 'Agu-23'  → 2023-08-01
    - '2021-01-01', '01/2021', dst.
    """
    s = str(value).strip()
    for id_name, en_name in _BULAN_ID.items():
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
    Normalisasi kolom DataFrame ke nama standar 'ds' dan 'y'.
 
    - Deteksi kolom tanggal dari _DATE_CANDIDATES (case-insensitive)
    - Deteksi kolom sales/qty dari _SALES_CANDIDATES
    - Jika kolom sudah bernama 'ds'/'y', tidak ada perubahan
    - Parse tanggal dengan _parse_date_id (support format Indonesia)
 
    Returns DataFrame dengan kolom 'ds' (Timestamp) dan 'y' (float),
    plus kolom lainnya dipertahankan.
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
 
    # ── Kolom tanggal ─────────────────────────────────────────────────
    if "ds" not in df.columns:
        found_date = None
        for cand in _DATE_CANDIDATES:
            if cand in cols_lower:
                found_date = cols_lower[cand]
                break
        if found_date is None:
            raise ValueError(
                "Kolom tanggal tidak ditemukan. "
                f"Kolom yang dicari: {_DATE_CANDIDATES}. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        df = df.rename(columns={found_date: "ds"})
 
    # ── Parse tanggal (termasuk format Indonesia) ─────────────────────
    # Gunakan is_datetime64_any_dtype agar kompatibel pandas 2+
    # (dtype 'str' tidak sama dengan object, tapi juga bukan datetime)
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = df["ds"].apply(_parse_date_id)
    # else: sudah datetime, tidak perlu diparse ulang
 
    # ── Kolom sales ───────────────────────────────────────────────────
    if "y" not in df.columns:
        found_sales = None
        for cand in _SALES_CANDIDATES:
            if cand in cols_lower:
                found_sales = cols_lower[cand]
                break
        if found_sales is None:
            raise ValueError(
                "Kolom penjualan tidak ditemukan. "
                f"Kolom yang dicari: {_SALES_CANDIDATES}. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        df = df.rename(columns={found_sales: "y"})
 
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
 
    return df
 
 
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing lengkap sebelum klasifikasi & forecasting.
 
    Langkah:
    0. Normalisasi nama kolom (ds, y) — handle format tanggal Indonesia
    1. Pastikan kolom ds & y ada
    2. Konversi ulang jika perlu (idempoten jika sudah datetime)
    3. Hapus baris dengan ds atau y NaN
    4. Tambahkan kolom Model jika belum ada
    5. Resample ke frekuensi bulanan per produk (jumlahkan y)
    6. Isi gap tanggal yang hilang dengan 0
    7. Reset index
 
    Returns DataFrame dengan kolom minimal: ds, y, Model
    """
    # ── 0. Normalisasi kolom ───────────────────────────────────────────
    df = normalize_to_ds_y(df)
 
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
 
    # ── 2. Konversi tipe (idempoten — skip jika sudah datetime) ───────
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = df["ds"].apply(_parse_date_id)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
 
    # ── 3. Buang baris invalid ────────────────────────────────────────
    n_before  = len(df)
    df        = df.dropna(subset=["ds", "y"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
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
    extra_cols = [c for c in df.columns if c not in {"ds", "y", "Model"}]
 
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
        grp      = grp.sort_values("ds").set_index("ds")
        full_idx = pd.date_range(
            start=grp.index.min(),
            end=grp.index.max(),
            freq="MS",
        )
        grp         = grp.reindex(full_idx)
        grp["y"]    = grp["y"].fillna(0)
        grp["Model"] = model_name
 
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
