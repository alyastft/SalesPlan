"""
utils/security.py
=================
Modul keamanan untuk Sales Forecasting Dashboard.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SECURITY] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dashboard.security")


# 1. SESSION ISOLATION

# FIX: Perluas whitelist dengan semua keys yang dipakai app.py
_ALLOWED_STATE_KEYS: set[str] = {
    "session_id",
    "session_created_at",
    "last_activity",
    # Data forecast
    "final_forecast",
    "history_df",
    "classify_df",
    "col_date",
    "col_sales",
    "forecast_ready",
    "selected_models",
    "periods",
}

# FIX: Perluas prefix agar semua key widget Streamlit tidak terhapus
_ALLOWED_STATE_PREFIXES: tuple[str, ...] = (
    "forecast_",
    "model_",
    "mape_",
    "eda_",
    "fc_",
    "btn_",
    "FormSubmitter:",
    "$$WIDGET_ID",
    "_stcore_",
    "uploader_",
    "file_uploader",
    "selectbox",
    "slider",
    "button",
    "radio",
    "checkbox",
    "text_input",
    "number_input",
    "multiselect",
)

SESSION_TIMEOUT_SECONDS: int = int(os.getenv("SESSION_TIMEOUT_SECONDS", 3600))


def init_session() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"]         = str(uuid.uuid4())
        st.session_state["session_created_at"] = time.time()
        log.info("Sesi baru: %s", st.session_state["session_id"])

    _touch_session()
    return st.session_state["session_id"]


def _touch_session() -> None:
    st.session_state["last_activity"] = time.time()


def check_session_expiry() -> bool:
    last    = st.session_state.get("last_activity", time.time())
    elapsed = time.time() - last

    if elapsed > SESSION_TIMEOUT_SECONDS:
        sid = st.session_state.get("session_id", "unknown")
        log.warning("Sesi kedaluwarsa %.0f detik: %s", elapsed, sid)
        purge_session()
        return True

    return False


def purge_session() -> None:
    sid = st.session_state.get("session_id", "unknown")
    keys_to_delete = [k for k in list(st.session_state.keys()) if k != "session_id"]
    for k in keys_to_delete:
        del st.session_state[k]
    log.info("Session dibersihkan: %s", sid)


def enforce_state_whitelist() -> None:
    """
    Hapus kunci session state yang mencurigakan.
    FIX: Whitelist diperluas agar data forecast & widget tidak terhapus.
    """
    rogue_keys = [
        k for k in list(st.session_state.keys())
        if k not in _ALLOWED_STATE_KEYS
        and not any(k.startswith(prefix) for prefix in _ALLOWED_STATE_PREFIXES)
        and not _looks_like_internal_key(k)
    ]

    if rogue_keys:
        log.warning("Key state tidak diizinkan dihapus: %s", rogue_keys)
        for k in rogue_keys:
            try:
                del st.session_state[k]
            except Exception:
                pass


def _looks_like_internal_key(key: str) -> bool:
    if any(c in key for c in ("$", "\x00", "widget")):
        return True
    if len(key) > 80:
        return True
    return False


# 2. UPLOAD VALIDATION

MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_MB", 20)) * 1024 * 1024
ALLOWED_EXTENSIONS: set[str] = {".csv"}
MAX_ROWS: int  = int(os.getenv("MAX_UPLOAD_ROWS", 500_000))
MAX_COLS: int  = int(os.getenv("MAX_UPLOAD_COLS", 200))


def validate_upload(uploaded_file) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "Tidak ada file yang diunggah."

    filename: str = uploaded_file.name
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        log.warning("Ekstensi tidak diizinkan: %s", ext)
        return False, f"Tipe file '{ext}' tidak diizinkan. Gunakan file CSV (.csv)."

    size = uploaded_file.size
    if size > MAX_FILE_SIZE_BYTES:
        mb    = size / (1024 * 1024)
        limit = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return False, f"Ukuran file ({mb:.1f} MB) melebihi batas {limit:.0f} MB."

    if _is_path_traversal(filename):
        log.error("Path traversal: %s", filename)
        return False, "Nama file tidak valid."

    log.info("Upload OK: %s (%.1f KB)", filename, size / 1024)
    return True, ""


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "File kosong atau tidak ada data yang dapat dibaca."

    if len(df) > MAX_ROWS:
        return False, f"Data terlalu besar ({len(df):,} baris). Batas: {MAX_ROWS:,}."

    if len(df.columns) > MAX_COLS:
        return False, f"Terlalu banyak kolom ({len(df.columns)}). Batas: {MAX_COLS}."

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100).astype(str)
        if sample.str.match(r"^[=+\-@|]").any():
            log.warning("Potensi formula injection di kolom: %s", col)
            df[col] = df[col].apply(_sanitize_cell)

    return True, ""


def _sanitize_cell(val: Any) -> Any:
    if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@", "|"):
        return "'" + val
    return val


def _is_path_traversal(filename: str) -> bool:
    dangerous = ["..", "/", "\\", "\x00", "%2e", "%2f", "%5c"]
    lower = filename.lower()
    return any(d in lower for d in dangerous)


# 3. DATA SANITIZATION (PII)

_PII_COLUMN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bemail\b",         re.I),
    re.compile(r"\bphone\b",         re.I),
    re.compile(r"\btelp(on)?\b",     re.I),
    re.compile(r"\bno_?hp\b",        re.I),
    re.compile(r"\baddress\b",       re.I),
    re.compile(r"\balamat\b",        re.I),
    re.compile(r"\bnik\b",           re.I),
    re.compile(r"\bnpwp\b",          re.I),
    re.compile(r"\bpassword\b",      re.I),
    re.compile(r"\bsandi\b",         re.I),
    re.compile(r"\bcredit.?card\b",  re.I),
    re.compile(r"\bkartu.?kredit\b", re.I),
    re.compile(r"\bssn\b",           re.I),
]


def drop_pii_columns(df: pd.DataFrame, *, mask: bool = False) -> pd.DataFrame:
    df = df.copy()
    pii_cols = [
        col for col in df.columns
        if any(pat.search(col) for pat in _PII_COLUMN_PATTERNS)
    ]
    if pii_cols:
        if mask:
            for col in pii_cols:
                df[col] = "***"
            log.info("Kolom PII di-mask: %s", pii_cols)
        else:
            df = df.drop(columns=pii_cols)
            log.info("Kolom PII dihapus: %s", pii_cols)
    return df


def hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


# 4. AUDIT LOG

_AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit.log")

_audit_handler = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
_audit_handler.setLevel(logging.INFO)
_audit_handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
)

_audit_logger = logging.getLogger("dashboard.audit")
if not _audit_logger.handlers:
    _audit_logger.addHandler(_audit_handler)
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False


def audit_log(action: str, detail: str = "") -> None:
    sid = st.session_state.get("session_id", "unknown")
    _audit_logger.info("%s\t%s\t%s", sid, action, detail)


def secure_download_button(
    label:       str,
    data:        bytes | BytesIO,
    file_name:   str,
    mime:        str,
    action_name: str = "DOWNLOAD",
    **kwargs,
) -> bool:
    clicked = st.download_button(
        label=label, data=data, file_name=file_name, mime=mime, **kwargs,
    )
    if clicked:
        audit_log(action_name, f"file={file_name}")
    return clicked


# 5. SECURE DISPLAY

_MAX_PREVIEW_ROWS: int = 500


def safe_dataframe_display(
    df:       pd.DataFrame,
    *,
    max_rows: int  = _MAX_PREVIEW_ROWS,
    mask_pii: bool = True,
    **kwargs,
) -> None:
    display_df = df.head(max_rows)
    if mask_pii:
        display_df = drop_pii_columns(display_df, mask=True)
    if len(df) > max_rows:
        st.caption(
            f"⚠️ Menampilkan {max_rows:,} dari {len(df):,} baris "
            "untuk menjaga performa dan keamanan."
        )
    st.dataframe(display_df, **kwargs)
    audit_log("VIEW_DATA", f"rows_shown={min(max_rows, len(df))}")


# 6. SECURITY SIDEBAR

def render_security_sidebar() -> None:
    with st.sidebar:
        st.divider()
        sid       = st.session_state.get("session_id", "-")
        created   = st.session_state.get("session_created_at", time.time())
        elapsed   = int(time.time() - created)
        remaining = max(0, SESSION_TIMEOUT_SECONDS - elapsed)

        st.caption(
            f"🔐 **Sesi:** `{sid[:8]}…`  \n"
            f"⏱️ **Kedaluwarsa dalam:** {remaining // 60} menit"
        )

        if st.button("🚪 Logout / Hapus Data", use_container_width=True):
            audit_log("LOGOUT")
            purge_session()
            st.success("Data sesi berhasil dihapus.")
            st.rerun()
