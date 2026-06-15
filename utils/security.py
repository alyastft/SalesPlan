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
 
# ── Logger ──────────────────────────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SECURITY] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dashboard.security")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. SESSION ISOLATION
# ══════════════════════════════════════════════════════════════════════════════
 
# Kunci session state yang diizinkan disimpan (whitelist)
_ALLOWED_STATE_KEYS: set[str] = {
    "session_id",
    "session_created_at",
    "last_activity",
    "final_forecast",
    "history_df",
    "col_date",
    "col_sales",
}
 
# Waktu idle maksimum sebelum session dihapus (detik)
SESSION_TIMEOUT_SECONDS: int = int(
    os.getenv("SESSION_TIMEOUT_SECONDS", 3600)  # default 1 jam
)
 
 
def init_session() -> str:
    """
    Inisialisasi sesi unik per user.
    Kembalikan session_id yang sudah ada atau buat baru.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["session_created_at"] = time.time()
        log.info("Sesi baru dibuat: %s", st.session_state["session_id"])
 
    _touch_session()
    return st.session_state["session_id"]
 
 
def _touch_session() -> None:
    """Perbarui timestamp aktivitas terakhir."""
    st.session_state["last_activity"] = time.time()
 
 
def check_session_expiry() -> bool:
    """
    Periksa apakah sesi sudah kedaluwarsa.
    Jika ya, hapus semua data dan kembalikan True.
    """
    last = st.session_state.get("last_activity", time.time())
    elapsed = time.time() - last
 
    if elapsed > SESSION_TIMEOUT_SECONDS:
        sid = st.session_state.get("session_id", "unknown")
        log.warning("Sesi kedaluwarsa setelah %.0f detik: %s", elapsed, sid)
        purge_session()
        return True
 
    return False
 
 
def purge_session() -> None:
    """Hapus semua data sensitif dari session state."""
    sid = st.session_state.get("session_id", "unknown")
    keys_to_delete = [
        k for k in list(st.session_state.keys())
        if k not in {"session_id"}
    ]
    for k in keys_to_delete:
        del st.session_state[k]
    log.info("Session state dibersihkan untuk sesi: %s", sid)
 
 
def enforce_state_whitelist() -> None:
    """
    Hapus kunci session state yang tidak ada di whitelist.
    Cegah injeksi data sembarangan ke dalam state.
    """
    rogue_keys = [
        k for k in list(st.session_state.keys())
        if k not in _ALLOWED_STATE_KEYS
    ]
    if rogue_keys:
        log.warning("Kunci state tidak diizinkan ditemukan dan dihapus: %s", rogue_keys)
        for k in rogue_keys:
            del st.session_state[k]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. UPLOAD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
 
# Batas ukuran file upload (bytes)
MAX_FILE_SIZE_BYTES: int = int(
    os.getenv("MAX_FILE_SIZE_MB", 20)
) * 1024 * 1024
 
# Tipe MIME yang diizinkan
ALLOWED_MIME_TYPES: set[str] = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "application/csv",
}
 
# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS: set[str] = {".xlsx", ".xls", ".csv"}
 
# Jumlah baris maksimum yang diizinkan
MAX_ROWS: int = int(os.getenv("MAX_UPLOAD_ROWS", 500_000))
 
# Jumlah kolom maksimum
MAX_COLS: int = int(os.getenv("MAX_UPLOAD_COLS", 200))
 
 
def validate_upload(uploaded_file) -> tuple[bool, str]:
    """
    Validasi file upload sebelum diproses.
 
    Returns
    -------
    (ok: bool, pesan_error: str)
    """
    if uploaded_file is None:
        return False, "Tidak ada file yang diunggah."
 
    # Periksa ekstensi
    filename: str = uploaded_file.name
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        log.warning("Ekstensi file tidak diizinkan: %s", ext)
        return False, f"Tipe file '{ext}' tidak diizinkan. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}"
 
    # Periksa ukuran
    size = uploaded_file.size
    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        log.warning("File terlalu besar: %.1f MB (limit %.0f MB)", mb, limit_mb)
        return False, f"Ukuran file ({mb:.1f} MB) melebihi batas {limit_mb:.0f} MB."
 
    # Periksa nama file — cegah path traversal
    if _is_path_traversal(filename):
        log.error("Percobaan path traversal terdeteksi: %s", filename)
        return False, "Nama file tidak valid."
 
    log.info(
        "Validasi file OK: %s (%.1f KB)",
        filename,
        size / 1024,
    )
    return True, ""
 
 
def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validasi DataFrame hasil parsing file.
    Periksa dimensi dan keberadaan nilai berbahaya.
    """
    if df.empty:
        return False, "File kosong atau tidak ada data yang dapat dibaca."
 
    if len(df) > MAX_ROWS:
        return False, f"Data terlalu besar ({len(df):,} baris). Batas: {MAX_ROWS:,} baris."
 
    if len(df.columns) > MAX_COLS:
        return False, f"Terlalu banyak kolom ({len(df.columns)}). Batas: {MAX_COLS}."
 
    # Periksa formula injection di sel string (CSV injection)
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100).astype(str)
        if sample.str.match(r"^[=+\-@|]").any():
            log.warning("Potensi formula injection di kolom: %s", col)
            # Sanitasi otomatis — tambahkan prefix apostrof
            df[col] = df[col].apply(_sanitize_cell)
 
    return True, ""
 
 
def _sanitize_cell(val: Any) -> Any:
    """Hapus karakter pemicu formula injection di awal string."""
    if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@", "|"):
        return "'" + val
    return val
 
 
def _is_path_traversal(filename: str) -> bool:
    """Deteksi upaya path traversal di nama file."""
    dangerous = ["..", "/", "\\", "\x00", "%2e", "%2f", "%5c"]
    lower = filename.lower()
    return any(d in lower for d in dangerous)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA SANITIZATION (hapus / mask PII)
# ══════════════════════════════════════════════════════════════════════════════
 
# Pola regex untuk mendeteksi PII di nama kolom
_PII_COLUMN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bemail\b",          re.I),
    re.compile(r"\bphone\b",          re.I),
    re.compile(r"\btelp(on)?\b",      re.I),
    re.compile(r"\bno_?hp\b",         re.I),
    re.compile(r"\baddress\b",        re.I),
    re.compile(r"\balamat\b",         re.I),
    re.compile(r"\bnik\b",            re.I),
    re.compile(r"\bnpwp\b",           re.I),
    re.compile(r"\bpassword\b",       re.I),
    re.compile(r"\bsandi\b",          re.I),
    re.compile(r"\bcredit.?card\b",   re.I),
    re.compile(r"\bkartu.?kredit\b",  re.I),
    re.compile(r"\bssn\b",            re.I),
]
 
 
def drop_pii_columns(df: pd.DataFrame, *, mask: bool = False) -> pd.DataFrame:
    """
    Hapus atau mask kolom yang diduga mengandung PII.
 
    Parameters
    ----------
    df   : DataFrame input
    mask : Jika True, isi kolom PII dengan '***' alih-alih menghapusnya
    """
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
    """
    Hash satu nilai identifier (mis. nama pelanggan) dengan SHA-256.
    Digunakan jika identifier perlu dipertahankan tapi tidak boleh terbaca.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:16]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
 
_AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit.log")
 
_audit_handler = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
_audit_handler.setLevel(logging.INFO)
_audit_handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
)
 
_audit_logger = logging.getLogger("dashboard.audit")
_audit_logger.addHandler(_audit_handler)
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False
 
 
def audit_log(action: str, detail: str = "") -> None:
    """
    Catat aksi sensitif ke audit log.
    Tidak menyimpan isi data, hanya metadata.
    """
    sid = st.session_state.get("session_id", "unknown")
    _audit_logger.info("%s\t%s\t%s", sid, action, detail)
 
 
def secure_download_button(
    label: str,
    data: bytes | BytesIO,
    file_name: str,
    mime: str,
    action_name: str = "DOWNLOAD",
    **kwargs,
) -> bool:
    """
    Wrapper st.download_button yang otomatis mencatat ke audit log
    setiap kali tombol diklik.
    """
    clicked = st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        **kwargs,
    )
    if clicked:
        audit_log(action_name, f"file={file_name}")
    return clicked
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. SECURE DISPLAY (masking pratinjau)
# ══════════════════════════════════════════════════════════════════════════════
 
_MAX_PREVIEW_ROWS: int = 500
 
 
def safe_dataframe_display(
    df: pd.DataFrame,
    *,
    max_rows: int = _MAX_PREVIEW_ROWS,
    mask_pii: bool = True,
    **kwargs,
) -> None:
    """
    Tampilkan DataFrame dengan aman:
    - Batasi jumlah baris yang ditampilkan
    - Mask kolom PII
    - Audit log aksi tampil
    """
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
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6. HELPER: SECURITY BANNER DI SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
 
def render_security_sidebar() -> None:
    """Tampilkan info sesi dan tombol logout di sidebar."""
    with st.sidebar:
        st.divider()
        sid = st.session_state.get("session_id", "-")
        created = st.session_state.get("session_created_at", time.time())
        elapsed = int(time.time() - created)
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
 
