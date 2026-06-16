"""
utils/helper.py
===============
Konstanta dan fungsi helper untuk Sales Forecasting Dashboard.
"""

# ── Model yang direkomendasikan per kategori produk ──────────────────────────
RECOMMENDED_MODELS: dict[str, list[str]] = {
    "Stable": [
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "ElasticNet",
        "Prophet",
        "BiLSTM",
    ],
    "Declining": [
        "ElasticNet",
        "XGBoost",
        "LightGBM",
        "Prophet",
        "Gradient Boosting",
        "Random Forest",
        "CatBoost",
        "Extra Trees",
        "BiLSTM",
    ],
    "Volatile": [
        "Random Forest",
        "Extra Trees",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Gradient Boosting",
        "ElasticNet",
        "Prophet",
        "BiLSTM",
    ],
    "Intermittent": [
        "XGBoost",
        "Random Forest",
        "LightGBM",
        "CatBoost",
        "Extra Trees",
        "Gradient Boosting",
        "ElasticNet",
        "Prophet",
        "BiLSTM",
    ],
    "Discontinued": [
        "Random Forest",
        "ElasticNet",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Extra Trees",
        "Gradient Boosting",
        "Prophet",
        "BiLSTM",
    ],
}

# ── Pemetaan label warna untuk tampilan ──────────────────────────────────────
CATEGORY_COLORS: dict[str, str] = {
    "Stable":       "#22c55e",   # hijau
    "Declining":    "#ef4444",   # merah
    "Volatile":     "#f59e0b",   # kuning
    "Intermittent": "#8b5cf6",   # ungu
    "Discontinued": "#6b7280",   # abu-abu
}

CATEGORY_ICONS: dict[str, str] = {
    "Stable":       "📈",
    "Declining":    "📉",
    "Volatile":     "〰️",
    "Intermittent": "⚡",
    "Discontinued": "🚫",
}
