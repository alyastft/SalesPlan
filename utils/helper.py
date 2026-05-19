RECOMMENDED_MODELS = {

    'Stable': [
        'XGBoost',
        'LightGBM',
        'Random Forest',
        'CatBoost',
        'Extra Trees'
    ],

    'Declining': [
        'XGBoost',
        'LightGBM',
        'ElasticNet',
        'Random Forest',
        'Gradient Boosting'
    ],

    'Volatile': [
        'LSTM',
        'GRU',
        'TCN',
        'N-BEATS',
        'Temporal Fusion Transformer'
    ],

    'Intermittent': [
        'CatBoost',
        'LightGBM',
        'Random Forest',
        'DeepAR',
        'XGBoost'
    ],

    'Discontinued': [
        'Random Forest',
        'XGBoost',
        'CatBoost',
        'LightGBM',
        'MLP Regressor'
    ]
}
