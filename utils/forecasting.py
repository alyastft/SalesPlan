import numpy as np
import pandas as pd

from prophet import Prophet

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Bidirectional
)

from tensorflow.keras.callbacks import EarlyStopping


# FEATURE ENGINEERING

def create_features(df):

    df = df.copy()

    df['month'] = df['ds'].dt.month

    df['year'] = df['ds'].dt.year

    df['quarter'] = df['ds'].dt.quarter

    df['lag_1'] = df['y'].shift(1)

    df['lag_2'] = df['y'].shift(2)

    df['lag_3'] = df['y'].shift(3)

    df['rolling_mean_3'] = (
        df['y']
        .rolling(3)
        .mean()
    )

    df['rolling_std_3'] = (
        df['y']
        .rolling(3)
        .std()
    )

    df = df.dropna()

    return df


# PREPARE ML DATA

def prepare_ml_data(df):

    df_feat = create_features(df)

    feature_cols = [
        'month',
        'year',
        'quarter',
        'lag_1',
        'lag_2',
        'lag_3',
        'rolling_mean_3',
        'rolling_std_3'
    ]

    X = df_feat[feature_cols]

    y = df_feat['y']

    return X, y, df_feat


# FUTURE FEATURE GENERATION

def generate_future_features(df, periods=12):

    temp_df = df.copy()

    predictions = []

    future_dates = pd.date_range(
        start=temp_df['ds'].max() + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )

    return future_dates


# RANDOM FOREST

def random_forest_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# EXTRA TREES

def extra_trees_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# GRADIENT BOOSTING

def gradient_boosting_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = GradientBoostingRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# ADABOOST

def adaboost_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = AdaBoostRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# XGBOOST

def xgboost_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# LIGHTGBM

def lightgbm_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# CATBOOST

def catboost_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        verbose=0
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# KNN

def knn_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = KNeighborsRegressor(
        n_neighbors=5
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# SVR

def svr_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = SVR()

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# MLP

def mlp_forecast(train_df, periods=12):

    X, y, df_feat = prepare_ml_data(train_df)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        random_state=42
    )

    model.fit(X, y)

    return recursive_forecast(
        model,
        train_df,
        periods
    )


# PROPHET

def prophet_forecast(train_df, periods=12):

    prophet_df = train_df[['ds', 'y']].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=periods,
        freq='MS'
    )

    forecast = model.predict(future)

    pred = forecast.tail(periods)['yhat'].values

    pred = np.clip(pred, 0, None)

    return pred


# BILSTM

def create_sequence(data, seq_length):

    X = []

    y = []

    for i in range(len(data) - seq_length):

        X.append(data[i:i + seq_length])

        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


def bilstm_forecast(train_df, periods=12):

    values = train_df['y'].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(values)

    seq_length = 12

    X, y = create_sequence(
        scaled,
        seq_length
    )

    X = X.reshape(
        (X.shape[0], X.shape[1], 1)
    )

    model = Sequential()

    model.add(
        Bidirectional(
            LSTM(
                64,
                activation='relu'
            ),
            input_shape=(seq_length, 1)
        )
    )

    model.add(Dense(32))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=100,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop]
    )

    forecast_input = scaled[-seq_length:]

    predictions = []

    current_batch = forecast_input.reshape(
        (1, seq_length, 1)
    )

    for _ in range(periods):

        current_pred = model.predict(
            current_batch,
            verbose=0
        )[0]

        predictions.append(current_pred)

        current_batch = np.append(
            current_batch[:, 1:, :],
            [[current_pred]],
            axis=1
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    predictions = predictions.flatten()

    predictions = np.clip(
        predictions,
        0,
        None
    )

    return predictions


# RECURSIVE FORECAST

def recursive_forecast(model, train_df, periods=12):

    temp_df = train_df.copy()

    predictions = []

    for _ in range(periods):

        feat_df = create_features(temp_df)

        latest = feat_df.iloc[-1:]

        X_latest = latest[[
            'month',
            'year',
            'quarter',
            'lag_1',
            'lag_2',
            'lag_3',
            'rolling_mean_3',
            'rolling_std_3'
        ]]

        pred = model.predict(X_latest)[0]

        pred = max(pred, 0)

        predictions.append(pred)

        next_date = (
            temp_df['ds'].max()
            + pd.DateOffset(months=1)
        )

        new_row = pd.DataFrame({
            'ds': [next_date],
            'y': [pred]
        })

        temp_df = pd.concat(
            [temp_df, new_row],
            ignore_index=True
        )

    return np.array(predictions)
