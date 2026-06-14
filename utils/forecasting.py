import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import ElasticNet

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Bidirectional
)

from tensorflow.keras.callbacks import EarlyStopping


def create_features(df):

    df = df.copy()

    df['ds'] = pd.to_datetime(df['ds'])

    df = df.sort_values('ds')

    df['month'] = df['ds'].dt.month

    df['year'] = df['ds'].dt.year

    df['lag_1'] = df['y'].shift(1)

    df['lag_2'] = df['y'].shift(2)

    df['lag_3'] = df['y'].shift(3)

    df['rolling_mean_3'] = (
        df['y']
        .rolling(3)
        .mean()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    return df


def get_model(method):

    if method == 'XGBoost':

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

    elif method == 'LightGBM':

        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

    elif method == 'Random Forest':

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

    elif method == 'CatBoost':

        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            verbose=0
        )

    elif method == 'Extra Trees':

        model = ExtraTreesRegressor(
            n_estimators=200,
            random_state=42
        )

    elif method == 'Gradient Boosting':

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

    elif method == 'ElasticNet':

        model = ElasticNet()

    else:

        model = RandomForestRegressor(
            random_state=42
        )

    return model


def prophet_forecast(item_df, periods=36):

    prophet_df = item_df[['ds', 'y']].copy()

    prophet_df['ds'] = pd.to_datetime(
        prophet_df['ds']
    )

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

    result = forecast.tail(periods)[[
        'ds',
        'yhat'
    ]]

    result.columns = [
        'Forecast Date',
        'Forecast'
    ]

    result['Forecast'] = (
    np.clip(
        result['Forecast'],
        0,
        None
    )
    .round()
    .astype(int)
    )    

    result['Forecast Date'] = pd.to_datetime(
        result['Forecast Date']
    )

    return result


def create_sequence(data, seq_length):

    X = []

    y = []

    for i in range(len(data) - seq_length):

        X.append(
            data[i:i + seq_length]
        )

        y.append(
            data[i + seq_length]
        )

    return np.array(X), np.array(y)


def bilstm_forecast(item_df, periods=36):

    values = item_df['y'].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(values)

    seq_length = 12

    if len(scaled) <= seq_length:

        return pd.DataFrame({
            'Forecast Date': [],
            'Forecast': []
        })

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
        epochs=50,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop]
    )

    current_batch = scaled[-seq_length:]

    current_batch = current_batch.reshape(
        (1, seq_length, 1)
    )

    predictions = []

    for i in range(periods):

        pred = model.predict(
            current_batch,
            verbose=0
        )[0]

        predictions.append(pred)

        current_batch = np.append(
            current_batch[:, 1:, :],
            [[pred]],
            axis=1
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    predictions = predictions.flatten()

    predictions = (
    np.clip(
        predictions,
        0,
        None
    )
    .round()
    .astype(int)
    )

    future_dates = pd.date_range(
        start=pd.to_datetime(
            item_df['ds'].max()
        ) + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )

    forecast_df = pd.DataFrame({

        'Forecast Date': future_dates,

        'Forecast': predictions

    })

    forecast_df['Forecast Date'] = pd.to_datetime(
        forecast_df['Forecast Date']
    )

    return forecast_df


def forecast_item(item_df, method, periods=36):

    item_df = item_df.copy()

    item_df['ds'] = pd.to_datetime(
        item_df['ds']
    )

    item_df = item_df.sort_values('ds')

    # Prophet
    if method == 'Prophet':

        return prophet_forecast(
            item_df,
            periods
        )

    # BiLSTM
    if method == 'BiLSTM':

        return bilstm_forecast(
            item_df,
            periods
        )

    # ML Features
    item_df = create_features(item_df)

    # Validation
    if item_df.empty or len(item_df) < 5:

        return pd.DataFrame({
            'Forecast Date': [],
            'Forecast': []
        })

    features = [

        'month',

        'year',

        'lag_1',

        'lag_2',

        'lag_3',

        'rolling_mean_3'

    ]

    X = item_df[features]

    y = item_df['y']

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    y = y.fillna(0)

    model = get_model(method)

    model.fit(X, y)

    future_predictions = []

    temp_df = item_df.copy()

    last_date = pd.to_datetime(
        temp_df['ds'].max()
    )

    for i in range(periods):

        future_date = (
            last_date
            + pd.DateOffset(months=i+1)
        )

        lag_1 = temp_df['y'].iloc[-1]

        lag_2 = temp_df['y'].iloc[-2]

        lag_3 = temp_df['y'].iloc[-3]

        rolling_mean_3 = (
            temp_df['y']
            .tail(3)
            .mean()
        )

        future_X = pd.DataFrame({

            'month': [future_date.month],

            'year': [future_date.year],

            'lag_1': [lag_1],

            'lag_2': [lag_2],

            'lag_3': [lag_3],

            'rolling_mean_3': [rolling_mean_3]

        })

        future_X = future_X.fillna(0)

        pred = model.predict(
            future_X
        )[0]

        pred = max(pred, 0)

        future_predictions.append({

            'Forecast Date': future_date,

            'Forecast': round(pred)

        })

        new_row = pd.DataFrame({

            'ds': [future_date],

            'y': [pred]

        })

        temp_df = pd.concat(
            [temp_df, new_row],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(
        future_predictions
    )

    forecast_df['Forecast Date'] = pd.to_datetime(
        forecast_df['Forecast Date']
    )

    return forecast_df

def calculate_mape(actual, forecast):

    actual = np.array(actual)
    forecast = np.array(forecast)

    mask = actual != 0

    if mask.sum() == 0:
        return None

    mape = np.mean(
        np.abs(
            (actual[mask] - forecast[mask])
            / actual[mask]
        )
    ) * 100

    return round(mape, 2)


def evaluate_model(
    item_df,
    method,
    test_period=12
):

    item_df = (
        item_df
        .sort_values("ds")
        .copy()
    )

    if len(item_df) <= test_period + 5:
        return None


    train = item_df.iloc[:-test_period]

    test = item_df.iloc[-test_period:]


    prediction = forecast_item(
        train,
        method,
        periods=test_period
    )


    if prediction.empty:
        return None


    actual = test["y"].values

    forecast = prediction["Forecast"].values


    return calculate_mape(
        actual,
        forecast
    )
