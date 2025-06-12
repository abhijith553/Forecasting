# Standard library
import json
import os

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Time series analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import History


def xgboost():
    df = pd.read_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\datasets\\retail_cleaned_for_lstm.csv")

    features = df.drop(columns=["Weekly_Sales", "Date"])  # Drop target + non-numeric date
    target = df["Weekly_Sales"]

    X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
        features, target, df["Date"], test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = round(mean_absolute_error(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)

    results_df = pd.DataFrame({
        "Date": date_test,
        "Actual": y_test.values,
        "XGBoost_Prediction": y_pred
    })
    results_df.sort_values("Date", inplace=True)
    results_df.to_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\xgboost_results.csv", index=False)

    metrics = {"MAE": mae, "RMSE": rmse}
    with open("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\xgboost_metrics.json", "w") as f:
        json.dump(metrics, f)

    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    importance_df.to_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\xgboost_feature_importance.csv", index=False)

    print("XGBoost training complete. Results and plots saved.")

def train_linear_regression():

    df = pd.read_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\datasets\\retail_cleaned_for_lstm.csv")

    features = df.drop(columns=["Weekly_Sales", "Date"])
    target = df["Weekly_Sales"]

    X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
        features, target, df["Date"], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    mae = round(mean_absolute_error(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    with open("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\linear_regression_metrics.json", "w") as f:
        json.dump(metrics, f)

    results_df = pd.DataFrame({
        "Date": date_test,
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    results_df.sort_values("Date", inplace=True)
    results_df.to_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\linear_regression_results.csv", index=False)

    coeff_df = pd.DataFrame({
        "Feature": features.columns,
        "Coefficient": model.coef_
    })
    coeff_df.to_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\linear_regression_coefficients.csv", index=False)

    print("Linear Regression model training complete. Results and plots saved.")
def lstm_model():
    df = pd.read_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\datasets\\retail_cleaned_for_lstm.csv")
    data = df[['Weekly_Sales']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_len = 10
    X, y = create_sequences(data_scaled, seq_len)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_len, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = round(mean_absolute_error(y_test_inv, y_pred_inv), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)), 3)

    dates = df['Date'].iloc[-len(y_test):].reset_index(drop=True)
    pred_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_test_inv.flatten(),
        'LSTM_Prediction': y_pred_inv.flatten()
    })
    pred_df.to_csv("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\lstm_results.csv", index=False)

    with open("C:\\Users\\Abhijith\\Documents\\programs\\personal\\model_outputs\\lstm_metrics.json", "w") as f:
        json.dump({'MAE': mae, 'RMSE': rmse}, f)


    print("LSTM model trained and results saved.")


def arima():
    df = pd.read_csv("/kaggle/input/forecast-datasets/retail_cleaned_for_lstm.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    result = adfuller(df['Weekly_Sales'])
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')

    stepwise_model = auto_arima(df['Weekly_Sales'],
                                seasonal=False,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True)
    print(stepwise_model.summary())

    train_size = int(0.8 * len(df))
    train = df['Weekly_Sales'][:train_size]
    test = df['Weekly_Sales'][train_size:]

    order = stepwise_model.order
    model = ARIMA(train, order=order)
    model_fit = model.fit()


    forecast_horizon = 4
    forecast_result = model_fit.forecast(steps=len(test)+forecast_horizon)
    forecast_values = forecast_result[:len(test)]
    future_values = forecast_result[-forecast_horizon:]

    rmse = round(np.sqrt(mean_squared_error(test, forecast_values)), 3)
    mae = round(mean_absolute_error(test, forecast_values), 3)
    metrics = {"MAE": mae, "RMSE": rmse}
    with open("/kaggle/working/arima_metrics.json", "w") as f:
        json.dump(metrics, f)

    results_df = pd.DataFrame({
        'Date': test.index,
        'Actual': test.values,
        'ARIMA_Prediction': forecast_values
    })
    results_df.to_csv("/kaggle/working/arima_results.csv", index=False)

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=forecast_horizon, freq='W')
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_values
    })
    forecast_df.to_csv("/kaggle/working/arima_forecast.csv", index=False)

    print("ARIMA model trained and results saved.")

xgboost()
train_linear_regression()
lstm_model()
arima()