# app\application\service\stock_prediction_service.py

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from app.infrastructure.yahoo_finance.yahoo_finance import fetch_data
from app.application.service.data_scaler_service import DataScalerService
from app.domain.exceptions.ticker_not_found_exception import TickerNotFoundException
from app.domain.model.prediction import Prediction

class StockPredictionService:
    def __init__(self, data_scaler_service: DataScalerService):
        self.data_scaler_service = data_scaler_service

    def create_sequences(self, data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def create_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model

    def train_model(self, model, x_train, y_train, x_val, y_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])


    def make_prediction(self, model, x):
        return model.predict(x)

    def predict_future(self, model, last_sequence, future_days):
        predictions = []
        for _ in range(future_days):
            prediction = model.predict(last_sequence)
            predictions.append(prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = prediction
        return self.data_scaler_service.inverse_transform(np.array(predictions).reshape(-1, 1))

    def process_data(self, raw_data):
        close_values = raw_data['Close'].to_numpy()
        dates = raw_data.index.to_numpy()
        processed_data = pd.DataFrame({'Date': dates, 'Close': close_values})
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        return processed_data


    def predict(self, ticker, training_period_days, future_days):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - timedelta(days=training_period_days)
        end_date = today

        # Obtener y escalar datos
        raw_data = fetch_data(ticker, start=start_date, end=end_date)

        if raw_data.empty:
            raise TickerNotFoundException()

        processed_data = self.process_data(raw_data)

        scaled_data = self.data_scaler_service.fit_transform(processed_data["Close"])

        # División y entrenamiento de datos
        training_data_size = int(len(scaled_data) * 0.8)
        training_data, test_data = scaled_data[:training_data_size], scaled_data[training_data_size:]

        intervals = 60
        x_train, y_train = self.create_sequences(training_data, intervals)
        x_test, y_test = self.create_sequences(test_data, intervals)

        # Redimensionar datos para LSTM
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Crear y entrenar el modelo
        model = self.create_model((x_train.shape[1], 1))
        self.train_model(model, x_train, y_train, x_test, y_test)

        # Predicciones
        train_preds = self.make_prediction(model, x_train)
        test_preds = self.make_prediction(model, x_test)
        train_preds, test_preds = self.data_scaler_service.inverse_transform(train_preds), self.data_scaler_service.inverse_transform(test_preds)

        # Predicción de días futuros
        last_sequence = x_test[-1].reshape(1, 60, 1)
        future_preds = self.predict_future(model, last_sequence, future_days)

        # Veredicto de inversión
        verdict = "Invertir" if future_preds[-1][0] > raw_data["Close"].values[-1] else "No invertir"

        dates = [d.isoformat() for d in raw_data.index.date]

        return Prediction(
            dates=dates,
            actual_prices=raw_data["Close"].tolist(),
            training_predictions=train_preds.flatten().tolist(),
            test_predictions=test_preds.flatten().tolist(),
            future_predictions=future_preds.flatten().tolist(),
            verdict=verdict
        )