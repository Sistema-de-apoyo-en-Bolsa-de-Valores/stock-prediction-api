# app\infrastructure\configuration\dependencies.py

from fastapi import Depends
from sklearn.preprocessing import MinMaxScaler
from app.application.service.stock_prediction_service import StockPredictionService
from app.application.service.data_scaler_service import DataScalerService

def get_min_max_scaler():
    return MinMaxScaler()

def get_data_scaler_service(min_max_scaler: MinMaxScaler = Depends(get_min_max_scaler)):
    return DataScalerService(min_max_scaler)

def get_stock_prediction_service(data_scaler_service: DataScalerService = Depends(get_data_scaler_service)):
    return StockPredictionService(data_scaler_service)