# app\application\service\data_scaler_service.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScalerService:
    def __init__(self, min_max_scaler: MinMaxScaler):
        self.min_max_scaler = min_max_scaler

    def fit_transform(self, data):
        return self.min_max_scaler.fit_transform(np.array(data).reshape(-1, 1))

    def inverse_transform(self, data):
        return self.min_max_scaler.inverse_transform(data)