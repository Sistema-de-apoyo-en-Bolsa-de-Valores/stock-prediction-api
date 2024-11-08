# app\domain\model\prediction.py

from pydantic import BaseModel
from datetime import date

class Prediction(BaseModel):
    dates: list[str]
    actual_prices: list[float]
    training_predictions: list[float]
    test_predictions: list[float]
    future_predictions: list[float]
    verdict: str