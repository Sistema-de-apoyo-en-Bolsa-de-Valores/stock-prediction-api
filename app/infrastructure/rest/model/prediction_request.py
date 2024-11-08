# app\infrastructure\rest\model\prediction_request.py

from pydantic import BaseModel, Field
from datetime import date
from typing import List

class PredictionRequest(BaseModel):
    ticker: str = Field(..., title="Símbolo de Acción", description="El símbolo o ticker de la acción (por ejemplo, AAPL para Apple Inc.)")
    training_period_days: int = Field(..., gt=0, title="Días de Período de Entrenamiento", description="Número de días a considerar para el entrenamiento del modelo")
    future_days: int = Field(..., gt=0, title="Días de Predicción Futura", description="Número de días a predecir en el futuro")