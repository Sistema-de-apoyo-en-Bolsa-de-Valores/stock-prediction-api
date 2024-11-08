# app\infrastructure\rest\controllers\stock_controller.py

from fastapi import APIRouter, HTTPException, Depends
from app.application.service.stock_prediction_service import StockPredictionService
from app.infrastructure.rest.model.prediction_request import PredictionRequest
from app.infrastructure.configuration.dependencies import get_stock_prediction_service
from fastapi.responses import JSONResponse
from app.domain.exceptions.ticker_not_found_exception import TickerNotFoundException
from datetime import date

router = APIRouter()

@router.post(
    "/predict",
    summary="Endpoint para predecir el precio de las acciones.",
    description=(
        "Se predicen los precios futuros de las acciones mediante un modelo LSTM entrenado con datos históricos. Este endpoint obtiene datos históricos de un símbolo bursátil específico, entrena el modelo y proporciona predicciones tanto para el período de prueba como para fechas futuras."
    )
)
async def predict_stock(request: PredictionRequest, stock_prediction_service: StockPredictionService = Depends(get_stock_prediction_service)):
    try:
        prediction = stock_prediction_service.predict(
            ticker=request.ticker,
            training_period_days=request.training_period_days,
            future_days=request.future_days
        )

        return JSONResponse(content=prediction.dict(), status_code=200)
    except TickerNotFoundException as e:
        return JSONResponse(content={"timestamp": date.today().isoformat(), "messages": [str(e)]}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"timestamp": date.today().isoformat(), "messages": [str(e)]}, status_code=500)