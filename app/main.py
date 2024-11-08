# app\main.py

from fastapi import FastAPI, HTTPException, Request
from app.infrastructure.rest.controllers.stock_controller import router
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from datetime import date

app = FastAPI(
    title="stock-prediction-api",
    description=(
        "Esta API permite predecir el precio de las acciones a partir de data histórica de la API de Yahoo Finance."
    ),
    version="1.0.0"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(content={"timestamp": date.today().isoformat(), "messages": [error['msg'] for error in exc.errors()]}, status_code=400)

app.include_router(router, prefix="/stock", tags=["Endpoints sobre el precio de las acciones"])