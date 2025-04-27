# backend/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas, xgboost
import uvicorn

app = FastAPI(title="Product Forecast API")

# CORS so your Vite frontend (localhost:5173) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response schemas
class ForecastRequest(BaseModel):
    stock_code: str
    horizon: int = 3

class ForecastResponse(BaseModel):
    stock_code: str
    predictions: list[float]

# Endpoint
@app.post("/api/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    df = data_loader.load_for_code(req.stock_code)
    if df is None:
        raise HTTPException(status_code=404, detail="Stock code not found")
    preds = model.forecast(df, horizon=req.horizon)
    return ForecastResponse(
        stock_code=req.stock_code,
        predictions=preds.tolist()
    )

# Uvicorn entrypoint
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True           # auto-reload on code changes
    )
