# api/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import RecommendationsResponse, Signal
from .trading_service import get_recommendations, ASSETS

app = FastAPI(title="Trading-Algorithmic-IA API")

# CORS para que tu frontend en React pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/assets")
def list_assets():
    return {"assets": ASSETS}


@app.get("/api/recommendations", response_model=RecommendationsResponse)
def recommendations(capital: float = Query(..., description="Capital disponible"), currency: str = "USD"):
    signals_raw = get_recommendations(capital)
    signals = [Signal(**s) for s in signals_raw]
    return RecommendationsResponse(
        capital=capital,
        currency=currency,
        signals=signals,
    )
