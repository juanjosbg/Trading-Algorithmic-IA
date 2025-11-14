# api/schemas.py
from pydantic import BaseModel
from typing import List


class Signal(BaseModel):
    symbol: str
    price: float
    ml_prob_up: float
    ml_signal: str
    sma_signal: str
    action: str
    suggested_qty: int


class RecommendationsResponse(BaseModel):
    capital: float
    currency: str
    signals: List[Signal]
