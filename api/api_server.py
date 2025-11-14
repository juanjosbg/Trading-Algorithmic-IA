from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from feature_engineering import add_features
from ml_model import load_model

app = FastAPI()

class Recommendation(BaseModel):
    symbol: str
    source: str
    price: float
    action: str
    reason: str
    ml_prob_up: float
    suggested_qty: int


@app.get("/api/recommendations")
def get_recommendations(capital: float = 10000):
    
    symbols = [
        ("AAPL", "USA • Acciones"),
        ("MSFT", "USA • Acciones"),
        ("AMZN", "USA • Acciones"),
        ("BTC-USD", "Crypto • Bitcoin"),
        ("ETH-USD", "Crypto • Ethereum"),
        ("VOO", "ETF • S&P500"),
        ("QQQ", "ETF • Nasdaq100"),
        ("BND", "ETF • Bonos"),
        ("COIN", "USA • Exchanges"),
        ("TSLA", "USA • Autos eléctricos"),
    ]

    recos = []

    for symbol, source in symbols:
        df = yf.download(symbol, period="6mo", interval="1d")
        df = add_features(df)
        df.dropna(inplace=True)

        try:
            model, features = load_model("AAPL")  # usar modelo base
            prob_up = float(model.predict_proba(df[features].iloc[-1:].values)[0][1])
            price = float(df["Close"].iloc[-1])

            action = (
                "BUY" if prob_up > 0.55 else
                "SELL" if prob_up < 0.45 else
                "HOLD"
            )

            qty = int((capital * 0.1) / price)  # 10% por activo

            recos.append(Recommendation(
                symbol=symbol,
                source=source,
                price=price,
                action=action,
                reason=f"Prob. IA {prob_up*100:.2f}%",
                ml_prob_up=prob_up,
                suggested_qty=max(qty, 0)
            ))

        except Exception as e:
            print("Error con", symbol, e)

    return {
        "capital": capital,
        "signals": recos
    }
