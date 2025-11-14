# ml_model.py
"""
Entrenamiento de un modelo ML simple para predecir si el precio subirá o bajará.
"""

import os
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

from feature_engineering import add_basic_features, add_target_direction

SYMBOL = "AAPL"
PERIOD = "5y"
INTERVAL = "1d"
MODEL_PATH = "models/random_forest_aapl.pkl"


def load_data():
    print(f"Descargando datos de {SYMBOL} ({PERIOD}, {INTERVAL})...")
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    df = add_basic_features(df)
    df = add_target_direction(df)
    return df


def train_model():
    df = load_data()

    feature_cols = []
    for c in df.columns:
        if isinstance(c, str):
            if c.startswith("return_") or c.startswith("volatility_") or c.startswith("lag_return_"):
                feature_cols.append(c)

    if not feature_cols:
        raise ValueError("❌ No se encontraron columnas de features. Revisa feature_engineering.py")

    X = df[feature_cols]
    y = df["target_up"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Entrenando modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)

    joblib.dump((model, feature_cols), MODEL_PATH)
    print(f"\n✅ Modelo guardado correctamente en: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
