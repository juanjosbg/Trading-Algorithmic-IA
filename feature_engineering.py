# feature_engineering.py
"""
Feature Engineering para el bot cuantitativo:
Incluye indicadores técnicos: RSI, MACD, EMA, Bollinger Bands, ATR, Momentum, OBV, ROC
"""

import pandas as pd
import numpy as np


# ================================
# CÁLCULO DE INDICADORES TÉCNICOS
# ================================

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_ema(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df


def add_bollinger(df, window=20):
    df["BB_middle"] = df["Close"].rolling(window).mean()
    df["BB_std"] = df["Close"].rolling(window).std()
    df["BB_upper"] = df["BB_middle"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_middle"] - 2 * df["BB_std"]
    return df


def add_atr(df, period=14):
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift())
    df["L-C"] = abs(df["Low"] - df["Close"].shift())
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(period).mean()
    return df


def add_momentum(df, period=10):
    df["Momentum"] = df["Close"] - df["Close"].shift(period)
    return df


def add_roc(df, period=10):
    df["ROC"] = df["Close"].pct_change(period)
    return df


def add_obv(df):
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    return df


def add_basic_features(df):
    """Agrega indicadores técnicos y features numéricas básicas."""

    df = df.copy()

    # Cambios de precio
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_5"] = df["Close"].pct_change().rolling(5).std()
    df["lag_return_1"] = df["return_1d"].shift(1)

    # Indicadores técnicos avanzados
    df = add_rsi(df)
    df = add_macd(df)
    df = add_ema(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_momentum(df)
    df = add_roc(df)
    df = add_obv(df)

    df = df.dropna()
    return df


def add_target_direction(df):
    """Target binario: 1 si mañana sube, 0 si baja."""
    df = df.copy()
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df
