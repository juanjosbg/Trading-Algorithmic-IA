import numpy as np

def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Máximo drawdown (en porcentaje negativo).
    equity_curve: array o Serie con el valor del portafolio en el tiempo.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)

    if equity_curve.size == 0:
        return 0.0

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak  # valores negativos
    return float(dd.min())


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.01) -> float:
    """
    Sharpe ratio anualizado.
    returns: array de retornos diarios de la estrategia.
    risk_free: tasa libre de riesgo anual (por defecto 1%).
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0 or returns.std() == 0:
        return 0.0

    # Convertimos risk-free anual a diario (aprox 252 días hábiles)
    excess_returns = returns - risk_free / 252
    return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())
