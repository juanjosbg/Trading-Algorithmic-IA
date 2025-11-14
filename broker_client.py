# broker_client.py
"""
Broker simulado para pruebas.
No se conecta a ningún broker real, solo mantiene un portafolio en memoria.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float


@dataclass
class SimulatedBroker:
    cash: float = 10_000.0  # capital inicial en USD (simulado)
    positions: Dict[str, Position] = field(default_factory=dict)

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calcula el valor total del portafolio: efectivo + valor de posiciones."""
        value = self.cash
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.avg_price)
            value += pos.qty * price
        return value

    def buy(self, symbol: str, qty: float, price: float):
        """Compra simulada a mercado."""
        cost = qty * price
        if cost > self.cash:
            print(f"❌ No hay suficiente efectivo para comprar {qty} de {symbol} a {price}")
            return

        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.qty + qty
            new_avg = (pos.avg_price * pos.qty + cost) / total_qty
            pos.qty = total_qty
            pos.avg_price = new_avg
        else:
            self.positions[symbol] = Position(symbol=symbol, qty=qty, avg_price=price)

        print(f"✅ COMPRA simulada: {qty} x {symbol} @ {price:.2f} | Cash restante: {self.cash:.2f}")

    def sell(self, symbol: str, qty: float, price: float):
        """Venta simulada a mercado."""
        if symbol not in self.positions:
            print(f"❌ No hay posición en {symbol} para vender.")
            return

        pos = self.positions[symbol]
        if qty > pos.qty:
            qty = pos.qty

        ingreso = qty * price
        self.cash += ingreso
        pos.qty -= qty

        if pos.qty <= 0:
            del self.positions[symbol]

        print(f"✅ VENTA simulada: {qty} x {symbol} @ {price:.2f} | Cash ahora: {self.cash:.2f}")

    def print_status(self, prices: Dict[str, float]):
        """Muestra estado general del portafolio."""
        print("\n=== ESTADO DEL BROKER SIMULADO ===")
        print(f"💵 Efectivo: {self.cash:.2f} USD")
        total = self.cash
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.avg_price)
            value = pos.qty * price
            total += value
            print(f"📈 {symbol}: {pos.qty} @ {pos.avg_price:.2f} | Precio actual: {price:.2f} | Valor: {value:.2f}")
        print(f"💰 Valor total del portafolio: {total:.2f} USD")
        print("==================================\n")
