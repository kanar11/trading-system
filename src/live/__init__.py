"""Broker adapter abstraction (paper today, IB / Alpaca / Binance tomorrow)."""

from src.live.broker import Broker, BrokerFill, PaperBroker

__all__ = ["Broker", "PaperBroker", "BrokerFill"]
