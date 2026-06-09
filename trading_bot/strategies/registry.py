"""Strategy registry used by live trading and backtesting."""

from __future__ import annotations

from .base import Strategy
from .breakout import BreakoutStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .ma_crossover import MovingAverageCrossoverStrategy
from .rsi_strategy import RsiStrategy


ALIASES = {
    "ma": "ma_crossover",
    "moving_average": "ma_crossover",
    "ma_crossover": "ma_crossover",
    "rsi": "rsi",
    "breakout": "breakout",
    "buy_hold": "buy_and_hold",
    "buy_and_hold": "buy_and_hold",
}


def available_strategy_names() -> list[str]:
    """Return canonical strategy names."""
    return ["ma_crossover", "rsi", "breakout", "buy_and_hold"]


def normalize_strategy_name(name: str) -> str:
    """Normalize friendly aliases into canonical strategy names."""
    normalized = name.strip().lower().replace("-", "_")

    if normalized not in ALIASES:
        allowed = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Choose one of: {allowed}.")

    return ALIASES[normalized]


def create_strategy(name: str, settings=None) -> Strategy:
    """Create a strategy by name.

    The optional settings object is used by the live bot so .env values can
    control strategy parameters. Backtests can omit settings and use defaults.
    """
    normalized = normalize_strategy_name(name)

    if normalized == "ma_crossover":
        fast_period = getattr(settings, "fast_ma_period", 20)
        slow_period = getattr(settings, "slow_ma_period", 50)
        return MovingAverageCrossoverStrategy(fast_period=fast_period, slow_period=slow_period)

    if normalized == "rsi":
        period = getattr(settings, "rsi_period", 14)
        buy_level = getattr(settings, "rsi_buy_level", 30.0)
        sell_level = getattr(settings, "rsi_sell_level", 70.0)
        return RsiStrategy(period=period, buy_level=buy_level, sell_level=sell_level)

    if normalized == "breakout":
        entry_period = getattr(settings, "breakout_entry_period", 20)
        exit_period = getattr(settings, "breakout_exit_period", 10)
        return BreakoutStrategy(entry_period=entry_period, exit_period=exit_period)

    if normalized == "buy_and_hold":
        return BuyAndHoldStrategy()

    raise ValueError(f"Strategy '{name}' is not implemented.")
