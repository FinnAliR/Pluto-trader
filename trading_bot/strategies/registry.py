"""Strategy registry used by live trading and backtesting."""

from __future__ import annotations

from .base import Strategy
from .breakout import BreakoutStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .core_tactical_ma import CoreTacticalMovingAverageStrategy
from .ma_crossover import MovingAverageCrossoverStrategy
from .rebound_reentry_ma import ReboundReentryMovingAverageStrategy
from .rsi_strategy import RsiStrategy


ALIASES = {
    "ma": "ma_crossover",
    "moving_average": "ma_crossover",
    "ma_crossover": "ma_crossover",
    "rsi": "rsi",
    "breakout": "breakout",
    "buy_hold": "buy_and_hold",
    "buy_and_hold": "buy_and_hold",
    "core": "core_tactical_ma",
    "core_tactical": "core_tactical_ma",
    "core_tactical_ma": "core_tactical_ma",
    "rebound": "rebound_reentry_ma",
    "rebound_ma": "rebound_reentry_ma",
    "rebound_reentry": "rebound_reentry_ma",
    "rebound_reentry_ma": "rebound_reentry_ma",
}


def available_strategy_names() -> list[str]:
    """Return canonical strategy names."""
    return [
        "ma_crossover",
        "core_tactical_ma",
        "rebound_reentry_ma",
        "rsi",
        "breakout",
        "buy_and_hold",
    ]


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

    if normalized == "core_tactical_ma":
        fast_period = getattr(settings, "fast_ma_period", 20)
        slow_period = getattr(settings, "slow_ma_period", 50)
        core_fraction = getattr(settings, "core_tactical_core_fraction", 0.40)
        return CoreTacticalMovingAverageStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            core_fraction=core_fraction,
        )

    if normalized == "rebound_reentry_ma":
        fast_period = getattr(settings, "fast_ma_period", 20)
        slow_period = getattr(settings, "slow_ma_period", 50)
        rebound_ma_period = getattr(settings, "rebound_ma_period", 10)
        rebound_return_days = getattr(settings, "rebound_return_days", 3)
        rebound_min_return = getattr(settings, "rebound_min_return", 0.0)
        return ReboundReentryMovingAverageStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            rebound_ma_period=rebound_ma_period,
            rebound_return_days=rebound_return_days,
            rebound_min_return=rebound_min_return,
        )

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
