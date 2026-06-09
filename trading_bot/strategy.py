"""Backward-compatible strategy imports.

The bot originally imported TradeAction, StrategyDecision, and
calculate_moving_average_signal from this file. New strategy code lives in the
strategies/ package, but these names remain here so older imports keep working.
"""

from __future__ import annotations

from logging import Logger

import pandas as pd

from config import Settings
from strategies.base import StrategyDecision, TradeAction
from strategies.ma_crossover import MovingAverageCrossoverStrategy


def calculate_moving_average_signal(
    price_data: pd.DataFrame,
    owns_position: bool,
    settings: Settings,
    logger: Logger,
) -> StrategyDecision:
    """Calculate the configured moving-average crossover decision."""
    strategy = MovingAverageCrossoverStrategy(
        fast_period=settings.fast_ma_period,
        slow_period=settings.slow_ma_period,
    )
    return strategy.make_decision(
        price_data=price_data,
        owns_position=owns_position,
        logger=logger,
    )
