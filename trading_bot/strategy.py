"""Simple moving-average crossover strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from logging import Logger

import pandas as pd

from config import Settings


class TradeAction(str, Enum):
    """The only actions this bot is allowed to consider."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class StrategyDecision:
    """The strategy output passed to risk checks."""

    action: TradeAction
    reason: str
    latest_close: float | None = None
    fast_ma: float | None = None
    slow_ma: float | None = None


def calculate_moving_average_signal(
    price_data: pd.DataFrame,
    owns_position: bool,
    settings: Settings,
    logger: Logger,
) -> StrategyDecision:
    """Calculate the moving-average crossover decision.

    Default settings use 20-day and 50-day simple moving averages. If TIMEFRAME
    is changed to 15Min, the same periods become 20-bar and 50-bar averages.
    """
    if price_data is None or price_data.empty:
        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Price data is missing.",
        )

    if "close" not in price_data.columns:
        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Price data does not include a close column.",
        )

    close_prices = price_data["close"].dropna()
    required_bars = settings.slow_ma_period

    if len(close_prices) < required_bars:
        return StrategyDecision(
            action=TradeAction.HOLD,
            reason=f"Need at least {required_bars} close prices, got {len(close_prices)}.",
        )

    latest_close = float(close_prices.iloc[-1])
    fast_ma = float(close_prices.rolling(settings.fast_ma_period).mean().iloc[-1])
    slow_ma = float(close_prices.rolling(settings.slow_ma_period).mean().iloc[-1])

    logger.info("Latest close: %.2f", latest_close)
    logger.info("MA%d: %.2f", settings.fast_ma_period, fast_ma)
    logger.info("MA%d: %.2f", settings.slow_ma_period, slow_ma)
    logger.info("Already owns SPY: %s", owns_position)

    if fast_ma > slow_ma and not owns_position:
        return StrategyDecision(
            action=TradeAction.BUY,
            reason=(
                f"MA{settings.fast_ma_period} is above MA{settings.slow_ma_period} "
                "and no SPY position is currently held."
            ),
            latest_close=latest_close,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
        )

    if fast_ma < slow_ma and owns_position:
        return StrategyDecision(
            action=TradeAction.SELL,
            reason=(
                f"MA{settings.fast_ma_period} is below MA{settings.slow_ma_period} "
                "and a SPY position is currently held."
            ),
            latest_close=latest_close,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
        )

    return StrategyDecision(
        action=TradeAction.HOLD,
        reason="Moving-average signal does not require a position change.",
        latest_close=latest_close,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
    )
