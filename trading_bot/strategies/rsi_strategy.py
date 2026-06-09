"""Simple RSI mean-reversion strategy."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class RsiStrategy(Strategy):
    """Buy oversold RSI and sell overbought RSI."""

    name = "rsi"

    def __init__(self, period: int = 14, buy_level: float = 30.0, sell_level: float = 70.0) -> None:
        self.period = period
        self.buy_level = buy_level
        self.sell_level = sell_level
        self.display_name = f"RSI{period} {buy_level:.0f}/{sell_level:.0f}"

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        close_prices = price_data["close"].dropna()
        rsi = calculate_rsi(close_prices, self.period)

        signal = pd.Series(0, index=close_prices.index, dtype=int)
        target_position = 0

        for timestamp, value in rsi.items():
            if pd.isna(value):
                signal.loc[timestamp] = target_position
                continue

            if target_position == 0 and value < self.buy_level:
                target_position = 1
            elif target_position == 1 and value > self.sell_level:
                target_position = 0

            signal.loc[timestamp] = target_position

        return signal.reindex(price_data.index).fillna(0).astype(int)

    def make_decision(
        self,
        price_data: pd.DataFrame,
        owns_position: bool,
        logger: Logger | None = None,
    ) -> StrategyDecision:
        if price_data is None or price_data.empty:
            return self._empty_data_decision()

        if "close" not in price_data.columns:
            return self._missing_close_decision()

        close_prices = price_data["close"].dropna()
        if len(close_prices) < self.period + 1:
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason=f"Need at least {self.period + 1} close prices, got {len(close_prices)}.",
            )

        latest_close = float(close_prices.iloc[-1])
        latest_rsi = float(calculate_rsi(close_prices, self.period).iloc[-1])

        if logger:
            logger.info("Latest close: %.2f", latest_close)
            logger.info("RSI%d: %.2f", self.period, latest_rsi)
            logger.info("Already owns SPY: %s", owns_position)

        if latest_rsi < self.buy_level and not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason=f"RSI{self.period} is below {self.buy_level:.0f} and no SPY position is held.",
                latest_close=latest_close,
                indicator_name=f"RSI{self.period}",
                indicator_value=latest_rsi,
            )

        if latest_rsi > self.sell_level and owns_position:
            return StrategyDecision(
                action=TradeAction.SELL,
                reason=f"RSI{self.period} is above {self.sell_level:.0f} and a SPY position is held.",
                latest_close=latest_close,
                indicator_name=f"RSI{self.period}",
                indicator_value=latest_rsi,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="RSI signal does not require a position change.",
            latest_close=latest_close,
            indicator_name=f"RSI{self.period}",
            indicator_value=latest_rsi,
        )


def calculate_rsi(close_prices: pd.Series, period: int) -> pd.Series:
    """Calculate a beginner-friendly simple RSI."""
    price_change = close_prices.diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(period).mean()
    average_loss = losses.rolling(period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.where(average_loss != 0, 100)
    rsi = rsi.where(average_gain != 0, 0)
    rsi = rsi.where(~((average_gain == 0) & (average_loss == 0)), 50)
    return rsi
