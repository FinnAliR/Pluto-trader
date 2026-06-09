"""Simple price breakout strategy."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class BreakoutStrategy(Strategy):
    """Buy a close above the prior high range and sell below the prior low range."""

    name = "breakout"

    def __init__(self, entry_period: int = 20, exit_period: int = 10) -> None:
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.display_name = f"{entry_period}-day breakout / {exit_period}-day exit"

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        close_prices = price_data["close"].dropna()
        prior_high = close_prices.rolling(self.entry_period).max().shift(1)
        prior_low = close_prices.rolling(self.exit_period).min().shift(1)

        signal = pd.Series(0, index=close_prices.index, dtype=int)
        target_position = 0

        for timestamp, close_price in close_prices.items():
            high_value = prior_high.loc[timestamp]
            low_value = prior_low.loc[timestamp]

            if pd.isna(high_value) or pd.isna(low_value):
                signal.loc[timestamp] = target_position
                continue

            if target_position == 0 and close_price > high_value:
                target_position = 1
            elif target_position == 1 and close_price < low_value:
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
        required_bars = max(self.entry_period, self.exit_period) + 1

        if len(close_prices) < required_bars:
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason=f"Need at least {required_bars} close prices, got {len(close_prices)}.",
            )

        latest_close = float(close_prices.iloc[-1])
        prior_high = float(close_prices.rolling(self.entry_period).max().shift(1).iloc[-1])
        prior_low = float(close_prices.rolling(self.exit_period).min().shift(1).iloc[-1])

        if logger:
            logger.info("Latest close: %.2f", latest_close)
            logger.info("Prior %d-day high: %.2f", self.entry_period, prior_high)
            logger.info("Prior %d-day low: %.2f", self.exit_period, prior_low)
            logger.info("Already owns SPY: %s", owns_position)

        if latest_close > prior_high and not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason=f"Close is above the prior {self.entry_period}-day high and no SPY position is held.",
                latest_close=latest_close,
                indicator_name=f"PriorHigh{self.entry_period}",
                indicator_value=prior_high,
            )

        if latest_close < prior_low and owns_position:
            return StrategyDecision(
                action=TradeAction.SELL,
                reason=f"Close is below the prior {self.exit_period}-day low and a SPY position is held.",
                latest_close=latest_close,
                indicator_name=f"PriorLow{self.exit_period}",
                indicator_value=prior_low,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Breakout signal does not require a position change.",
            latest_close=latest_close,
            indicator_name=f"PriorHigh{self.entry_period}",
            indicator_value=prior_high,
        )
