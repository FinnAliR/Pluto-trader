"""Buy-and-hold reference strategy."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class BuyAndHoldStrategy(Strategy):
    """Buy SPY once and hold it."""

    name = "buy_and_hold"
    display_name = "Buy and hold"

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        signal = pd.Series(1, index=price_data.index, dtype=int)
        signal.loc[price_data["close"].isna()] = 0
        return signal

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
        if close_prices.empty:
            return self._empty_data_decision()

        latest_close = float(close_prices.iloc[-1])

        if logger:
            logger.info("Latest close: %.2f", latest_close)
            logger.info("Already owns SPY: %s", owns_position)

        if not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason="Buy-and-hold strategy wants one SPY position and none is currently held.",
                latest_close=latest_close,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Buy-and-hold strategy already has a SPY position.",
            latest_close=latest_close,
        )
