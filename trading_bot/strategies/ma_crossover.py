"""Moving-average crossover strategy."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class MovingAverageCrossoverStrategy(Strategy):
    """Buy when the fast moving average is above the slow moving average."""

    name = "ma_crossover"

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.display_name = f"MA{fast_period}/MA{slow_period} crossover"

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        close_prices = price_data["close"].dropna()
        fast_ma = close_prices.rolling(self.fast_period).mean()
        slow_ma = close_prices.rolling(self.slow_period).mean()

        signal = pd.Series(0, index=close_prices.index, dtype=int)
        signal.loc[fast_ma > slow_ma] = 1
        signal.loc[fast_ma.isna() | slow_ma.isna()] = 0
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
        if len(close_prices) < self.slow_period:
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason=f"Need at least {self.slow_period} close prices, got {len(close_prices)}.",
            )

        latest_close = float(close_prices.iloc[-1])
        fast_ma = float(close_prices.rolling(self.fast_period).mean().iloc[-1])
        slow_ma = float(close_prices.rolling(self.slow_period).mean().iloc[-1])

        if logger:
            logger.info("Latest close: %.2f", latest_close)
            logger.info("MA%d: %.2f", self.fast_period, fast_ma)
            logger.info("MA%d: %.2f", self.slow_period, slow_ma)
            logger.info("Already owns SPY: %s", owns_position)

        if fast_ma > slow_ma and not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason=(
                    f"MA{self.fast_period} is above MA{self.slow_period} "
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
                    f"MA{self.fast_period} is below MA{self.slow_period} "
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
