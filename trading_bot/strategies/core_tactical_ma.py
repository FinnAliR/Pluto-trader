"""Core-plus-tactical moving-average strategy."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class CoreTacticalMovingAverageStrategy(Strategy):
    """Keep a core SPY allocation and use MA trend for the tactical portion."""

    name = "core_tactical_ma"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        core_fraction: float = 0.40,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.core_fraction = core_fraction
        self.display_name = f"Core {core_fraction:.0%} + MA{fast_period}/MA{slow_period}"

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        close_prices = price_data["close"].dropna()
        fast_ma = close_prices.rolling(self.fast_period).mean()
        slow_ma = close_prices.rolling(self.slow_period).mean()

        signal = pd.Series(self.core_fraction, index=close_prices.index, dtype=float)
        signal.loc[fast_ma > slow_ma] = 1.0
        signal.loc[fast_ma.isna() | slow_ma.isna()] = self.core_fraction
        return signal.reindex(price_data.index).ffill().fillna(self.core_fraction).clip(0, 1)

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
            latest_close = float(close_prices.iloc[-1]) if not close_prices.empty else None
            if not owns_position and latest_close is not None:
                return StrategyDecision(
                    action=TradeAction.BUY,
                    reason=(
                        f"Core allocation wants {self.core_fraction:.0%} of configured exposure "
                        "while waiting for enough MA history."
                    ),
                    latest_close=latest_close,
                    indicator_name="target_fraction",
                    indicator_value=self.core_fraction,
                    target_fraction=self.core_fraction,
                )
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason=(
                    f"Need at least {self.slow_period} close prices, got {len(close_prices)}. "
                    f"Maintaining the {self.core_fraction:.0%} core allocation target."
                ),
                latest_close=latest_close,
                indicator_name="target_fraction",
                indicator_value=self.core_fraction,
                target_fraction=self.core_fraction,
            )

        latest_close = float(close_prices.iloc[-1])
        fast_ma = float(close_prices.rolling(self.fast_period).mean().iloc[-1])
        slow_ma = float(close_prices.rolling(self.slow_period).mean().iloc[-1])
        target_fraction = 1.0 if fast_ma > slow_ma else self.core_fraction

        if logger:
            logger.info("Latest close: %.2f", latest_close)
            logger.info("MA%d: %.2f", self.fast_period, fast_ma)
            logger.info("MA%d: %.2f", self.slow_period, slow_ma)
            logger.info("Core tactical target fraction: %.0f%%", target_fraction * 100)
            logger.info("Already owns SPY: %s", owns_position)

        if target_fraction > 0 and not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason=f"Core tactical strategy wants {target_fraction:.0%} of configured exposure.",
                latest_close=latest_close,
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                indicator_name="target_fraction",
                indicator_value=target_fraction,
                target_fraction=target_fraction,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Core tactical target is active; live bot will rebalance if current exposure is off target.",
            latest_close=latest_close,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            indicator_name="target_fraction",
            indicator_value=target_fraction,
            target_fraction=target_fraction,
        )
