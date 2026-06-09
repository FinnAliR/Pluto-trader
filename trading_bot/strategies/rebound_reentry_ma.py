"""Moving-average strategy with faster rebound re-entry."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class ReboundReentryMovingAverageStrategy(Strategy):
    """Use MA exits but re-enter faster after strong rebounds."""

    name = "rebound_reentry_ma"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        rebound_ma_period: int = 10,
        rebound_return_days: int = 3,
        rebound_min_return: float = 0.0,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rebound_ma_period = rebound_ma_period
        self.rebound_return_days = rebound_return_days
        self.rebound_min_return = rebound_min_return
        self.display_name = (
            f"MA{fast_period}/MA{slow_period} + "
            f"{rebound_return_days}d rebound re-entry"
        )

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        close_prices = price_data["close"].dropna()
        fast_ma = close_prices.rolling(self.fast_period).mean()
        slow_ma = close_prices.rolling(self.slow_period).mean()
        rebound_ma = close_prices.rolling(self.rebound_ma_period).mean()
        rebound_return = close_prices.pct_change(self.rebound_return_days)

        signal = pd.Series(0.0, index=close_prices.index)
        target_position = 0.0

        for timestamp, close_price in close_prices.items():
            bullish_trend = fast_ma.loc[timestamp] > slow_ma.loc[timestamp]
            bearish_trend = fast_ma.loc[timestamp] < slow_ma.loc[timestamp]
            rebound_entry = (
                close_price > rebound_ma.loc[timestamp]
                and rebound_return.loc[timestamp] > self.rebound_min_return
            )

            if pd.isna(fast_ma.loc[timestamp]) or pd.isna(slow_ma.loc[timestamp]):
                signal.loc[timestamp] = target_position
                continue

            if target_position == 0 and (bullish_trend or rebound_entry):
                target_position = 1.0
            elif target_position > 0 and bearish_trend and not rebound_entry:
                target_position = 0.0

            signal.loc[timestamp] = target_position

        return signal.reindex(price_data.index).ffill().fillna(0.0).clip(0, 1)

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
        required_bars = max(self.slow_period, self.rebound_ma_period, self.rebound_return_days + 1)
        if len(close_prices) < required_bars:
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason=f"Need at least {required_bars} close prices, got {len(close_prices)}.",
            )

        latest_close = float(close_prices.iloc[-1])
        fast_ma = float(close_prices.rolling(self.fast_period).mean().iloc[-1])
        slow_ma = float(close_prices.rolling(self.slow_period).mean().iloc[-1])
        rebound_ma = float(close_prices.rolling(self.rebound_ma_period).mean().iloc[-1])
        rebound_return = float(close_prices.pct_change(self.rebound_return_days).iloc[-1])

        bullish_trend = fast_ma > slow_ma
        bearish_trend = fast_ma < slow_ma
        rebound_entry = latest_close > rebound_ma and rebound_return > self.rebound_min_return

        if logger:
            logger.info("Latest close: %.2f", latest_close)
            logger.info("MA%d: %.2f", self.fast_period, fast_ma)
            logger.info("MA%d: %.2f", self.slow_period, slow_ma)
            logger.info("Rebound MA%d: %.2f", self.rebound_ma_period, rebound_ma)
            logger.info("%d-day rebound return: %.2f%%", self.rebound_return_days, rebound_return * 100)
            logger.info("Already owns SPY: %s", owns_position)

        if (bullish_trend or rebound_entry) and not owns_position:
            reason = "Bullish MA trend" if bullish_trend else "Rebound re-entry condition"
            return StrategyDecision(
                action=TradeAction.BUY,
                reason=f"{reason} is active and no SPY position is held.",
                latest_close=latest_close,
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                indicator_name=f"{self.rebound_return_days}d_return",
                indicator_value=rebound_return,
            )

        if bearish_trend and not rebound_entry and owns_position:
            return StrategyDecision(
                action=TradeAction.SELL,
                reason="MA trend is bearish and rebound re-entry condition is not active.",
                latest_close=latest_close,
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                indicator_name=f"{self.rebound_return_days}d_return",
                indicator_value=rebound_return,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Rebound MA signal does not require a position change.",
            latest_close=latest_close,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            indicator_name=f"{self.rebound_return_days}d_return",
            indicator_value=rebound_return,
        )
