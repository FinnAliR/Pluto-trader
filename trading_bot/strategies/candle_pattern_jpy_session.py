"""Candle-pattern strategy adapted from the JPY session notebook."""

from __future__ import annotations

from logging import Logger

import pandas as pd

from .base import Strategy, StrategyDecision, TradeAction


class CandlePatternJpySessionStrategy(Strategy):
    """Long/flat adaptation of the notebook candle reversal pattern.

    The source notebook used backtesting.py with long and short stop orders.
    This project backtester is long-only, so bullish notebook signals enter or
    hold long exposure and bearish notebook signals exit to cash.
    """

    name = "candle_pattern_jpy_session"

    def __init__(
        self,
        return_threshold: float = 0.005,
        max_hold_bars: int = 2,
        profit_take_return: float = 0.0,
    ) -> None:
        self.return_threshold = return_threshold
        self.max_hold_bars = max_hold_bars
        self.profit_take_return = profit_take_return
        self.display_name = (
            f"Candle pattern ret>{return_threshold:.2%} "
            f"hold<={max_hold_bars} profit>{profit_take_return:.2%}"
        )

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if not _has_ohlc(price_data):
            return pd.Series(dtype=float)

        data = _ohlc_frame(price_data)
        bullish, bearish = _candle_pattern_signals(data, self.return_threshold)
        signal = pd.Series(0.0, index=data.index)

        in_position = False
        entry_close = 0.0
        entry_pos = 0

        for position, timestamp in enumerate(data.index):
            close_price = float(data["close"].iloc[position])

            if in_position:
                bars_held = position - entry_pos
                trade_return = (close_price / entry_close) - 1 if entry_close > 0 else 0.0
                if (
                    bearish.iloc[position]
                    or bars_held >= self.max_hold_bars
                    or trade_return > self.profit_take_return
                ):
                    in_position = False

            if not in_position and bullish.iloc[position]:
                in_position = True
                entry_close = close_price
                entry_pos = position

            signal.loc[timestamp] = 1.0 if in_position else 0.0

        return signal.reindex(price_data.index).ffill().fillna(0.0).clip(0, 1)

    def make_decision(
        self,
        price_data: pd.DataFrame,
        owns_position: bool,
        logger: Logger | None = None,
    ) -> StrategyDecision:
        if price_data is None or price_data.empty:
            return self._empty_data_decision()

        if not _has_ohlc(price_data):
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason="Candle pattern requires high, low, and close columns.",
            )

        data = _ohlc_frame(price_data)
        if len(data) < 3:
            return StrategyDecision(
                action=TradeAction.HOLD,
                reason=f"Need at least 3 candles, got {len(data)}.",
            )

        bullish, bearish = _candle_pattern_signals(data, self.return_threshold)
        latest_close = float(data["close"].iloc[-1])
        latest_bullish = bool(bullish.iloc[-1])
        latest_bearish = bool(bearish.iloc[-1])

        if logger:
            logger.info("Latest close: %.5f", latest_close)
            logger.info("Candle bullish signal: %s", latest_bullish)
            logger.info("Candle bearish signal: %s", latest_bearish)
            logger.info("Already owns position: %s", owns_position)

        if latest_bullish and not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason="Bullish candle-pattern reversal is active and no position is held.",
                latest_close=latest_close,
                indicator_name="candle_signal",
                indicator_value=2.0,
            )

        if latest_bearish and owns_position:
            return StrategyDecision(
                action=TradeAction.SELL,
                reason="Bearish candle-pattern reversal is active and a position is held.",
                latest_close=latest_close,
                indicator_name="candle_signal",
                indicator_value=1.0,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Candle-pattern signal does not require a position change.",
            latest_close=latest_close,
            indicator_name="candle_signal",
            indicator_value=2.0 if latest_bullish else 1.0 if latest_bearish else 0.0,
        )


def _has_ohlc(price_data: pd.DataFrame) -> bool:
    if price_data is None or price_data.empty:
        return False
    columns = {str(column).lower() for column in price_data.columns}
    return {"high", "low", "close"}.issubset(columns)


def _ohlc_frame(price_data: pd.DataFrame) -> pd.DataFrame:
    columns = {str(column).lower(): column for column in price_data.columns}
    return pd.DataFrame(
        {
            "high": price_data[columns["high"]].astype(float),
            "low": price_data[columns["low"]].astype(float),
            "close": price_data[columns["close"]].astype(float),
        },
        index=price_data.index,
    ).dropna()


def _candle_pattern_signals(data: pd.DataFrame, return_threshold: float) -> tuple[pd.Series, pd.Series]:
    low = data["low"]
    high = data["high"]
    close = data["close"]

    bullish = (
        (low > low.shift(1))
        & (low.shift(1) < low.shift(2))
        & (((close - close.shift(1)) / close.shift(1)) > return_threshold)
    )
    bearish = (
        (high < high.shift(1))
        & (high.shift(1) > high.shift(2))
        & (((close.shift(1) - close) / close) > return_threshold)
    )
    return bullish.fillna(False), bearish.fillna(False)
