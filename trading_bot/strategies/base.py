"""Shared strategy types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from logging import Logger

import pandas as pd


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
    indicator_name: str = ""
    indicator_value: float | None = None


class Strategy(ABC):
    """Base class every strategy must follow."""

    name: str
    display_name: str

    @abstractmethod
    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Return a 0/1 target-position series.

        1 means the strategy wants to own SPY.
        0 means the strategy wants to be in cash.
        """

    @abstractmethod
    def make_decision(
        self,
        price_data: pd.DataFrame,
        owns_position: bool,
        logger: Logger | None = None,
    ) -> StrategyDecision:
        """Return BUY, SELL, or HOLD for the latest available candle."""

    def _empty_data_decision(self) -> StrategyDecision:
        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Price data is missing.",
        )

    def _missing_close_decision(self) -> StrategyDecision:
        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Price data does not include a close column.",
        )
