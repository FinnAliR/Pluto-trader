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
    """The strategy output passed to risk checks.

    target_fraction is the desired final SPY exposure as a fraction of the
    configured max position size. None means use the action default.
    """

    action: TradeAction
    reason: str
    latest_close: float | None = None
    fast_ma: float | None = None
    slow_ma: float | None = None
    indicator_name: str = ""
    indicator_value: float | None = None
    target_fraction: float | None = None


class Strategy(ABC):
    """Base class every strategy must follow."""

    name: str
    display_name: str

    @abstractmethod
    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Return a target-position series from 0.0 to 1.0.

        1.0 means use the full configured exposure.
        0.0 means stay in cash.
        Fractional values allow core/tactical strategies, for example 0.4.
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
