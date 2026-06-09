"""Historical market data helpers for backtests and preflight checks."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from requests.exceptions import RequestException

from config import Settings
from data import parse_data_feed


class HistoricalDataError(RuntimeError):
    """Raised when historical market data cannot be fetched."""


def parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD text into a timezone-aware UTC datetime."""
    try:
        parsed_date = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from error

    return parsed_date.replace(tzinfo=timezone.utc)


def fetch_daily_bars(
    settings: Settings,
    start: datetime,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download daily adjusted bars for the configured symbol.

    This uses Alpaca's data client only. It does not create TradingClient and
    cannot place orders.
    """
    client = StockHistoricalDataClient(settings.api_key, settings.secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=settings.symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=parse_data_feed(settings.data_feed),
        adjustment=Adjustment.ALL,
    )

    try:
        bars = client.get_stock_bars(request)
    except (APIError, RequestException) as error:
        raise HistoricalDataError(f"Failed to fetch historical data: {error}") from error

    data_frame = bars.df
    if data_frame is None or data_frame.empty:
        return pd.DataFrame()

    if isinstance(data_frame.index, pd.MultiIndex):
        symbol_level = "symbol" if "symbol" in data_frame.index.names else 0
        data_frame = data_frame.xs(settings.symbol, level=symbol_level)

    return data_frame.sort_index()
