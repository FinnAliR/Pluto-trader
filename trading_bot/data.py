"""Market data helpers for Alpaca stock bars."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from logging import Logger

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from requests.exceptions import RequestException

from config import Settings, SettingsError


class MarketDataError(RuntimeError):
    """Raised when Alpaca market data API calls fail."""


def create_data_client(settings: Settings) -> StockHistoricalDataClient:
    """Create the official alpaca-py stock data client."""
    return StockHistoricalDataClient(settings.api_key, settings.secret_key)


def get_price_data(
    data_client: StockHistoricalDataClient,
    settings: Settings,
    logger: Logger,
) -> pd.DataFrame:
    """Fetch historical bars for the configured symbol and timeframe."""
    timeframe = parse_timeframe(settings.timeframe)
    data_feed = parse_data_feed(settings.data_feed)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=settings.lookback_days)

    logger.info(
        "Requesting %s bars for %s from %s to %s using %s feed.",
        settings.timeframe,
        settings.symbol,
        start.date(),
        end.date(),
        settings.data_feed,
    )

    request = StockBarsRequest(
        symbol_or_symbols=settings.symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=data_feed,
        adjustment=Adjustment.ALL,
    )

    try:
        bars = data_client.get_stock_bars(request)
    except (APIError, RequestException) as error:
        raise MarketDataError(f"Failed to fetch Alpaca price data: {error}") from error

    data_frame = bars.df

    if data_frame is None or data_frame.empty:
        logger.warning("No price data returned for %s.", settings.symbol)
        return pd.DataFrame()

    # Alpaca returns a multi-index dataframe for bars. For one symbol, keep only
    # that symbol's rows so strategy code can stay simple.
    if isinstance(data_frame.index, pd.MultiIndex):
        symbol_level = "symbol" if "symbol" in data_frame.index.names else 0
        data_frame = data_frame.xs(settings.symbol, level=symbol_level)

    data_frame = data_frame.sort_index()
    logger.info("Received %d bars for %s.", len(data_frame), settings.symbol)
    return data_frame


def parse_timeframe(timeframe_value: str) -> TimeFrame:
    """Convert a simple config string into an alpaca-py TimeFrame."""
    normalized = timeframe_value.strip().lower()

    if normalized in {"1day", "day", "daily", "1d"}:
        return TimeFrame.Day

    if normalized in {"15min", "15m", "15minute", "15minutes"}:
        return TimeFrame(15, TimeFrameUnit.Minute)

    raise SettingsError("TIMEFRAME must be either 1Day or 15Min.")


def parse_data_feed(feed_value: str) -> DataFeed:
    """Convert DATA_FEED text into an alpaca-py DataFeed enum."""
    normalized = feed_value.strip().lower()

    if normalized == "iex":
        return DataFeed.IEX

    if normalized == "sip":
        return DataFeed.SIP

    raise SettingsError("DATA_FEED must be iex or sip.")
