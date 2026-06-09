"""Configuration loading and paper-trading safety checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class SettingsError(ValueError):
    """Raised when the bot configuration is missing or unsafe."""


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the trading bot."""

    api_key: str
    secret_key: str
    paper_trading: bool
    symbol: str
    timeframe: str
    data_feed: str
    fast_ma_period: int
    slow_ma_period: int
    buying_power_fraction: float
    lookback_days: int
    min_order_notional: float
    client_order_id_prefix: str
    log_level: str


def load_settings() -> Settings:
    """Load settings from a .env file and environment variables.

    The .env file is expected to live in the same directory as this file.
    Environment variables already set in the shell can override the file.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    api_key = _required_env("ALPACA_API_KEY")
    secret_key = _required_env("ALPACA_SECRET_KEY")

    paper_value = os.getenv("ALPACA_PAPER", "true").strip().lower()
    if paper_value not in TRUE_VALUES:
        raise SettingsError(
            "This bot only supports Alpaca paper trading. Set ALPACA_PAPER=true."
        )

    settings = Settings(
        api_key=api_key,
        secret_key=secret_key,
        paper_trading=True,
        symbol=os.getenv("SYMBOL", "SPY").strip().upper(),
        timeframe=_normalize_timeframe(os.getenv("TIMEFRAME", "1Day")),
        data_feed=os.getenv("DATA_FEED", "iex").strip().lower(),
        fast_ma_period=_int_env("FAST_MA_PERIOD", 20),
        slow_ma_period=_int_env("SLOW_MA_PERIOD", 50),
        buying_power_fraction=_float_env("BUYING_POWER_FRACTION", 0.25),
        lookback_days=_int_env("LOOKBACK_DAYS", 220),
        min_order_notional=_float_env("MIN_ORDER_NOTIONAL", 1.00),
        client_order_id_prefix=os.getenv("CLIENT_ORDER_ID_PREFIX", "pluto-mac").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    )

    _validate_settings(settings)
    return settings


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SettingsError(f"Missing required environment variable: {name}")
    if value.lower() in {"your_api_key_here", "your_secret_key_here"}:
        raise SettingsError(f"Replace the placeholder value for {name} in your .env file.")
    return value


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise SettingsError(f"{name} must be an integer.") from error


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise SettingsError(f"{name} must be a number.") from error


def _normalize_timeframe(value: str) -> str:
    normalized = value.strip().lower()

    if normalized in {"1day", "day", "daily", "1d"}:
        return "1Day"

    if normalized in {"15min", "15m", "15minute", "15minutes"}:
        return "15Min"

    raise SettingsError("TIMEFRAME must be either 1Day or 15Min.")


def _validate_settings(settings: Settings) -> None:
    if settings.symbol != "SPY":
        raise SettingsError("This beginner bot is locked to SYMBOL=SPY.")

    if settings.fast_ma_period <= 0 or settings.slow_ma_period <= 0:
        raise SettingsError("Moving-average periods must be positive integers.")

    if settings.fast_ma_period >= settings.slow_ma_period:
        raise SettingsError("FAST_MA_PERIOD must be smaller than SLOW_MA_PERIOD.")

    if not 0 < settings.buying_power_fraction <= 0.25:
        raise SettingsError("BUYING_POWER_FRACTION must be greater than 0 and no more than 0.25.")

    if settings.lookback_days < 60:
        raise SettingsError("LOOKBACK_DAYS must be at least 60 for the 50-period average.")

    if settings.min_order_notional <= 0:
        raise SettingsError("MIN_ORDER_NOTIONAL must be greater than 0.")

    if len(settings.client_order_id_prefix) > 20:
        raise SettingsError("CLIENT_ORDER_ID_PREFIX must be 20 characters or fewer.")
