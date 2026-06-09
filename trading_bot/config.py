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
    strategy_name: str
    timeframe: str
    data_feed: str
    fast_ma_period: int
    slow_ma_period: int
    rsi_period: int
    rsi_buy_level: float
    rsi_sell_level: float
    breakout_entry_period: int
    breakout_exit_period: int
    buying_power_fraction: float
    max_daily_trades: int
    max_position_size_fraction: float
    trial_mode: bool
    trial_max_notional: float
    lookback_days: int
    min_order_notional: float
    client_order_id_prefix: str
    trades_csv_path: Path
    log_level: str


def load_settings() -> Settings:
    """Load settings from a .env file and environment variables."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    api_key = _required_env("ALPACA_API_KEY")
    secret_key = _required_env("ALPACA_SECRET_KEY")

    paper_trading_value = os.getenv("PAPER_TRADING", "").strip().lower()
    alpaca_paper_value = os.getenv("ALPACA_PAPER", "").strip().lower()

    if paper_trading_value not in TRUE_VALUES:
        raise SettingsError(
            "This bot only supports paper trading. Set PAPER_TRADING=true in your .env file."
        )

    if alpaca_paper_value not in TRUE_VALUES:
        raise SettingsError(
            "This bot only supports Alpaca paper trading. Set ALPACA_PAPER=true in your .env file."
        )

    bot_dir = Path(__file__).resolve().parent

    settings = Settings(
        api_key=api_key,
        secret_key=secret_key,
        paper_trading=True,
        symbol=os.getenv("SYMBOL", "SPY").strip().upper(),
        strategy_name=_normalize_strategy_name(os.getenv("STRATEGY", "ma_crossover")),
        timeframe=_normalize_timeframe(os.getenv("TIMEFRAME", "1Day")),
        data_feed=os.getenv("DATA_FEED", "iex").strip().lower(),
        fast_ma_period=_int_env("FAST_MA_PERIOD", 20),
        slow_ma_period=_int_env("SLOW_MA_PERIOD", 50),
        rsi_period=_int_env("RSI_PERIOD", 14),
        rsi_buy_level=_float_env("RSI_BUY_LEVEL", 30.0),
        rsi_sell_level=_float_env("RSI_SELL_LEVEL", 70.0),
        breakout_entry_period=_int_env("BREAKOUT_ENTRY_PERIOD", 20),
        breakout_exit_period=_int_env("BREAKOUT_EXIT_PERIOD", 10),
        buying_power_fraction=_float_env("BUYING_POWER_FRACTION", 0.25),
        max_daily_trades=_int_env("MAX_DAILY_TRADES", 1),
        max_position_size_fraction=_float_env("MAX_POSITION_SIZE_FRACTION", 0.25),
        trial_mode=_bool_env("TRIAL_MODE", False),
        trial_max_notional=_float_env("TRIAL_MAX_NOTIONAL", 25.00),
        lookback_days=_int_env("LOOKBACK_DAYS", 220),
        min_order_notional=_float_env("MIN_ORDER_NOTIONAL", 1.00),
        client_order_id_prefix=os.getenv("CLIENT_ORDER_ID_PREFIX", "pluto-mac").strip(),
        trades_csv_path=_path_env("TRADES_CSV", bot_dir / "trades.csv", bot_dir),
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


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in TRUE_VALUES


def _path_env(name: str, default: Path, base_dir: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    path = Path(raw_value.strip())
    if not path.is_absolute():
        path = base_dir / path
    return path


def _normalize_timeframe(value: str) -> str:
    normalized = value.strip().lower()

    if normalized in {"1day", "day", "daily", "1d"}:
        return "1Day"

    if normalized in {"15min", "15m", "15minute", "15minutes"}:
        return "15Min"

    raise SettingsError("TIMEFRAME must be either 1Day or 15Min.")


def _normalize_strategy_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "ma": "ma_crossover",
        "moving_average": "ma_crossover",
        "ma_crossover": "ma_crossover",
        "rsi": "rsi",
        "breakout": "breakout",
        "buy_hold": "buy_and_hold",
        "buy_and_hold": "buy_and_hold",
    }

    if normalized not in aliases:
        allowed = ", ".join(["ma_crossover", "rsi", "breakout", "buy_and_hold"])
        raise SettingsError(f"STRATEGY must be one of: {allowed}.")

    return aliases[normalized]


def _validate_settings(settings: Settings) -> None:
    if settings.symbol != "SPY":
        raise SettingsError("This beginner bot is locked to SYMBOL=SPY.")

    if settings.fast_ma_period <= 0 or settings.slow_ma_period <= 0:
        raise SettingsError("Moving-average periods must be positive integers.")

    if settings.fast_ma_period >= settings.slow_ma_period:
        raise SettingsError("FAST_MA_PERIOD must be smaller than SLOW_MA_PERIOD.")

    if settings.rsi_period <= 1:
        raise SettingsError("RSI_PERIOD must be greater than 1.")

    if not 0 < settings.rsi_buy_level < settings.rsi_sell_level < 100:
        raise SettingsError("Require 0 < RSI_BUY_LEVEL < RSI_SELL_LEVEL < 100.")

    if settings.breakout_entry_period <= 1 or settings.breakout_exit_period <= 1:
        raise SettingsError("Breakout periods must be greater than 1.")

    if not 0 < settings.buying_power_fraction <= 0.25:
        raise SettingsError("BUYING_POWER_FRACTION must be greater than 0 and no more than 0.25.")

    if settings.max_daily_trades < 1:
        raise SettingsError("MAX_DAILY_TRADES must be at least 1.")

    if not 0 < settings.max_position_size_fraction <= 0.25:
        raise SettingsError("MAX_POSITION_SIZE_FRACTION must be greater than 0 and no more than 0.25.")

    if settings.trial_max_notional <= 0:
        raise SettingsError("TRIAL_MAX_NOTIONAL must be greater than 0.")

    if settings.lookback_days < 60:
        raise SettingsError("LOOKBACK_DAYS must be at least 60 for the 50-period average.")

    if settings.min_order_notional <= 0:
        raise SettingsError("MIN_ORDER_NOTIONAL must be greater than 0.")

    if len(settings.client_order_id_prefix) > 20:
        raise SettingsError("CLIENT_ORDER_ID_PREFIX must be 20 characters or fewer.")
