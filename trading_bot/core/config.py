"""Configuration loading and paper-trading safety checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class SettingsError(ValueError):
    """Raised when the bot configuration is missing or unsafe."""


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
BOT_DIR = Path(__file__).resolve().parents[1]
EXECUTION_MODE_ALPACA_PAPER = "alpaca_paper"
EXECUTION_MODE_LIVE_PAPER = "live_paper"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the trading bot."""

    api_key: str
    secret_key: str
    paper_trading: bool
    execution_mode: str
    symbol: str
    strategy_name: str
    timeframe: str
    data_feed: str
    fast_ma_period: int
    slow_ma_period: int
    core_tactical_core_fraction: float
    rebound_ma_period: int
    rebound_return_days: int
    rebound_min_return: float
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
    live_paper_initial_cash: float
    live_paper_state_path: Path
    lookback_days: int
    min_order_notional: float
    client_order_id_prefix: str
    trades_csv_path: Path
    log_level: str


def load_settings() -> Settings:
    """Load settings from a .env file and environment variables."""
    env_path = BOT_DIR / ".env"
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

    bot_dir = BOT_DIR

    settings = Settings(
        api_key=api_key,
        secret_key=secret_key,
        paper_trading=True,
        execution_mode=_normalize_execution_mode(os.getenv("EXECUTION_MODE", EXECUTION_MODE_ALPACA_PAPER)),
        symbol=os.getenv("SYMBOL", "SPY").strip().upper(),
        strategy_name=_normalize_strategy_name(os.getenv("STRATEGY", "ma_crossover")),
        timeframe=_normalize_timeframe(os.getenv("TIMEFRAME", "1Day")),
        data_feed=os.getenv("DATA_FEED", "iex").strip().lower(),
        fast_ma_period=_int_env("FAST_MA_PERIOD", 20),
        slow_ma_period=_int_env("SLOW_MA_PERIOD", 50),
        core_tactical_core_fraction=_float_env("CORE_TACTICAL_CORE_FRACTION", 0.40),
        rebound_ma_period=_int_env("REBOUND_MA_PERIOD", 10),
        rebound_return_days=_int_env("REBOUND_RETURN_DAYS", 3),
        rebound_min_return=_float_env("REBOUND_MIN_RETURN", 0.0),
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
        live_paper_initial_cash=_float_env("LIVE_PAPER_INITIAL_CASH", 10_000.00),
        live_paper_state_path=_path_env("LIVE_PAPER_STATE_PATH", bot_dir / "live_paper_state.json", bot_dir),
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
    from trading_bot.strategies.registry import available_strategy_names, normalize_strategy_name

    try:
        return normalize_strategy_name(value)
    except ValueError as error:
        allowed = ", ".join(available_strategy_names())
        raise SettingsError(f"STRATEGY must be one of: {allowed}.") from error


def _normalize_execution_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "alpaca": EXECUTION_MODE_ALPACA_PAPER,
        "paper": EXECUTION_MODE_ALPACA_PAPER,
        "paper_alpaca": EXECUTION_MODE_ALPACA_PAPER,
        "alpaca_paper": EXECUTION_MODE_ALPACA_PAPER,
        "broker_paper": EXECUTION_MODE_ALPACA_PAPER,
        "live_paper": EXECUTION_MODE_LIVE_PAPER,
        "local_paper": EXECUTION_MODE_LIVE_PAPER,
        "sim": EXECUTION_MODE_LIVE_PAPER,
        "simulated": EXECUTION_MODE_LIVE_PAPER,
        "simulation": EXECUTION_MODE_LIVE_PAPER,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise SettingsError("EXECUTION_MODE must be alpaca_paper or live_paper.") from error


def _validate_settings(settings: Settings) -> None:
    if settings.symbol != "SPY":
        raise SettingsError("This beginner bot is locked to SYMBOL=SPY.")

    if settings.fast_ma_period <= 0 or settings.slow_ma_period <= 0:
        raise SettingsError("Moving-average periods must be positive integers.")

    if settings.fast_ma_period >= settings.slow_ma_period:
        raise SettingsError("FAST_MA_PERIOD must be smaller than SLOW_MA_PERIOD.")

    if not 0 < settings.core_tactical_core_fraction < 1:
        raise SettingsError("CORE_TACTICAL_CORE_FRACTION must be greater than 0 and less than 1.")

    if settings.rebound_ma_period <= 1:
        raise SettingsError("REBOUND_MA_PERIOD must be greater than 1.")

    if settings.rebound_return_days < 1:
        raise SettingsError("REBOUND_RETURN_DAYS must be at least 1.")

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

    if settings.live_paper_initial_cash <= 0:
        raise SettingsError("LIVE_PAPER_INITIAL_CASH must be greater than 0.")

    if settings.lookback_days < 60:
        raise SettingsError("LOOKBACK_DAYS must be at least 60 for the 50-period average.")

    if settings.min_order_notional <= 0:
        raise SettingsError("MIN_ORDER_NOTIONAL must be greater than 0.")

    if len(settings.client_order_id_prefix) > 20:
        raise SettingsError("CLIENT_ORDER_ID_PREFIX must be 20 characters or fewer.")

