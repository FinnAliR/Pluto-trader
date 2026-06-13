"""Strategy registry used by live trading and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .base import Strategy
from .breakout import BreakoutStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .candle_pattern_jpy_session import CandlePatternJpySessionStrategy
from .core_tactical_ma import CoreTacticalMovingAverageStrategy
from .ma_crossover import MovingAverageCrossoverStrategy
from .rebound_reentry_ma import ReboundReentryMovingAverageStrategy
from .rsi_strategy import RsiStrategy


ALIASES = {
    "ma": "ma_crossover",
    "moving_average": "ma_crossover",
    "ma_crossover": "ma_crossover",
    "rsi": "rsi",
    "breakout": "breakout",
    "buy_hold": "buy_and_hold",
    "buy_and_hold": "buy_and_hold",
    "candle": "candle_pattern_jpy_session",
    "candle_pattern": "candle_pattern_jpy_session",
    "candle_pattern_jpy": "candle_pattern_jpy_session",
    "candle_pattern_jpy_session": "candle_pattern_jpy_session",
    "jpy_session": "candle_pattern_jpy_session",
    "core": "core_tactical_ma",
    "core_tactical": "core_tactical_ma",
    "core_tactical_ma": "core_tactical_ma",
    "rebound": "rebound_reentry_ma",
    "rebound_ma": "rebound_reentry_ma",
    "rebound_reentry": "rebound_reentry_ma",
    "rebound_reentry_ma": "rebound_reentry_ma",
}


@dataclass(frozen=True)
class StrategyParameterSpec:
    """Editable parameter metadata for one strategy constructor argument."""

    name: str
    label: str
    value_type: type
    default: int | float
    minimum: int | float | None = None
    maximum: int | float | None = None
    description: str = ""


def available_strategy_names() -> list[str]:
    """Return canonical strategy names."""
    return [
        "ma_crossover",
        "core_tactical_ma",
        "rebound_reentry_ma",
        "rsi",
        "breakout",
        "candle_pattern_jpy_session",
        "buy_and_hold",
    ]


def normalize_strategy_name(name: str) -> str:
    """Normalize friendly aliases into canonical strategy names."""
    normalized = name.strip().lower().replace("-", "_")

    if normalized not in ALIASES:
        allowed = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Choose one of: {allowed}.")

    return ALIASES[normalized]


def strategy_parameter_specs(name: str, settings=None) -> list[StrategyParameterSpec]:
    """Return editable parameter specs for a strategy."""
    normalized = normalize_strategy_name(name)

    if normalized == "ma_crossover":
        return [
            StrategyParameterSpec(
                name="fast_period",
                label="Fast MA",
                value_type=int,
                default=getattr(settings, "fast_ma_period", 20),
                minimum=1,
                description="Fast moving-average lookback in trading days.",
            ),
            StrategyParameterSpec(
                name="slow_period",
                label="Slow MA",
                value_type=int,
                default=getattr(settings, "slow_ma_period", 50),
                minimum=2,
                description="Slow moving-average lookback in trading days.",
            ),
        ]

    if normalized == "core_tactical_ma":
        return [
            StrategyParameterSpec(
                name="fast_period",
                label="Fast MA",
                value_type=int,
                default=getattr(settings, "fast_ma_period", 20),
                minimum=1,
                description="Fast moving-average lookback in trading days.",
            ),
            StrategyParameterSpec(
                name="slow_period",
                label="Slow MA",
                value_type=int,
                default=getattr(settings, "slow_ma_period", 50),
                minimum=2,
                description="Slow moving-average lookback in trading days.",
            ),
            StrategyParameterSpec(
                name="core_fraction",
                label="Core fraction",
                value_type=float,
                default=getattr(settings, "core_tactical_core_fraction", 0.40),
                minimum=0.0,
                maximum=1.0,
                description="Always-on fraction of configured exposure.",
            ),
        ]

    if normalized == "rebound_reentry_ma":
        return [
            StrategyParameterSpec(
                name="fast_period",
                label="Fast MA",
                value_type=int,
                default=getattr(settings, "fast_ma_period", 20),
                minimum=1,
                description="Fast moving-average lookback in trading days.",
            ),
            StrategyParameterSpec(
                name="slow_period",
                label="Slow MA",
                value_type=int,
                default=getattr(settings, "slow_ma_period", 50),
                minimum=2,
                description="Slow moving-average lookback in trading days.",
            ),
            StrategyParameterSpec(
                name="rebound_ma_period",
                label="Rebound MA",
                value_type=int,
                default=getattr(settings, "rebound_ma_period", 10),
                minimum=1,
                description="Moving average used to confirm rebound re-entry.",
            ),
            StrategyParameterSpec(
                name="rebound_return_days",
                label="Rebound days",
                value_type=int,
                default=getattr(settings, "rebound_return_days", 3),
                minimum=1,
                description="Lookback days for the rebound return test.",
            ),
            StrategyParameterSpec(
                name="rebound_min_return",
                label="Min rebound",
                value_type=float,
                default=getattr(settings, "rebound_min_return", 0.0),
                minimum=-1.0,
                description="Minimum return over rebound days; 0.02 means 2%.",
            ),
        ]

    if normalized == "rsi":
        return [
            StrategyParameterSpec(
                name="period",
                label="RSI period",
                value_type=int,
                default=getattr(settings, "rsi_period", 14),
                minimum=1,
                description="RSI lookback period in trading days.",
            ),
            StrategyParameterSpec(
                name="buy_level",
                label="Buy level",
                value_type=float,
                default=getattr(settings, "rsi_buy_level", 30.0),
                minimum=0.0,
                maximum=100.0,
                description="RSI threshold below which the strategy buys.",
            ),
            StrategyParameterSpec(
                name="sell_level",
                label="Sell level",
                value_type=float,
                default=getattr(settings, "rsi_sell_level", 70.0),
                minimum=0.0,
                maximum=100.0,
                description="RSI threshold above which the strategy sells.",
            ),
        ]

    if normalized == "breakout":
        return [
            StrategyParameterSpec(
                name="entry_period",
                label="Entry period",
                value_type=int,
                default=getattr(settings, "breakout_entry_period", 20),
                minimum=1,
                description="Prior high lookback period for breakout entries.",
            ),
            StrategyParameterSpec(
                name="exit_period",
                label="Exit period",
                value_type=int,
                default=getattr(settings, "breakout_exit_period", 10),
                minimum=1,
                description="Prior low lookback period for breakout exits.",
            ),
        ]

    if normalized == "candle_pattern_jpy_session":
        return [
            StrategyParameterSpec(
                name="return_threshold",
                label="Return threshold",
                value_type=float,
                default=0.005,
                minimum=0.0,
                description="Minimum candle-to-candle return needed to confirm a reversal.",
            ),
            StrategyParameterSpec(
                name="max_hold_bars",
                label="Max hold bars",
                value_type=int,
                default=2,
                minimum=1,
                description="Maximum bars to hold after a bullish candle signal.",
            ),
            StrategyParameterSpec(
                name="profit_take_return",
                label="Profit take",
                value_type=float,
                default=0.0,
                minimum=-1.0,
                description="Exit once unrealized return exceeds this value; 0 means any profit.",
            ),
        ]

    return []


def strategy_default_params(name: str, settings=None) -> dict[str, int | float]:
    """Return default editable parameters for a strategy."""
    return {spec.name: spec.default for spec in strategy_parameter_specs(name, settings=settings)}


def normalize_strategy_params(
    name: str,
    params: Mapping[str, Any] | None = None,
    settings=None,
) -> dict[str, int | float]:
    """Merge defaults with user overrides and validate parameter values."""
    specs = strategy_parameter_specs(name, settings=settings)
    specs_by_name = {spec.name: spec for spec in specs}
    normalized_params = strategy_default_params(name, settings=settings)
    overrides = dict(params or {})

    unknown_params = sorted(set(overrides) - set(specs_by_name))
    if unknown_params:
        allowed = ", ".join(specs_by_name) or "none"
        raise ValueError(
            f"Unknown parameter(s) for {normalize_strategy_name(name)}: {', '.join(unknown_params)}. "
            f"Allowed: {allowed}."
        )

    for param_name, raw_value in overrides.items():
        spec = specs_by_name[param_name]
        normalized_params[param_name] = _coerce_strategy_param(spec, raw_value)

    return normalized_params


def create_strategy(name: str, settings=None, params: Mapping[str, Any] | None = None) -> Strategy:
    """Create a strategy by name.

    The optional settings object is used by the live bot so .env values can
    control strategy parameters. Backtests can omit settings and use defaults.
    """
    normalized = normalize_strategy_name(name)
    strategy_params = normalize_strategy_params(normalized, params=params, settings=settings)

    if normalized == "ma_crossover":
        return MovingAverageCrossoverStrategy(**strategy_params)

    if normalized == "core_tactical_ma":
        return CoreTacticalMovingAverageStrategy(**strategy_params)

    if normalized == "rebound_reentry_ma":
        return ReboundReentryMovingAverageStrategy(**strategy_params)

    if normalized == "rsi":
        return RsiStrategy(**strategy_params)

    if normalized == "breakout":
        return BreakoutStrategy(**strategy_params)

    if normalized == "candle_pattern_jpy_session":
        return CandlePatternJpySessionStrategy(**strategy_params)

    if normalized == "buy_and_hold":
        return BuyAndHoldStrategy()

    raise ValueError(f"Strategy '{name}' is not implemented.")


def _coerce_strategy_param(spec: StrategyParameterSpec, raw_value: Any) -> int | float:
    if raw_value is None or raw_value == "":
        value = spec.default
    elif spec.value_type is int:
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{spec.label} must be an integer.") from error
        if not number.is_integer():
            raise ValueError(f"{spec.label} must be an integer.")
        value = int(number)
    elif spec.value_type is float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{spec.label} must be a number.") from error
    else:
        raise TypeError(f"Unsupported parameter type for {spec.name}: {spec.value_type!r}")

    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"{spec.label} must be at least {spec.minimum}.")

    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"{spec.label} must be no more than {spec.maximum}.")

    return value
