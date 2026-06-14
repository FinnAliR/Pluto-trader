"""Auto-discover strategy classes and editable constructor parameters."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .base import Strategy


@dataclass(frozen=True)
class StrategyParameterSpec:
    """Editable parameter metadata for one strategy constructor argument."""

    name: str
    label: str
    value_type: type
    default: Any
    minimum: int | float | None = None
    maximum: int | float | None = None
    description: str = ""


BUILTIN_ORDER = [
    "ma_crossover",
    "core_tactical_ma",
    "rebound_reentry_ma",
    "rsi",
    "breakout",
    "candle_pattern_jpy_session",
    "buy_and_hold",
]

BUILTIN_MODULE_ORDER = {
    "ma_crossover": 0,
    "core_tactical_ma": 1,
    "rebound_reentry_ma": 2,
    "rsi_strategy": 3,
    "breakout": 4,
    "candle_pattern_jpy_session": 5,
    "buy_and_hold": 6,
}

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

PARAMETER_METADATA: dict[str, dict[str, dict[str, Any]]] = {
    "ma_crossover": {
        "fast_period": {
            "label": "Fast MA",
            "minimum": 1,
            "description": "Fast moving-average lookback in trading days.",
            "settings_attr": "fast_ma_period",
        },
        "slow_period": {
            "label": "Slow MA",
            "minimum": 2,
            "description": "Slow moving-average lookback in trading days.",
            "settings_attr": "slow_ma_period",
        },
    },
    "core_tactical_ma": {
        "fast_period": {
            "label": "Fast MA",
            "minimum": 1,
            "description": "Fast moving-average lookback in trading days.",
            "settings_attr": "fast_ma_period",
        },
        "slow_period": {
            "label": "Slow MA",
            "minimum": 2,
            "description": "Slow moving-average lookback in trading days.",
            "settings_attr": "slow_ma_period",
        },
        "core_fraction": {
            "label": "Core fraction",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Always-on fraction of configured exposure.",
            "settings_attr": "core_tactical_core_fraction",
        },
    },
    "rebound_reentry_ma": {
        "fast_period": {
            "label": "Fast MA",
            "minimum": 1,
            "description": "Fast moving-average lookback in trading days.",
            "settings_attr": "fast_ma_period",
        },
        "slow_period": {
            "label": "Slow MA",
            "minimum": 2,
            "description": "Slow moving-average lookback in trading days.",
            "settings_attr": "slow_ma_period",
        },
        "rebound_ma_period": {
            "label": "Rebound MA",
            "minimum": 1,
            "description": "Moving average used to confirm rebound re-entry.",
            "settings_attr": "rebound_ma_period",
        },
        "rebound_return_days": {
            "label": "Rebound days",
            "minimum": 1,
            "description": "Lookback days for the rebound return test.",
            "settings_attr": "rebound_return_days",
        },
        "rebound_min_return": {
            "label": "Min rebound",
            "minimum": -1.0,
            "description": "Minimum return over rebound days; 0.02 means 2%.",
            "settings_attr": "rebound_min_return",
        },
    },
    "rsi": {
        "period": {
            "label": "RSI period",
            "minimum": 1,
            "description": "RSI lookback period in trading days.",
            "settings_attr": "rsi_period",
        },
        "buy_level": {
            "label": "Buy level",
            "minimum": 0.0,
            "maximum": 100.0,
            "description": "RSI threshold below which the strategy buys.",
            "settings_attr": "rsi_buy_level",
        },
        "sell_level": {
            "label": "Sell level",
            "minimum": 0.0,
            "maximum": 100.0,
            "description": "RSI threshold above which the strategy sells.",
            "settings_attr": "rsi_sell_level",
        },
    },
    "breakout": {
        "entry_period": {
            "label": "Entry period",
            "minimum": 1,
            "description": "Prior high lookback period for breakout entries.",
            "settings_attr": "breakout_entry_period",
        },
        "exit_period": {
            "label": "Exit period",
            "minimum": 1,
            "description": "Prior low lookback period for breakout exits.",
            "settings_attr": "breakout_exit_period",
        },
    },
    "candle_pattern_jpy_session": {
        "return_threshold": {
            "label": "Return threshold",
            "minimum": 0.0,
            "description": "Minimum candle-to-candle return needed to confirm a reversal.",
        },
        "max_hold_bars": {
            "label": "Max hold bars",
            "minimum": 1,
            "description": "Maximum bars to hold after a bullish candle signal.",
        },
        "profit_take_return": {
            "label": "Profit take",
            "minimum": -1.0,
            "description": "Exit once unrealized return exceeds this value; 0 means any profit.",
        },
    },
}

_DISCOVERY_ERRORS: dict[str, Exception] = {}


def available_strategy_names() -> list[str]:
    """Return canonical strategy names discovered in the strategies package."""
    names = list(discover_strategy_classes())
    builtin = [name for name in BUILTIN_ORDER if name in names]
    custom = sorted(name for name in names if name not in BUILTIN_ORDER)
    return builtin + custom


def discover_strategy_classes() -> dict[str, type[Strategy]]:
    """Import strategy modules and return every concrete Strategy subclass."""
    _DISCOVERY_ERRORS.clear()
    strategy_classes: dict[str, type[Strategy]] = {}

    package_name = __package__ or "trading_bot.strategies"
    package = importlib.import_module(package_name)
    module_infos = [
        module_info
        for module_info in pkgutil.iter_modules(package.__path__)
        if not module_info.ispkg
        and not module_info.name.startswith("_")
        and module_info.name not in {"base", "registry"}
    ]
    module_infos.sort(key=lambda module_info: (BUILTIN_MODULE_ORDER.get(module_info.name, 999), module_info.name))

    for module_info in module_infos:
        module_name = f"{package_name}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - keep registry usable when a custom strategy is broken.
            _DISCOVERY_ERRORS[module_name] = error
            continue

        for _class_name, strategy_class in inspect.getmembers(module, inspect.isclass):
            if strategy_class.__module__ != module.__name__:
                continue
            if strategy_class is Strategy or not issubclass(strategy_class, Strategy) or inspect.isabstract(strategy_class):
                continue

            strategy_name = _strategy_name_for_class(strategy_class)
            existing = strategy_classes.get(strategy_name)
            if existing is not None and existing is not strategy_class:
                _DISCOVERY_ERRORS[f"{module_name}.{strategy_class.__name__}"] = ValueError(
                    f"Duplicate strategy name '{strategy_name}'. "
                    f"Already loaded from {existing.__module__}.{existing.__name__}."
                )
                continue
            strategy_classes[strategy_name] = strategy_class

    return strategy_classes


def strategy_discovery_errors() -> dict[str, Exception]:
    """Return import errors found while scanning custom strategy modules."""
    discover_strategy_classes()
    return dict(_DISCOVERY_ERRORS)


def normalize_strategy_name(name: str) -> str:
    """Normalize friendly aliases into canonical strategy names."""
    normalized = name.strip().lower().replace("-", "_")
    strategy_classes = discover_strategy_classes()
    aliased = ALIASES.get(normalized, normalized)

    if aliased not in strategy_classes:
        allowed = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Choose one of: {allowed}.")

    return aliased


def strategy_parameter_specs(name: str, settings=None) -> list[StrategyParameterSpec]:
    """Return editable parameter specs for a strategy."""
    normalized = normalize_strategy_name(name)
    strategy_class = discover_strategy_classes()[normalized]

    custom_specs = getattr(strategy_class, "parameter_specs", None)
    if callable(custom_specs):
        return list(custom_specs(settings=settings))

    return _parameter_specs_from_signature(normalized, strategy_class, settings=settings)


def strategy_default_params(name: str, settings=None) -> dict[str, Any]:
    """Return default editable parameters for a strategy."""
    return {spec.name: spec.default for spec in strategy_parameter_specs(name, settings=settings)}


def normalize_strategy_params(
    name: str,
    params: Mapping[str, Any] | None = None,
    settings=None,
) -> dict[str, Any]:
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
    """Create a discovered strategy by name with validated parameters."""
    normalized = normalize_strategy_name(name)
    strategy_class = discover_strategy_classes()[normalized]
    strategy_params = normalize_strategy_params(normalized, params=params, settings=settings)
    return strategy_class(**strategy_params)


def _parameter_specs_from_signature(
    strategy_name: str,
    strategy_class: type[Strategy],
    settings=None,
) -> list[StrategyParameterSpec]:
    signature = inspect.signature(strategy_class.__init__)
    metadata_by_name = _metadata_for(strategy_name, strategy_class)
    specs: list[StrategyParameterSpec] = []

    for param_name, parameter in signature.parameters.items():
        if param_name == "self":
            continue
        if parameter.kind not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
            continue
        if parameter.default is inspect.Parameter.empty:
            # Required constructor arguments cannot be safely exposed in the GUI.
            continue

        metadata = metadata_by_name.get(param_name, {})
        default = parameter.default
        settings_attr = metadata.get("settings_attr")
        if settings is not None and settings_attr:
            default = getattr(settings, settings_attr, default)

        value_type = metadata.get("value_type") or _value_type_for(parameter.annotation, default)
        specs.append(
            StrategyParameterSpec(
                name=param_name,
                label=metadata.get("label", _humanize_name(param_name)),
                value_type=value_type,
                default=default,
                minimum=metadata.get("minimum"),
                maximum=metadata.get("maximum"),
                description=metadata.get("description", ""),
            )
        )

    return specs


def _metadata_for(strategy_name: str, strategy_class: type[Strategy]) -> dict[str, dict[str, Any]]:
    metadata = dict(PARAMETER_METADATA.get(strategy_name, {}))
    class_metadata = getattr(strategy_class, "parameter_metadata", None)
    if isinstance(class_metadata, Mapping):
        for param_name, values in class_metadata.items():
            merged = dict(metadata.get(param_name, {}))
            merged.update(dict(values))
            metadata[param_name] = merged
    return metadata


def _strategy_name_for_class(strategy_class: type[Strategy]) -> str:
    explicit_name = getattr(strategy_class, "name", None)
    if isinstance(explicit_name, str) and explicit_name.strip():
        return explicit_name.strip().lower().replace("-", "_")
    return _snake_case(strategy_class.__name__)


def _value_type_for(annotation: Any, default: Any) -> type:
    if annotation in {int, float, bool, str}:
        return annotation
    if isinstance(default, bool):
        return bool
    if isinstance(default, int) and not isinstance(default, bool):
        return int
    if isinstance(default, float):
        return float
    if isinstance(default, str):
        return str
    return str


def _coerce_strategy_param(spec: StrategyParameterSpec, raw_value: Any) -> Any:
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
    elif spec.value_type is bool:
        value = _coerce_bool(spec, raw_value)
    elif spec.value_type is str:
        value = str(raw_value)
    else:
        raise TypeError(f"Unsupported parameter type for {spec.name}: {spec.value_type!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{spec.label} must be at least {spec.minimum}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{spec.label} must be no more than {spec.maximum}.")

    return value


def _coerce_bool(spec: StrategyParameterSpec, raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{spec.label} must be true or false.")


def _humanize_name(name: str) -> str:
    return name.replace("_", " ").title()


def _snake_case(name: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower().removesuffix("_strategy")
