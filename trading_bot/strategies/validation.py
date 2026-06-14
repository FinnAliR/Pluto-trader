"""Validation helpers for discovered strategies."""

from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import Any

import pandas as pd

from trading_bot.strategies.base import StrategyDecision, TradeAction
from trading_bot.strategies.registry import (
    available_strategy_names,
    create_strategy,
    normalize_strategy_name,
    strategy_discovery_errors,
    strategy_parameter_specs,
)


SUPPORTED_PARAMETER_TYPES = {int, float, bool, str}


@dataclass(frozen=True)
class StrategyValidationResult:
    """One strategy validation outcome."""

    strategy: str
    display_name: str
    status: str
    editable_parameters: str
    checks: str
    warnings: str
    errors: str


def validate_all_strategies(sample_data: pd.DataFrame | None = None) -> list[StrategyValidationResult]:
    """Validate discovery, editable params, signals, and live decisions."""
    data = sample_data if sample_data is not None else build_sample_price_data()
    results = [
        StrategyValidationResult(
            strategy=module_name,
            display_name="Discovery error",
            status="fail",
            editable_parameters="",
            checks="discovery",
            warnings="",
            errors=f"{type(error).__name__}: {error}",
        )
        for module_name, error in sorted(strategy_discovery_errors().items())
    ]

    for strategy_name in available_strategy_names():
        results.append(validate_strategy(strategy_name, data))

    return results


def validate_strategy(strategy_name: str, sample_data: pd.DataFrame) -> StrategyValidationResult:
    """Validate one strategy using deterministic sample OHLC data."""
    checks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    display_name = ""
    parameter_text = ""

    try:
        normalized_name = normalize_strategy_name(strategy_name)
    except Exception as error:  # noqa: BLE001 - validation should report every failure.
        return _result(
            strategy=strategy_name,
            display_name="",
            checks=checks,
            warnings=warnings,
            errors=[f"Name normalization failed: {type(error).__name__}: {error}"],
            editable_parameters=parameter_text,
        )

    try:
        specs = strategy_parameter_specs(normalized_name)
        parameter_text = ", ".join(spec.name for spec in specs) if specs else "none"
        checks.append("params")
        for spec in specs:
            if spec.value_type not in SUPPORTED_PARAMETER_TYPES:
                errors.append(f"{spec.name} uses unsupported parameter type {spec.value_type!r}.")
            if spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum:
                errors.append(f"{spec.name} has minimum greater than maximum.")
    except Exception as error:  # noqa: BLE001
        errors.append(f"Parameter introspection failed: {type(error).__name__}: {error}")

    try:
        strategy = create_strategy(normalized_name)
        checks.append("construct")
    except Exception as error:  # noqa: BLE001
        return _result(
            strategy=normalized_name,
            display_name=display_name,
            checks=checks,
            warnings=warnings,
            errors=errors + [f"Construction failed: {type(error).__name__}: {error}"],
            editable_parameters=parameter_text,
        )

    display_name = str(getattr(strategy, "display_name", "") or "")
    actual_name = getattr(strategy, "name", None)
    if not isinstance(actual_name, str) or not actual_name.strip():
        errors.append("Strategy instance must expose a non-empty string name.")
    elif actual_name != normalized_name:
        errors.append(f"Strategy instance name is {actual_name!r}, expected {normalized_name!r}.")

    if not display_name.strip():
        errors.append("Strategy instance must expose a non-empty display_name.")

    try:
        signals = strategy.generate_signals(sample_data.copy())
        checks.append("signals")
        _validate_signals(signals, sample_data, errors, warnings)
    except Exception as error:  # noqa: BLE001
        errors.append(f"generate_signals failed: {type(error).__name__}: {error}")

    try:
        decision = strategy.make_decision(sample_data.copy(), owns_position=False, logger=None)
        checks.append("decision")
        if not isinstance(decision, StrategyDecision):
            errors.append("make_decision must return StrategyDecision.")
        elif decision.action not in set(TradeAction):
            errors.append(f"make_decision returned unsupported action {decision.action!r}.")
    except Exception as error:  # noqa: BLE001
        errors.append(f"make_decision failed: {type(error).__name__}: {error}")

    return _result(
        strategy=normalized_name,
        display_name=display_name,
        checks=checks,
        warnings=warnings,
        errors=errors,
        editable_parameters=parameter_text,
    )


def validation_results_to_frame(results: list[StrategyValidationResult]) -> pd.DataFrame:
    """Convert validation results to a GUI/CSV-friendly table."""
    return pd.DataFrame(
        [
            {
                "status": result.status,
                "strategy": result.strategy,
                "display_name": result.display_name,
                "editable_parameters": result.editable_parameters,
                "checks": result.checks,
                "warnings": result.warnings,
                "errors": result.errors,
            }
            for result in results
        ]
    )


def build_sample_price_data(rows: int = 90) -> pd.DataFrame:
    """Create deterministic OHLC data for strategy validation."""
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    close_values = [
        100.0 + (position * 0.12) + (sin(position / 4.0) * 2.5)
        for position in range(rows)
    ]
    close = pd.Series(close_values, index=index, dtype=float)
    open_price = close.shift(1).fillna(close.iloc[0]) * 0.999
    high = pd.concat([open_price, close], axis=1).max(axis=1) * 1.01
    low = pd.concat([open_price, close], axis=1).min(axis=1) * 0.99
    volume = pd.Series(1_000_000, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _validate_signals(
    signals: Any,
    sample_data: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(signals, pd.Series):
        errors.append("generate_signals must return a pandas Series.")
        return

    if signals.empty:
        errors.append("generate_signals returned an empty Series for valid sample data.")
        return

    overlap = signals.index.intersection(sample_data.index)
    if overlap.empty:
        errors.append("generate_signals index does not overlap the input price-data index.")

    if not signals.index.equals(sample_data.index):
        warnings.append("generate_signals index differs from the input index; the backtester will reindex it.")

    numeric_signals = pd.to_numeric(signals, errors="coerce")
    if numeric_signals.isna().any():
        errors.append("generate_signals returned non-numeric or NaN signal values.")
        return

    if (numeric_signals < 0).any() or (numeric_signals > 1).any():
        errors.append("generate_signals values must stay between 0.0 and 1.0.")


def _result(
    strategy: str,
    display_name: str,
    checks: list[str],
    warnings: list[str],
    errors: list[str],
    editable_parameters: str,
) -> StrategyValidationResult:
    status = "fail" if errors else "warn" if warnings else "pass"
    return StrategyValidationResult(
        strategy=strategy,
        display_name=display_name,
        status=status,
        editable_parameters=editable_parameters,
        checks=", ".join(checks),
        warnings="; ".join(warnings),
        errors="; ".join(errors),
    )
