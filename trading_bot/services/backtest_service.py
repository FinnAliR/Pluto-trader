"""Backtest workflows shared by command-line scripts and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trading_bot.backtesting.engine import BacktestResult, results_to_summary_frame, run_backtest
from trading_bot.core.config import Settings, load_settings
from trading_bot.market_data.history import HistoricalDataError, fetch_daily_bars, parse_date
from trading_bot.strategies.registry import available_strategy_names, create_strategy, normalize_strategy_name


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INITIAL_CASH = 10_000.00
DEFAULT_EXPOSURE = 0.25


@dataclass(frozen=True)
class BacktestRequest:
    """Inputs for one strategy backtest."""

    strategy_name: str = "ma_crossover"
    symbol: str = "SPY"
    start: str = DEFAULT_START_DATE
    end: str | None = None
    initial_cash: float = DEFAULT_INITIAL_CASH
    exposure: float = DEFAULT_EXPOSURE
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestRunResult:
    """Outputs from one strategy backtest."""

    settings: Settings
    symbol: str
    strategy_name: str
    start_date: datetime
    end_date: datetime | None
    price_data: pd.DataFrame
    result: BacktestResult
    summary: pd.DataFrame


@dataclass(frozen=True)
class StrategyComparisonRequest:
    """Inputs for comparing several strategies over the same data."""

    strategy_names: list[str] = field(default_factory=available_strategy_names)
    symbol: str = "SPY"
    start: str = DEFAULT_START_DATE
    end: str | None = None
    initial_cash: float = DEFAULT_INITIAL_CASH
    exposure: float = DEFAULT_EXPOSURE
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    split_date: str | None = None
    strategy_params_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyComparisonResult:
    """Outputs from a strategy comparison run."""

    settings: Settings
    symbol: str
    strategy_names: list[str]
    start_date: datetime
    end_date: datetime | None
    price_data: pd.DataFrame
    full_results: list[BacktestResult]
    full_summary: pd.DataFrame
    split_date: datetime | None = None
    train_results: list[BacktestResult] | None = None
    train_summary: pd.DataFrame | None = None
    test_results: list[BacktestResult] | None = None
    test_summary: pd.DataFrame | None = None


def run_single_backtest(
    request: BacktestRequest,
    settings: Settings | None = None,
    price_data: pd.DataFrame | None = None,
) -> BacktestRunResult:
    """Run one backtest and return its data and summary table."""
    loaded_settings = settings or load_settings()
    strategy_name = normalize_strategy_name(request.strategy_name)
    symbol = normalize_symbol(request.symbol)
    start_date = parse_date(request.start)
    end_date = parse_date(request.end) if request.end else None
    data = price_data if price_data is not None else load_daily_price_data(loaded_settings, start_date, end_date, symbol)

    strategy = create_strategy(strategy_name, settings=loaded_settings, params=request.strategy_params)
    result = run_backtest(
        strategy=strategy,
        price_data=data,
        initial_cash=request.initial_cash,
        exposure=request.exposure,
        transaction_cost_bps=request.transaction_cost_bps,
        slippage_bps=request.slippage_bps,
    )
    summary = results_to_summary_frame([result])

    return BacktestRunResult(
        settings=loaded_settings,
        symbol=symbol,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        price_data=data,
        result=result,
        summary=summary,
    )


def run_strategy_comparison(
    request: StrategyComparisonRequest,
    settings: Settings | None = None,
    price_data: pd.DataFrame | None = None,
) -> StrategyComparisonResult:
    """Compare several strategies against one historical data set."""
    loaded_settings = settings or load_settings()
    strategy_names = [normalize_strategy_name(name) for name in request.strategy_names]
    symbol = normalize_symbol(request.symbol)
    strategy_params_by_name = {
        normalize_strategy_name(name): params
        for name, params in request.strategy_params_by_name.items()
    }
    start_date = parse_date(request.start)
    end_date = parse_date(request.end) if request.end else None
    data = price_data if price_data is not None else load_daily_price_data(loaded_settings, start_date, end_date, symbol)

    full_results = run_strategy_suite(
        strategy_names=strategy_names,
        settings=loaded_settings,
        price_data=data,
        initial_cash=request.initial_cash,
        exposure=request.exposure,
        transaction_cost_bps=request.transaction_cost_bps,
        slippage_bps=request.slippage_bps,
        strategy_params_by_name=strategy_params_by_name,
    )
    full_summary = results_to_summary_frame(full_results)

    split_date = parse_date(request.split_date) if request.split_date else None
    train_results = None
    train_summary = None
    test_results = None
    test_summary = None

    if split_date:
        train_data = data.loc[data.index <= split_date]
        test_data = data.loc[data.index > split_date]

        if not train_data.empty and not test_data.empty:
            train_results = run_strategy_suite(
                strategy_names=strategy_names,
                settings=loaded_settings,
                price_data=train_data,
                initial_cash=request.initial_cash,
                exposure=request.exposure,
                transaction_cost_bps=request.transaction_cost_bps,
                slippage_bps=request.slippage_bps,
                strategy_params_by_name=strategy_params_by_name,
            )
            test_results = run_strategy_suite(
                strategy_names=strategy_names,
                settings=loaded_settings,
                price_data=test_data,
                initial_cash=request.initial_cash,
                exposure=request.exposure,
                transaction_cost_bps=request.transaction_cost_bps,
                slippage_bps=request.slippage_bps,
                strategy_params_by_name=strategy_params_by_name,
            )
            train_summary = results_to_summary_frame(train_results)
            test_summary = results_to_summary_frame(test_results)

    return StrategyComparisonResult(
        settings=loaded_settings,
        symbol=symbol,
        strategy_names=strategy_names,
        start_date=start_date,
        end_date=end_date,
        price_data=data,
        full_results=full_results,
        full_summary=full_summary,
        split_date=split_date,
        train_results=train_results,
        train_summary=train_summary,
        test_results=test_results,
        test_summary=test_summary,
    )


def run_strategy_suite(
    strategy_names: list[str],
    settings: Settings,
    price_data: pd.DataFrame,
    initial_cash: float,
    exposure: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    strategy_params_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[BacktestResult]:
    """Run every strategy in strategy_names over the same price data."""
    results = []
    params_by_name = strategy_params_by_name or {}

    for strategy_name in strategy_names:
        strategy = create_strategy(
            strategy_name,
            settings=settings,
            params=params_by_name.get(strategy_name, {}),
        )
        result = run_backtest(
            strategy=strategy,
            price_data=price_data,
            initial_cash=initial_cash,
            exposure=exposure,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        results.append(result)

    return results


def load_daily_price_data(
    settings: Settings,
    start_date: datetime,
    end_date: datetime | None,
    symbol: str,
) -> pd.DataFrame:
    """Fetch daily bars and fail early if the provider returns no rows."""
    price_data = fetch_daily_bars(settings=settings, start=start_date, end=end_date, symbol=symbol)

    if price_data.empty:
        raise HistoricalDataError(f"No historical price data was returned for {symbol}. Check the symbol, Alpaca keys, and data feed.")

    return price_data


def normalize_symbol(symbol: str | None) -> str:
    normalized = (symbol or "SPY").strip().upper()
    if not normalized:
        raise ValueError("Market symbol is required.")
    if any(character.isspace() for character in normalized):
        raise ValueError("Market symbol cannot contain spaces.")
    return normalized
