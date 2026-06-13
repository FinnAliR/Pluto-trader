"""Trade-analysis workflow shared by the CLI and GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_bot.backtesting.engine import BacktestResult, run_backtest
from trading_bot.core.config import Settings, load_settings
from trading_bot.market_data.history import HistoricalDataError, fetch_daily_bars, parse_date
from trading_bot.services.backtest_service import normalize_symbol
from trading_bot.strategies.registry import create_strategy, normalize_strategy_name
from trading_bot.diagnostics.reports import (
    build_trade_report,
    find_missed_best_days,
    find_suspicious_moves,
    find_whipsaws,
)


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INITIAL_CASH = 10_000.00
DEFAULT_EXPOSURE = 0.25
DEFAULT_TOP_DAYS = 20
DEFAULT_SHORT_TRADE_DAYS = 30
DEFAULT_SUSPICIOUS_MOVE_THRESHOLD = 0.10


@dataclass(frozen=True)
class TradeAnalysisRequest:
    """Inputs for one strategy trade-diagnostic run."""

    strategy_name: str = "ma_crossover"
    symbol: str = "SPY"
    start: str = DEFAULT_START_DATE
    end: str | None = None
    initial_cash: float = DEFAULT_INITIAL_CASH
    exposure: float = DEFAULT_EXPOSURE
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    top_days: int = DEFAULT_TOP_DAYS
    short_trade_days: int = DEFAULT_SHORT_TRADE_DAYS
    suspicious_move_threshold: float = DEFAULT_SUSPICIOUS_MOVE_THRESHOLD
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeAnalysisResult:
    """Outputs from a trade-diagnostic run."""

    settings: Settings
    symbol: str
    strategy_name: str
    result: BacktestResult
    trades: pd.DataFrame
    missed_days: pd.DataFrame
    whipsaws: pd.DataFrame
    suspicious_moves: pd.DataFrame


def run_trade_analysis(
    request: TradeAnalysisRequest,
    settings: Settings | None = None,
    price_data: pd.DataFrame | None = None,
) -> TradeAnalysisResult:
    """Run a strategy backtest and build diagnostic report tables."""
    loaded_settings = settings or load_settings()
    strategy_name = normalize_strategy_name(request.strategy_name)
    symbol = normalize_symbol(request.symbol)
    start_date = parse_date(request.start)
    end_date = parse_date(request.end) if request.end else None

    data = price_data
    if data is None:
        data = fetch_daily_bars(settings=loaded_settings, start=start_date, end=end_date, symbol=symbol)

    if data.empty:
        raise HistoricalDataError(f"No historical price data was returned for {symbol}. Check the symbol, Alpaca keys, and data feed.")

    strategy = create_strategy(strategy_name, settings=loaded_settings, params=request.strategy_params)
    result = run_backtest(
        strategy=strategy,
        price_data=data,
        initial_cash=request.initial_cash,
        exposure=request.exposure,
        transaction_cost_bps=request.transaction_cost_bps,
        slippage_bps=request.slippage_bps,
    )

    trades = build_trade_report(result.data)
    missed_days = find_missed_best_days(result.data, top_days=request.top_days)
    whipsaws = find_whipsaws(trades=trades, max_days_held=request.short_trade_days)
    suspicious_moves = find_suspicious_moves(
        data=result.data,
        threshold=request.suspicious_move_threshold,
    )

    return TradeAnalysisResult(
        settings=loaded_settings,
        symbol=symbol,
        strategy_name=strategy_name,
        result=result,
        trades=trades,
        missed_days=missed_days,
        whipsaws=whipsaws,
        suspicious_moves=suspicious_moves,
    )
