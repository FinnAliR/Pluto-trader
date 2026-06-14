"""Backtest one strategy with historical daily data.

This script uses Alpaca market data only. It does not import TradingClient,
does not check positions, and cannot place orders.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading_bot.backtesting.engine import plot_equity_curves, print_summary_table
from trading_bot.core.config import SettingsError
from trading_bot.market_data.history import HistoricalDataError
from trading_bot.services.backtest_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    BacktestRequest,
    run_single_backtest,
)
from trading_bot.strategies.registry import available_strategy_names


PLOT_FILE_NAME = "backtest_portfolio_value.png"


def main() -> int:
    args = parse_args()

    try:
        run_result = run_single_backtest(
            BacktestRequest(
                strategy_name=args.strategy,
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                initial_cash=args.initial_cash,
                exposure=args.exposure,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
            )
        )
    except SettingsError as error:
        print(f"Configuration error: {error}")
        return 1
    except HistoricalDataError as error:
        print(f"Historical data error: {error}")
        return 1
    except ValueError as error:
        print(f"Input error: {error}")
        return 1

    print("Running data-only backtest. No live trading connection is used.")
    print(f"Symbol: {run_result.symbol}")
    print(f"Strategy: {run_result.result.display_name}")
    print(f"Exposure when long: {args.exposure:.0%}")
    print(f"Start date: {run_result.start_date.date()}")
    print(f"End date: {run_result.end_date.date() if run_result.end_date else 'latest available'}")
    print()

    print_summary_table(run_result.summary, "Backtest results")

    plot_path = Path(__file__).resolve().parents[1] / PLOT_FILE_NAME
    plot_equity_curves(
        [run_result.result],
        path=plot_path,
        show_plot=args.show and not args.no_show,
    )
    print(f"Saved plot to: {plot_path}")

    return 0


def parse_args() -> argparse.Namespace:
    available_strategies = ", ".join(available_strategy_names())
    parser = argparse.ArgumentParser(description="Backtest one strategy using Alpaca daily historical data.")
    parser.add_argument(
        "--strategy",
        default=os.getenv("STRATEGY", "ma_crossover"),
        help=f"Strategy to test. Auto-discovered strategies now available: {available_strategies}.",
    )
    parser.add_argument(
        "--symbol",
        default=os.getenv("BACKTEST_SYMBOL", "SPY"),
        help="Market symbol to backtest. Default: SPY.",
    )
    parser.add_argument(
        "--start",
        default=os.getenv("BACKTEST_START_DATE", DEFAULT_START_DATE),
        help=f"Backtest start date in YYYY-MM-DD format. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end",
        default=os.getenv("BACKTEST_END_DATE"),
        help="Optional backtest end date in YYYY-MM-DD format. Default: latest available.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=float(os.getenv("BACKTEST_INITIAL_CASH", DEFAULT_INITIAL_CASH)),
        help=f"Starting portfolio value. Default: {DEFAULT_INITIAL_CASH:.2f}",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=float(os.getenv("BACKTEST_EXPOSURE", DEFAULT_EXPOSURE)),
        help="Portfolio exposure when long. Use 0.25 to match the live bot's 25%% cap.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="Estimated transaction cost in basis points per position change.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Estimated slippage in basis points per position change.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot window after saving it. By default the script only saves the PNG.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()

if __name__ == "__main__":
    raise SystemExit(main())

