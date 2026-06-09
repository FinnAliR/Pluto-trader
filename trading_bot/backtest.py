"""Backtest one SPY strategy with historical daily data.

This script uses Alpaca market data only. It does not import TradingClient,
does not check positions, and cannot place orders.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from backtester import plot_equity_curves, print_summary_table, results_to_summary_frame, run_backtest
from config import SettingsError, load_settings
from historical_data import HistoricalDataError, fetch_daily_bars, parse_date
from strategies.registry import create_strategy, normalize_strategy_name


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INITIAL_CASH = 10_000.00
DEFAULT_EXPOSURE = 0.25
PLOT_FILE_NAME = "backtest_portfolio_value.png"


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings()
    except SettingsError as error:
        print(f"Configuration error: {error}")
        return 1

    strategy_name = normalize_strategy_name(args.strategy)
    strategy = create_strategy(strategy_name, settings=settings)
    start_date = parse_date(args.start)
    end_date = parse_date(args.end) if args.end else None

    print("Running data-only backtest. No live trading connection is used.")
    print(f"Symbol: {settings.symbol}")
    print(f"Strategy: {strategy.display_name}")
    print(f"Exposure when long: {args.exposure:.0%}")
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date() if end_date else 'latest available'}")
    print()

    try:
        price_data = fetch_daily_bars(settings=settings, start=start_date, end=end_date)
    except (HistoricalDataError, ValueError) as error:
        print(f"Historical data error: {error}")
        return 1

    if price_data.empty:
        print("No historical price data was returned. Check your Alpaca keys and data feed.")
        return 1

    result = run_backtest(
        strategy=strategy,
        price_data=price_data,
        initial_cash=args.initial_cash,
        exposure=args.exposure,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )
    summary = results_to_summary_frame([result])
    print_summary_table(summary, "Backtest results")

    plot_path = Path(__file__).resolve().parent / PLOT_FILE_NAME
    plot_equity_curves([result], path=plot_path, show_plot=not args.no_show)
    print(f"Saved plot to: {plot_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest one SPY strategy using Alpaca daily historical data.")
    parser.add_argument(
        "--strategy",
        default=os.getenv("STRATEGY", "ma_crossover"),
        help="Strategy to test: ma_crossover, rsi, breakout, buy_and_hold.",
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
        default=DEFAULT_EXPOSURE,
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
        "--no-show",
        action="store_true",
        help="Save the plot without opening a chart window.",
    )
    return parser.parse_args()
#test

if __name__ == "__main__":
    raise SystemExit(main())
