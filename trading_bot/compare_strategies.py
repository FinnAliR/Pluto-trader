"""Compare multiple SPY strategies with one historical-data request."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from backtester import (
    BacktestResult,
    plot_equity_curves,
    print_summary_table,
    results_to_summary_frame,
    run_backtest,
    save_summary_csv,
)
from config import SettingsError, load_settings
from historical_data import HistoricalDataError, fetch_daily_bars, parse_date
from strategies.registry import available_strategy_names, create_strategy, normalize_strategy_name


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INITIAL_CASH = 10_000.00
DEFAULT_EXPOSURE = 0.25
RESULTS_FILE_NAME = "backtest_results.csv"
PLOT_FILE_NAME = "backtest_equity_curves.png"


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings()
    except SettingsError as error:
        print(f"Configuration error: {error}")
        return 1

    strategy_names = [normalize_strategy_name(name) for name in args.strategies]
    start_date = parse_date(args.start)
    end_date = parse_date(args.end) if args.end else None

    print("Running data-only strategy comparison. No live trading connection is used.")
    print(f"Symbol: {settings.symbol}")
    print(f"Strategies: {', '.join(strategy_names)}")
    print(f"Exposure when long: {args.exposure:.0%}")
    print(f"Costs: transaction={args.transaction_cost_bps:.2f} bps, slippage={args.slippage_bps:.2f} bps")
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

    full_results = run_strategy_suite(
        strategy_names=strategy_names,
        settings=settings,
        price_data=price_data,
        initial_cash=args.initial_cash,
        exposure=args.exposure,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )
    full_summary = results_to_summary_frame(full_results)
    print_summary_table(full_summary, "Full-period comparison")

    output_dir = Path(__file__).resolve().parent
    results_path = output_dir / RESULTS_FILE_NAME
    plot_path = output_dir / PLOT_FILE_NAME
    save_summary_csv(full_summary, results_path)
    plot_equity_curves(full_results, path=plot_path, show_plot=args.show and not args.no_show)

    print(f"Saved comparison CSV to: {results_path}")
    print(f"Saved equity-curve plot to: {plot_path}")
    print()

    if args.split_date:
        run_train_test_split(
            split_date_text=args.split_date,
            strategy_names=strategy_names,
            settings=settings,
            price_data=price_data,
            initial_cash=args.initial_cash,
            exposure=args.exposure,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
        )

    return 0


def run_strategy_suite(
    strategy_names: list[str],
    settings,
    price_data,
    initial_cash: float,
    exposure: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> list[BacktestResult]:
    results = []

    for strategy_name in strategy_names:
        strategy = create_strategy(strategy_name, settings=settings)
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


def run_train_test_split(
    split_date_text: str,
    strategy_names: list[str],
    settings,
    price_data,
    initial_cash: float,
    exposure: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> None:
    split_date = parse_date(split_date_text)
    train_data = price_data.loc[price_data.index <= split_date]
    test_data = price_data.loc[price_data.index > split_date]

    if train_data.empty or test_data.empty:
        print("Train/test split skipped: split date leaves one side empty.")
        return

    train_results = run_strategy_suite(
        strategy_names=strategy_names,
        settings=settings,
        price_data=train_data,
        initial_cash=initial_cash,
        exposure=exposure,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    test_results = run_strategy_suite(
        strategy_names=strategy_names,
        settings=settings,
        price_data=test_data,
        initial_cash=initial_cash,
        exposure=exposure,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

    print_summary_table(results_to_summary_frame(train_results), f"Train period ending {split_date.date()}")
    print_summary_table(results_to_summary_frame(test_results), f"Test period after {split_date.date()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple SPY strategies using Alpaca daily historical data.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=available_strategy_names(),
        help="Strategies to compare. Choices: ma_crossover core_tactical_ma rebound_reentry_ma rsi breakout buy_and_hold.",
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
        "--split-date",
        help="Optional train/test split date in YYYY-MM-DD format.",
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
