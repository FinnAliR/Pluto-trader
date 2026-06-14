"""Compare multiple strategies with one historical-data request."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading_bot.backtesting.engine import (
    plot_equity_curves,
    print_summary_table,
    save_summary_csv,
)
from trading_bot.core.config import SettingsError
from trading_bot.market_data.history import HistoricalDataError
from trading_bot.services.backtest_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    StrategyComparisonRequest,
    run_strategy_comparison,
)
from trading_bot.strategies.registry import available_strategy_names


RESULTS_FILE_NAME = "backtest_results.csv"
PLOT_FILE_NAME = "backtest_equity_curves.png"


def main() -> int:
    args = parse_args()

    try:
        comparison = run_strategy_comparison(
            StrategyComparisonRequest(
                strategy_names=args.strategies,
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                initial_cash=args.initial_cash,
                exposure=args.exposure,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                split_date=args.split_date,
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

    print("Running data-only strategy comparison. No live trading connection is used.")
    print(f"Symbol: {comparison.symbol}")
    print(f"Strategies: {', '.join(comparison.strategy_names)}")
    print(f"Exposure when long: {args.exposure:.0%}")
    print(f"Costs: transaction={args.transaction_cost_bps:.2f} bps, slippage={args.slippage_bps:.2f} bps")
    print(f"Start date: {comparison.start_date.date()}")
    print(f"End date: {comparison.end_date.date() if comparison.end_date else 'latest available'}")
    print()

    print_summary_table(comparison.full_summary, "Full-period comparison")

    output_dir = Path(__file__).resolve().parents[1]
    results_path = output_dir / RESULTS_FILE_NAME
    plot_path = output_dir / PLOT_FILE_NAME
    save_summary_csv(comparison.full_summary, results_path)
    plot_equity_curves(
        comparison.full_results,
        path=plot_path,
        show_plot=args.show and not args.no_show,
    )

    print(f"Saved comparison CSV to: {results_path}")
    print(f"Saved equity-curve plot to: {plot_path}")
    print()

    if comparison.split_date:
        if comparison.train_summary is None or comparison.test_summary is None:
            print("Train/test split skipped: split date leaves one side empty.")
        else:
            print_summary_table(
                comparison.train_summary,
                f"Train period ending {comparison.split_date.date()}",
            )
            print_summary_table(
                comparison.test_summary,
                f"Test period after {comparison.split_date.date()}",
            )

    return 0


def parse_args() -> argparse.Namespace:
    available_strategies = ", ".join(available_strategy_names())
    parser = argparse.ArgumentParser(description="Compare multiple strategies using Alpaca daily historical data.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=available_strategy_names(),
        help=f"Strategies to compare. Auto-discovered strategies now available: {available_strategies}.",
    )
    parser.add_argument(
        "--symbol",
        default=os.getenv("BACKTEST_SYMBOL", "SPY"),
        help="Market symbol to compare strategies on. Default: SPY.",
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

