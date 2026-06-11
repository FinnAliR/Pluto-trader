"""Analyze why a strategy underperformed in a backtest.

This script uses historical market data only. It does not import TradingClient
and cannot place orders.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from config import SettingsError
from historical_data import HistoricalDataError
from services.trade_analysis_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SHORT_TRADE_DAYS,
    DEFAULT_START_DATE,
    DEFAULT_SUSPICIOUS_MOVE_THRESHOLD,
    DEFAULT_TOP_DAYS,
    TradeAnalysisRequest,
    run_trade_analysis,
)
from trade_diagnostics import AnalysisFiles, save_reports


def main() -> int:
    args = parse_args()

    try:
        analysis = run_trade_analysis(
            TradeAnalysisRequest(
                strategy_name=args.strategy,
                start=args.start,
                end=args.end,
                initial_cash=args.initial_cash,
                exposure=args.exposure,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                top_days=args.top_days,
                short_trade_days=args.short_trade_days,
                suspicious_move_threshold=args.suspicious_move_threshold,
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

    print("Running data-only trade analysis. No live trading connection is used.")
    print(f"Symbol: {analysis.settings.symbol}")
    print(f"Strategy: {analysis.result.display_name}")
    print(f"Exposure when long: {args.exposure:.0%}")
    print()

    output_dir = Path(__file__).resolve().parent
    files = AnalysisFiles(
        trades=output_dir / f"trade_analysis_{analysis.strategy_name}_trades.csv",
        missed_days=output_dir / f"trade_analysis_{analysis.strategy_name}_missed_best_days.csv",
        whipsaws=output_dir / f"trade_analysis_{analysis.strategy_name}_whipsaws.csv",
        suspicious_moves=output_dir / f"trade_analysis_{analysis.strategy_name}_suspicious_moves.csv",
    )
    save_reports(
        trades=analysis.trades,
        missed_days=analysis.missed_days,
        whipsaws=analysis.whipsaws,
        suspicious_moves=analysis.suspicious_moves,
        files=files,
    )
    print_report_summary(
        analysis.result.data,
        trades=analysis.trades,
        missed_days=analysis.missed_days,
        whipsaws=analysis.whipsaws,
        suspicious_moves=analysis.suspicious_moves,
    )

    print(f"Saved trade report to: {files.trades}")
    print(f"Saved missed best days to: {files.missed_days}")
    print(f"Saved whipsaw report to: {files.whipsaws}")
    print(f"Saved suspicious move report to: {files.suspicious_moves}")
    return 0


def print_report_summary(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    missed_days: pd.DataFrame,
    whipsaws: pd.DataFrame,
    suspicious_moves: pd.DataFrame,
) -> None:
    """Print the main diagnostics."""
    total_return = (float(data["portfolio_value"].iloc[-1]) / float(data["portfolio_value"].iloc[0])) - 1
    same_exposure_return = (
        float(data["buy_hold_same_exposure_value"].iloc[-1])
        / float(data["buy_hold_same_exposure_value"].iloc[0])
    ) - 1
    missed_count = int(missed_days["missed_by_strategy"].sum()) if not missed_days.empty else 0
    if trades.empty:
        completed_trades = trades
        has_open_trade = False
    else:
        open_trade = trades["open_trade"].fillna(False).astype(bool)
        completed_trades = trades[~open_trade]
        has_open_trade = bool(open_trade.any())

    print("Trade analysis summary")
    print("----------------------")
    print(f"Strategy return:              {total_return:.2%}")
    print(f"Same-exposure buy-and-hold:   {same_exposure_return:.2%}")
    print(f"Completed trades:             {len(completed_trades)}")
    print(f"Open trade at end:            {has_open_trade}")
    print(f"Missed top up days:           {missed_count}/{len(missed_days)}")
    print(f"Short losing trades:          {len(whipsaws)}")
    print(f"Suspicious daily moves:       {len(suspicious_moves)}")

    if not completed_trades.empty:
        print(f"Best trade:                   {completed_trades['trade_return'].max():.2%}")
        print(f"Worst trade:                  {completed_trades['trade_return'].min():.2%}")
        print(f"Average days held:            {completed_trades['days_held'].mean():.1f}")

    print()
    print_top_table("Worst trades", completed_trades.sort_values("trade_return").head(5))
    print_top_table("Missed best SPY days", missed_days[missed_days["missed_by_strategy"]].head(5))
    print_top_table("Worst short whipsaws", whipsaws.head(5))
    print_top_table("Suspicious daily moves", suspicious_moves.head(5))


def print_top_table(title: str, frame: pd.DataFrame) -> None:
    """Print a compact top-five table."""
    print(title)
    print("-" * len(title))

    if frame.empty:
        print("None")
        print()
        return

    print(frame.to_string(index=False, max_cols=8))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trades, missed rallies, and whipsaws for one strategy.")
    parser.add_argument(
        "--strategy",
        default=os.getenv("STRATEGY", "ma_crossover"),
        help="Strategy to analyze: ma_crossover, core_tactical_ma, rebound_reentry_ma, rsi, breakout, buy_and_hold.",
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
        "--top-days",
        type=int,
        default=DEFAULT_TOP_DAYS,
        help=f"Number of strongest SPY days to inspect. Default: {DEFAULT_TOP_DAYS}",
    )
    parser.add_argument(
        "--short-trade-days",
        type=int,
        default=DEFAULT_SHORT_TRADE_DAYS,
        help=f"Max holding days for a losing trade to count as a whipsaw. Default: {DEFAULT_SHORT_TRADE_DAYS}",
    )
    parser.add_argument(
        "--suspicious-move-threshold",
        type=float,
        default=DEFAULT_SUSPICIOUS_MOVE_THRESHOLD,
        help="Flag absolute daily SPY moves at or above this value. Default: 0.10",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
