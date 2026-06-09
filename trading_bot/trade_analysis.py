"""Analyze why a strategy underperformed in a backtest.

This script uses historical market data only. It does not import TradingClient
and cannot place orders.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtester import run_backtest
from config import SettingsError, load_settings
from historical_data import HistoricalDataError, fetch_daily_bars, parse_date
from strategies.registry import create_strategy, normalize_strategy_name


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INITIAL_CASH = 10_000.00
DEFAULT_EXPOSURE = 0.25
DEFAULT_TOP_DAYS = 20
DEFAULT_SHORT_TRADE_DAYS = 30
DEFAULT_SUSPICIOUS_MOVE_THRESHOLD = 0.10


@dataclass(frozen=True)
class AnalysisFiles:
    """Output CSV paths."""

    trades: Path
    missed_days: Path
    whipsaws: Path
    suspicious_moves: Path


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

    print("Running data-only trade analysis. No live trading connection is used.")
    print(f"Symbol: {settings.symbol}")
    print(f"Strategy: {strategy.display_name}")
    print(f"Exposure when long: {args.exposure:.0%}")
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

    trades = build_trade_report(result.data)
    missed_days = find_missed_best_days(result.data, top_days=args.top_days)
    whipsaws = find_whipsaws(trades=trades, max_days_held=args.short_trade_days)
    suspicious_moves = find_suspicious_moves(
        data=result.data,
        threshold=args.suspicious_move_threshold,
    )

    output_dir = Path(__file__).resolve().parent
    files = AnalysisFiles(
        trades=output_dir / f"trade_analysis_{strategy_name}_trades.csv",
        missed_days=output_dir / f"trade_analysis_{strategy_name}_missed_best_days.csv",
        whipsaws=output_dir / f"trade_analysis_{strategy_name}_whipsaws.csv",
        suspicious_moves=output_dir / f"trade_analysis_{strategy_name}_suspicious_moves.csv",
    )
    save_reports(
        trades=trades,
        missed_days=missed_days,
        whipsaws=whipsaws,
        suspicious_moves=suspicious_moves,
        files=files,
    )
    print_report_summary(
        result.data,
        trades=trades,
        missed_days=missed_days,
        whipsaws=whipsaws,
        suspicious_moves=suspicious_moves,
    )

    print(f"Saved trade report to: {files.trades}")
    print(f"Saved missed best days to: {files.missed_days}")
    print(f"Saved whipsaw report to: {files.whipsaws}")
    print(f"Saved suspicious move report to: {files.suspicious_moves}")
    return 0


def build_trade_report(data: pd.DataFrame) -> pd.DataFrame:
    """Build one row per completed strategy trade."""
    rows = []
    in_trade = False
    trade_number = 0
    entry_index = 0
    entry_date = None
    entry_signal_date = None
    entry_close = 0.0
    cumulative_return = 1.0

    index_values = list(data.index)

    for i, (timestamp, row) in enumerate(data.iterrows()):
        position = float(row["position"])

        if position > 0 and not in_trade:
            in_trade = True
            trade_number += 1
            entry_index = i
            entry_date = timestamp
            entry_signal_date = index_values[i - 1] if i > 0 else timestamp
            entry_close = float(row["close"])
            cumulative_return = 1.0

        if in_trade and position > 0:
            cumulative_return *= 1 + float(row["strategy_return"])

        if in_trade and position == 0:
            exit_date = timestamp
            exit_signal_date = index_values[i - 1] if i > 0 else timestamp
            exit_close = float(row["close"])
            days_held = max(i - entry_index, 1)
            trade_return = cumulative_return - 1

            rows.append(
                {
                    "trade_number": trade_number,
                    "entry_signal_date": _date_text(entry_signal_date),
                    "entry_active_date": _date_text(entry_date),
                    "exit_signal_date": _date_text(exit_signal_date),
                    "exit_active_date": _date_text(exit_date),
                    "days_held": days_held,
                    "entry_close": entry_close,
                    "exit_close": exit_close,
                    "trade_return": trade_return,
                    "trade_return_pct": trade_return * 100,
                    "profitable": trade_return > 0,
                    "open_trade": False,
                    "last_backtest_date": "",
                }
            )
            in_trade = False

    if in_trade:
        final_timestamp = data.index[-1]
        final_close = float(data["close"].iloc[-1])
        days_held = max(len(data) - entry_index, 1)
        trade_return = cumulative_return - 1
        rows.append(
            {
                "trade_number": trade_number,
                "entry_signal_date": _date_text(entry_signal_date),
                "entry_active_date": _date_text(entry_date),
                "exit_signal_date": "",
                "exit_active_date": "",
                "days_held": days_held,
                "entry_close": entry_close,
                "exit_close": final_close,
                "trade_return": trade_return,
                "trade_return_pct": trade_return * 100,
                "profitable": trade_return > 0,
                "open_trade": True,
                "last_backtest_date": _date_text(final_timestamp),
            }
        )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "trade_number",
                "entry_signal_date",
                "entry_active_date",
                "exit_signal_date",
                "exit_active_date",
                "days_held",
                "entry_close",
                "exit_close",
                "trade_return",
                "trade_return_pct",
                "profitable",
            ]
        )

    if "open_trade" not in trades.columns:
        trades["open_trade"] = False

    if "last_backtest_date" not in trades.columns:
        trades["last_backtest_date"] = ""

    return trades


def find_missed_best_days(data: pd.DataFrame, top_days: int) -> pd.DataFrame:
    """Find the strongest SPY days where the strategy had no exposure."""
    columns = [
        "date",
        "spy_daily_return",
        "spy_daily_return_pct",
        "strategy_position",
        "strategy_return",
        "missed_by_strategy",
    ]

    if top_days <= 0:
        return pd.DataFrame(columns=columns)

    top_up_days = data.sort_values("daily_return", ascending=False).head(top_days)
    report = pd.DataFrame(
        {
            "date": [_date_text(timestamp) for timestamp in top_up_days.index],
            "spy_daily_return": top_up_days["daily_return"].values,
            "spy_daily_return_pct": (top_up_days["daily_return"] * 100).values,
            "strategy_position": top_up_days["position"].values,
            "strategy_return": top_up_days["strategy_return"].values,
            "missed_by_strategy": (top_up_days["position"] == 0).values,
        }
    )
    return report


def find_whipsaws(trades: pd.DataFrame, max_days_held: int) -> pd.DataFrame:
    """Find short losing trades."""
    if trades.empty:
        return trades.copy()

    open_trade = trades["open_trade"].fillna(False).astype(bool)
    whipsaws = trades[
        (trades["trade_return"] < 0)
        & (trades["days_held"] <= max_days_held)
        & (~open_trade)
    ].copy()
    return whipsaws.sort_values("trade_return")


def find_suspicious_moves(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Flag unusually large daily moves that may indicate bad data."""
    suspicious = data[data["daily_return"].abs() >= threshold].copy()
    if suspicious.empty:
        return pd.DataFrame(columns=["date", "close", "daily_return", "daily_return_pct"])

    return pd.DataFrame(
        {
            "date": [_date_text(timestamp) for timestamp in suspicious.index],
            "close": suspicious["close"].values,
            "daily_return": suspicious["daily_return"].values,
            "daily_return_pct": (suspicious["daily_return"] * 100).values,
        }
    ).sort_values("daily_return", key=lambda series: series.abs(), ascending=False)


def save_reports(
    trades: pd.DataFrame,
    missed_days: pd.DataFrame,
    whipsaws: pd.DataFrame,
    suspicious_moves: pd.DataFrame,
    files: AnalysisFiles,
) -> None:
    """Save all analysis CSVs."""
    trades.to_csv(files.trades, index=False)
    missed_days.to_csv(files.missed_days, index=False)
    whipsaws.to_csv(files.whipsaws, index=False)
    suspicious_moves.to_csv(files.suspicious_moves, index=False)


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


def _date_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
