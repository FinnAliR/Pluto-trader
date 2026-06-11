"""Trade diagnostic reports for completed backtest data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AnalysisFiles:
    """Output CSV paths for trade diagnostics."""

    trades: Path
    missed_days: Path
    whipsaws: Path
    suspicious_moves: Path


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


def _date_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)
