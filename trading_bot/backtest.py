"""Backtest the SPY MA20/MA50 strategy with historical daily data.

This script uses Alpaca market data only. It does not import TradingClient,
does not check positions, and cannot place orders.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


SYMBOL = "SPY"
FAST_MA_PERIOD = 20
SLOW_MA_PERIOD = 50
POSITION_SIZE_FRACTION = 0.25
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INITIAL_CASH = 10_000.00
PLOT_FILE_NAME = "backtest_portfolio_value.png"


@dataclass(frozen=True)
class BacktestConfig:
    """Settings used by the backtest."""

    api_key: str
    secret_key: str
    data_feed: DataFeed
    start_date: datetime
    end_date: datetime | None
    initial_cash: float
    show_plot: bool


def main() -> int:
    """Run the backtest and print the results."""
    config = load_config()

    print("Running data-only backtest. No live trading connection is used.")
    print(f"Symbol: {SYMBOL}")
    print(f"Strategy: MA{FAST_MA_PERIOD} / MA{SLOW_MA_PERIOD} crossover")
    print(f"Strategy exposure when long: {POSITION_SIZE_FRACTION:.0%}")
    print(f"Start date: {config.start_date.date()}")
    print(f"End date: {config.end_date.date() if config.end_date else 'latest available'}")
    print()

    price_data = fetch_daily_spy_data(config)
    if price_data.empty:
        print("No historical price data was returned. Check your Alpaca keys and data feed.")
        return 1

    results = run_backtest(price_data=price_data, initial_cash=config.initial_cash)
    print_results(results)
    plot_portfolio_value(results=results, show_plot=config.show_plot)

    return 0


def load_config() -> BacktestConfig:
    """Load API keys from .env and simple command-line settings."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Backtest the SPY MA20/MA50 strategy using Alpaca daily historical data."
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
        "--no-show",
        action="store_true",
        help="Save the plot without opening a chart window.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    paper_value = os.getenv("ALPACA_PAPER", "true").strip().lower()

    if not api_key or api_key == "your_api_key_here":
        raise SystemExit("Missing ALPACA_API_KEY. Add your Alpaca paper key to trading_bot/.env.")

    if not secret_key or secret_key == "your_secret_key_here":
        raise SystemExit("Missing ALPACA_SECRET_KEY. Add your Alpaca paper secret to trading_bot/.env.")

    if paper_value not in {"1", "true", "yes", "y", "on"}:
        raise SystemExit("ALPACA_PAPER must be true. This project is paper-only.")

    if args.initial_cash <= 0:
        raise SystemExit("--initial-cash must be greater than 0.")

    return BacktestConfig(
        api_key=api_key,
        secret_key=secret_key,
        data_feed=parse_data_feed(os.getenv("DATA_FEED", "iex")),
        start_date=parse_date(args.start),
        end_date=parse_date(args.end) if args.end else None,
        initial_cash=args.initial_cash,
        show_plot=not args.no_show,
    )


def parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD text into a timezone-aware UTC datetime."""
    try:
        parsed_date = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise SystemExit(f"Invalid date '{value}'. Use YYYY-MM-DD.") from error

    return parsed_date.replace(tzinfo=timezone.utc)


def parse_data_feed(feed_value: str) -> DataFeed:
    """Convert DATA_FEED text into an Alpaca data feed enum."""
    normalized = feed_value.strip().lower()

    if normalized == "iex":
        return DataFeed.IEX

    if normalized == "sip":
        return DataFeed.SIP

    raise SystemExit("DATA_FEED must be iex or sip.")


def fetch_daily_spy_data(config: BacktestConfig) -> pd.DataFrame:
    """Download daily adjusted SPY bars from Alpaca market data."""
    client = StockHistoricalDataClient(config.api_key, config.secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Day,
        start=config.start_date,
        end=config.end_date,
        feed=config.data_feed,
        adjustment=Adjustment.ALL,
    )

    bars = client.get_stock_bars(request)
    data_frame = bars.df

    if data_frame is None or data_frame.empty:
        return pd.DataFrame()

    if isinstance(data_frame.index, pd.MultiIndex):
        symbol_level = "symbol" if "symbol" in data_frame.index.names else 0
        data_frame = data_frame.xs(SYMBOL, level=symbol_level)

    return data_frame.sort_index()


def run_backtest(price_data: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """Calculate signals, trades, returns, and portfolio value."""
    data = price_data.copy()
    data["ma20"] = data["close"].rolling(FAST_MA_PERIOD).mean()
    data["ma50"] = data["close"].rolling(SLOW_MA_PERIOD).mean()

    # Signal says what we want to own after today's close.
    data["signal"] = 0
    data.loc[data["ma20"] > data["ma50"], "signal"] = 1

    # Shift by one day so today's signal affects tomorrow's return.
    data["position"] = data["signal"].shift(1).fillna(0) * POSITION_SIZE_FRACTION

    data["daily_return"] = data["close"].pct_change().fillna(0)
    data["strategy_return"] = data["position"] * data["daily_return"]
    data["portfolio_value"] = initial_cash * (1 + data["strategy_return"]).cumprod()
    data["buy_hold_value"] = initial_cash * (1 + data["daily_return"]).cumprod()
    data["trade"] = (data["position"].diff().abs().fillna(0) > 0).astype(int)

    return data.dropna(subset=["ma20", "ma50"]).copy()


def print_results(results: pd.DataFrame) -> None:
    """Print beginner-friendly performance metrics."""
    if results.empty:
        print(f"Not enough data to calculate MA{SLOW_MA_PERIOD}.")
        return

    total_return = (results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0]) - 1
    buy_hold_return = (results["buy_hold_value"].iloc[-1] / results["buy_hold_value"].iloc[0]) - 1
    number_of_trades = int(results["trade"].sum())
    max_drawdown = calculate_max_drawdown(results["portfolio_value"])

    print("Backtest results")
    print("----------------")
    print(f"First test date:       {results.index[0].date()}")
    print(f"Last test date:        {results.index[-1].date()}")
    print(f"Starting value:        ${results['portfolio_value'].iloc[0]:,.2f}")
    print(f"Ending value:          ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total return:          {total_return:.2%}")
    print(f"Buy-and-hold return:   {buy_hold_return:.2%} full SPY exposure")
    print(f"Number of trades:      {number_of_trades}")
    print(f"Max drawdown:          {max_drawdown:.2%}")
    print()


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """Return the worst peak-to-trough portfolio decline."""
    running_high = portfolio_value.cummax()
    drawdown = (portfolio_value / running_high) - 1
    return float(drawdown.min())


def plot_portfolio_value(results: pd.DataFrame, show_plot: bool) -> None:
    """Save and optionally show a portfolio value chart."""
    if results.empty:
        return

    plot_path = Path(__file__).resolve().parent / PLOT_FILE_NAME

    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["portfolio_value"], label="MA20/MA50 strategy")
    plt.plot(results.index, results["buy_hold_value"], label="Buy and hold", alpha=0.75)
    plt.title("SPY Backtest Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)

    print(f"Saved plot to: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
