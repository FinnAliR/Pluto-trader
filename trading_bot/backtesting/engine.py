"""Reusable backtest engine for strategy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import pandas as pd

from trading_bot.strategies.base import Strategy


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    """Results for one strategy backtest."""

    strategy_name: str
    display_name: str
    data: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    strategy: Strategy,
    price_data: pd.DataFrame,
    initial_cash: float,
    exposure: float,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Run one long-only backtest.

    Signals are generated from today's close and applied to tomorrow's return.
    That avoids lookahead bias from pretending we knew today's close before the
    close happened.
    """
    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty.")

    if "close" not in price_data.columns:
        raise ValueError("Price data must include a close column.")

    if not 0 < exposure <= 1:
        raise ValueError("Exposure must be greater than 0 and no more than 1.")

    data = price_data.copy()
    data["close"] = data["close"].astype(float)

    signal = strategy.generate_signals(data)
    data["signal"] = signal.reindex(data.index).fillna(0).clip(lower=0, upper=1)
    data["position"] = data["signal"].shift(1).fillna(0) * exposure
    data["daily_return"] = data["close"].pct_change().fillna(0)

    data["position_change"] = data["position"].diff().abs().fillna(data["position"].abs())
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    data["trade_cost"] = data["position_change"] * cost_rate

    data["strategy_return"] = (data["position"] * data["daily_return"]) - data["trade_cost"]
    data["portfolio_value"] = initial_cash * (1 + data["strategy_return"]).cumprod()
    data["buy_hold_full_return"] = data["daily_return"]
    data["buy_hold_same_exposure_return"] = exposure * data["daily_return"]
    data["buy_hold_full_value"] = initial_cash * (1 + data["buy_hold_full_return"]).cumprod()
    data["buy_hold_same_exposure_value"] = initial_cash * (1 + data["buy_hold_same_exposure_return"]).cumprod()
    data["buy_hold_value"] = data["buy_hold_full_value"]
    data["trade"] = (data["position_change"] > 0).astype(int)

    metrics = calculate_metrics(data=data, initial_cash=initial_cash, exposure=exposure)
    return BacktestResult(
        strategy_name=strategy.name,
        display_name=strategy.display_name,
        data=data,
        metrics=metrics,
    )


def calculate_metrics(data: pd.DataFrame, initial_cash: float, exposure: float) -> dict[str, float]:
    """Calculate practical comparison metrics."""
    ending_value = float(data["portfolio_value"].iloc[-1])
    buy_hold_full_ending_value = float(data["buy_hold_full_value"].iloc[-1])
    buy_hold_same_exposure_ending_value = float(data["buy_hold_same_exposure_value"].iloc[-1])
    total_return = (ending_value / initial_cash) - 1
    buy_hold_full_return = (buy_hold_full_ending_value / initial_cash) - 1
    buy_hold_same_exposure_return = (buy_hold_same_exposure_ending_value / initial_cash) - 1

    elapsed_days = max((data.index[-1] - data.index[0]).days, 1)
    years = elapsed_days / 365.25
    cagr = (ending_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0.0

    max_drawdown = calculate_max_drawdown(data["portfolio_value"])
    returns = data["strategy_return"]
    sharpe = calculate_sharpe_ratio(returns)
    trades = int(data["trade"].sum())
    trade_returns = calculate_closed_trade_returns(data)
    win_rate = float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0
    average_trade_return = float(trade_returns.mean()) if not trade_returns.empty else 0.0
    time_in_market = float((data["position"] > 0).mean())
    average_position = float(data["position"].mean())
    missed_top_20_up_days = count_missed_top_up_days(data=data, top_days=20)

    return {
        "exposure": exposure,
        "total_return": total_return,
        "buy_hold_same_exposure_return": buy_hold_same_exposure_return,
        "buy_hold_full_return": buy_hold_full_return,
        "buy_hold_return": buy_hold_full_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": float(trades),
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "time_in_market": time_in_market,
        "average_position": average_position,
        "missed_top_20_up_days": float(missed_top_20_up_days),
        "final_value": ending_value,
    }


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """Return the worst peak-to-trough portfolio decline."""
    running_high = portfolio_value.cummax()
    drawdown = (portfolio_value / running_high) - 1
    return float(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """Calculate a simple annualized Sharpe ratio with 0% risk-free rate."""
    standard_deviation = returns.std()
    if standard_deviation == 0 or pd.isna(standard_deviation):
        return 0.0

    return float((returns.mean() / standard_deviation) * (TRADING_DAYS_PER_YEAR**0.5))


def calculate_closed_trade_returns(data: pd.DataFrame) -> pd.Series:
    """Estimate returns for completed long periods only."""
    trade_returns: list[float] = []
    in_trade = False
    cumulative_return = 1.0

    for _, row in data.iterrows():
        if row["position"] > 0:
            if not in_trade:
                in_trade = True
                cumulative_return = 1.0
            cumulative_return *= 1 + float(row["strategy_return"])
            continue

        if in_trade:
            trade_returns.append(cumulative_return - 1)
            in_trade = False

    if in_trade:
        trade_returns.append(cumulative_return - 1)

    return pd.Series(trade_returns, dtype=float)


def count_missed_top_up_days(data: pd.DataFrame, top_days: int) -> int:
    """Count top SPY up days where the strategy had zero exposure."""
    if top_days <= 0 or data.empty:
        return 0

    top_up_days = data.sort_values("daily_return", ascending=False).head(top_days)
    return int((top_up_days["position"] == 0).sum())


def results_to_summary_frame(results: list[BacktestResult]) -> pd.DataFrame:
    """Convert strategy results into a printable/savable table."""
    rows = []
    for result in results:
        row = {
            "strategy": result.strategy_name,
            "display_name": result.display_name,
        }
        row.update(result.metrics)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_return", ascending=False)


def print_summary_table(summary: pd.DataFrame, title: str) -> None:
    """Print a readable comparison table."""
    if summary.empty:
        print(f"{title}: no results")
        return

    print(title)
    print("-" * len(title))
    headers = [
        "strategy",
        "total_return",
        "buy_hold_same_exposure_return",
        "buy_hold_full_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "trades",
        "win_rate",
        "time_in_market",
        "average_position",
        "missed_top_20_up_days",
    ]
    print(
        f"{'Strategy':<16} {'Return':>10} {'B&H Same':>10} {'B&H Full':>10} "
        f"{'CAGR':>10} {'Max DD':>10} {'Sharpe':>8} {'Trades':>8} "
        f"{'Win%':>8} {'In Mkt':>8} {'Avg Pos':>8} {'Miss20':>7}"
    )
    for _, row in summary[headers].iterrows():
        print(
            f"{row['strategy']:<16} "
            f"{row['total_return']:>9.2%} "
            f"{row['buy_hold_same_exposure_return']:>9.2%} "
            f"{row['buy_hold_full_return']:>9.2%} "
            f"{row['cagr']:>9.2%} "
            f"{row['max_drawdown']:>9.2%} "
            f"{row['sharpe']:>8.2f} "
            f"{int(row['trades']):>8} "
            f"{row['win_rate']:>8.2%} "
            f"{row['time_in_market']:>8.2%} "
            f"{row['average_position']:>8.2%} "
            f"{int(row['missed_top_20_up_days']):>7}"
        )
    print()


def save_summary_csv(summary: pd.DataFrame, path: Path) -> None:
    """Save strategy metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def plot_equity_curves(
    results: list[BacktestResult],
    path: Path,
    show_plot: bool,
) -> None:
    """Plot all strategy portfolio values on one chart."""
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_equity_curve_figure(results)
    figure.savefig(path)

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def create_equity_curve_figure(
    results: list[BacktestResult],
    focus_recent_days: int | None = None,
) -> Figure:
    """Create an equity-curve and drawdown figure without saving it."""
    figure, (equity_axis, drawdown_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    figure.patch.set_facecolor("#f8fafc")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, result in enumerate(results):
        color = color_cycle[index % len(color_cycle)] if color_cycle else None
        equity_axis.plot(
            result.data.index,
            result.data["portfolio_value"],
            label=result.display_name,
            linewidth=2.2,
            color=color,
        )
        drawdown = (result.data["portfolio_value"] / result.data["portfolio_value"].cummax()) - 1
        drawdown_axis.plot(
            result.data.index,
            drawdown,
            label=result.display_name,
            linewidth=1.4,
            color=color,
            alpha=0.9,
        )

    first_result = results[0]
    benchmark_exposure = first_result.metrics["exposure"]
    equity_axis.plot(
        first_result.data.index,
        first_result.data["buy_hold_same_exposure_value"],
        label=f"buy_hold_same_exposure_{benchmark_exposure:.0%}",
        linestyle=":",
        linewidth=2,
        alpha=0.9,
        color="#334155",
    )
    equity_axis.plot(
        first_result.data.index,
        first_result.data["buy_hold_full_value"],
        label="buy_hold_full_exposure",
        linestyle="--",
        alpha=0.7,
        color="#64748b",
    )

    same_exposure_drawdown = (
        first_result.data["buy_hold_same_exposure_value"]
        / first_result.data["buy_hold_same_exposure_value"].cummax()
    ) - 1
    drawdown_axis.plot(
        first_result.data.index,
        same_exposure_drawdown,
        label=f"buy_hold_same_exposure_{benchmark_exposure:.0%}",
        linestyle=":",
        linewidth=1.3,
        color="#334155",
        alpha=0.9,
    )

    for axis in (equity_axis, drawdown_axis):
        axis.set_facecolor("#ffffff")
        axis.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    equity_axis.set_title("Strategy Equity Curves", loc="left", fontsize=13, fontweight="bold")
    equity_axis.set_ylabel("Portfolio value")
    equity_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"${value:,.0f}"))
    equity_axis.legend(loc="upper left", ncols=2, fontsize=9, frameon=False)

    drawdown_axis.axhline(0, color="#94a3b8", linewidth=0.8)
    drawdown_axis.set_ylabel("Drawdown")
    drawdown_axis.set_xlabel("Date")
    drawdown_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.0%}"))

    locator = _date_axis_locator()
    drawdown_axis.xaxis.set_major_locator(locator)
    drawdown_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    if focus_recent_days is not None and focus_recent_days > 0:
        index = first_result.data.index
        if len(index) > focus_recent_days:
            equity_axis.set_xlim(index[-focus_recent_days], index[-1])

    return figure


def _date_axis_locator() -> mdates.AutoDateLocator:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    locator.intervald[mdates.MONTHLY] = [1, 2, 3, 4, 6, 12]
    locator.intervald[mdates.DAILY] = [1, 2, 3, 7, 14, 21, 30, 60, 90]
    return locator

