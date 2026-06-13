"""Pre-market safety checks for the paper trading bot.

This script does not submit orders. It is safe to run before market open.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from trading_bot.execution.broker import AlpacaBroker, BrokerError
from trading_bot.core.config import SettingsError, load_settings
from trading_bot.market_data.client import MarketDataError, create_data_client, get_price_data
from trading_bot.core.logger import setup_logger
from trading_bot.strategies.registry import create_strategy
from trading_bot.execution.trade_log import TradeLogger


def main() -> int:
    logger = setup_logger("INFO")
    failures = 0

    print("Running paper-trading preflight checks.")
    print("No orders will be placed.")
    print()

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        print_pass(".env file exists")
    else:
        print_fail(".env file is missing")
        return 1

    try:
        settings = load_settings()
    except SettingsError as error:
        print_fail(f"Configuration failed: {error}")
        return 1

    print_pass("PAPER_TRADING=true is set")
    print_pass("ALPACA_PAPER=true is set")
    print_pass(f"Symbol locked to {settings.symbol}")

    try:
        strategy = create_strategy(settings.strategy_name, settings=settings)
        print_pass(f"Strategy loads: {strategy.display_name}")
    except ValueError as error:
        print_fail(str(error))
        return 1

    if settings.trial_mode:
        print_warn(f"TRIAL_MODE is ON. Buy orders are capped at ${settings.trial_max_notional:.2f}.")
    else:
        print_warn("TRIAL_MODE is OFF. Consider turning it on for the first paper run.")

    broker = AlpacaBroker(settings=settings, logger=logger)
    data_client = create_data_client(settings=settings)
    trade_logger = TradeLogger(settings.trades_csv_path)

    try:
        market_open = broker.is_market_open()
        if market_open:
            print_pass("Alpaca market clock is reachable: market is open")
        else:
            print_warn("Alpaca market clock is reachable: market is closed")

        account = broker.get_account()
        print_pass(f"Alpaca account reachable. Status={account.status}")

        position_qty = broker.get_position_qty(settings.symbol)
        print_pass(f"Current {settings.symbol} position quantity: {position_qty:.6f}")

        has_open_order = broker.has_open_order(settings.symbol)
        if has_open_order:
            print_fail(f"Open {settings.symbol} order exists. Bot should not run until it is resolved.")
            failures += 1
        else:
            print_pass(f"No open {settings.symbol} orders found")

    except BrokerError as error:
        print_fail(f"Alpaca trading API check failed: {error}")
        return 1

    try:
        price_data = get_price_data(data_client=data_client, settings=settings, logger=logger)
    except MarketDataError as error:
        print_fail(f"Market data check failed: {error}")
        return 1

    if price_data.empty or "close" not in price_data.columns:
        print_fail("SPY price data is missing")
        failures += 1
    else:
        latest_close = Decimal(str(price_data["close"].dropna().iloc[-1]))
        print_pass(f"SPY price data available. Latest close=${latest_close:.2f}")

        equity = Decimal(str(account.equity))
        max_position_value = equity * Decimal(str(settings.max_position_size_fraction))
        current_position_value = Decimal(str(position_qty)) * latest_close

        if current_position_value > max_position_value:
            print_fail(
                f"Current SPY value ${current_position_value:.2f} exceeds "
                f"max position limit ${max_position_value:.2f}."
            )
            failures += 1
        else:
            print_pass(
                f"Current SPY value ${current_position_value:.2f} is within "
                f"max position limit ${max_position_value:.2f}."
            )

    local_trades_today = trade_logger.count_submitted_trades_today()
    try:
        alpaca_orders_today = broker.count_orders_submitted_today(settings.symbol)
    except BrokerError as error:
        print_fail(f"Alpaca order-history check failed: {error}")
        return 1

    trades_today = max(local_trades_today, alpaca_orders_today)
    if trades_today >= settings.max_daily_trades:
        print_fail(
            f"Daily trade limit reached: effective={trades_today}/{settings.max_daily_trades} "
            f"(local_csv={local_trades_today}, alpaca={alpaca_orders_today})"
        )
        failures += 1
    else:
        print_pass(
            f"Daily trade limit available: effective={trades_today}/{settings.max_daily_trades} "
            f"(local_csv={local_trades_today}, alpaca={alpaca_orders_today})"
        )

    print()
    if failures:
        print_fail(f"Preflight finished with {failures} blocking issue(s). Do not run main.py yet.")
        return 1

    print_pass("Preflight passed. main.py is safe to run manually when you are ready.")
    return 0


def print_pass(message: str) -> None:
    print(f"[PASS] {message}")


def print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def print_fail(message: str) -> None:
    print(f"[FAIL] {message}")


if __name__ == "__main__":
    raise SystemExit(main())

