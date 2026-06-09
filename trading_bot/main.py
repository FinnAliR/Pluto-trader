"""Entry point for the Alpaca paper trading bot."""

from __future__ import annotations

import sys

from broker import AlpacaBroker, BrokerError
from config import SettingsError, load_settings
from data import MarketDataError, create_data_client, get_price_data
from logger import setup_logger
from risk import RiskManager
from strategy import TradeAction, calculate_moving_average_signal
from trade_log import TradeLogEntry, TradeLogger


def run_bot() -> int:
    """Run one trading decision cycle.

    This bot is intended to be run on a schedule, for example once per day
    after the market opens. Each run checks the market, data, current position,
    open orders, strategy signal, and risk rules before submitting an order.
    """
    try:
        settings = load_settings()
    except SettingsError as error:
        logger = setup_logger()
        logger.error("Configuration error: %s", error)
        return 1

    logger = setup_logger(settings.log_level)
    logger.info("Starting SPY moving-average bot.")
    logger.info("Safety check: paper trading is hard-coded ON. Live trading is not supported.")
    logger.info("Configured symbol=%s timeframe=%s", settings.symbol, settings.timeframe)
    logger.info(
        "Configured risk limits: max_daily_trades=%d max_position_size=%.0f%% of equity",
        settings.max_daily_trades,
        settings.max_position_size_fraction * 100,
    )

    broker = AlpacaBroker(settings=settings, logger=logger)
    data_client = create_data_client(settings=settings)
    risk_manager = RiskManager(settings=settings, logger=logger)
    trade_logger = TradeLogger(settings.trades_csv_path)

    try:
        if not broker.is_market_open():
            logger.info("Decision: market is closed, so no trade will be placed.")
            return 0

        account = broker.get_account()
        position_qty = broker.get_position_qty(settings.symbol)
        owns_position = position_qty > 0
        has_open_order = broker.has_open_order(settings.symbol)
    except BrokerError as error:
        logger.error("Alpaca trading API error: %s", error)
        return 1

    logger.info("Current SPY position quantity: %.6f", position_qty)
    logger.info("Open SPY order exists: %s", has_open_order)

    try:
        price_data = get_price_data(
            data_client=data_client,
            settings=settings,
            logger=logger,
        )
    except MarketDataError as error:
        logger.error("Alpaca market data API error: %s", error)
        return 1

    decision = calculate_moving_average_signal(
        price_data=price_data,
        owns_position=owns_position,
        settings=settings,
        logger=logger,
    )

    logger.info("Strategy decision: %s. Reason: %s", decision.action.value, decision.reason)

    if decision.action == TradeAction.HOLD:
        logger.info("Decision: no order needed.")
        return 0

    risk_result = risk_manager.evaluate(
        decision=decision,
        account=account,
        position_qty=position_qty,
        has_open_order=has_open_order,
        submitted_trades_today=trade_logger.count_submitted_trades_today(),
    )

    if not risk_result.approved:
        logger.warning("Risk check blocked order: %s", risk_result.reason)
        trade_logger.record(
            TradeLogEntry(
                symbol=settings.symbol,
                decision=decision,
                status="BLOCKED",
                risk_reason=risk_result.reason,
                position_qty_before=position_qty,
            )
        )
        return 0

    try:
        if decision.action == TradeAction.BUY:
            order = broker.submit_market_buy(
                symbol=settings.symbol,
                notional=risk_result.notional,
                client_order_id=risk_result.client_order_id,
            )
            trade_logger.record(
                TradeLogEntry(
                    symbol=settings.symbol,
                    decision=decision,
                    status="SUBMITTED",
                    risk_reason=risk_result.reason,
                    order_id=str(order.id),
                    order_status=str(order.status),
                    client_order_id=risk_result.client_order_id,
                    notional=risk_result.notional,
                    position_qty_before=position_qty,
                )
            )
            return 0

        if decision.action == TradeAction.SELL:
            order = broker.submit_market_sell(
                symbol=settings.symbol,
                qty=risk_result.qty,
                client_order_id=risk_result.client_order_id,
            )
            trade_logger.record(
                TradeLogEntry(
                    symbol=settings.symbol,
                    decision=decision,
                    status="SUBMITTED",
                    risk_reason=risk_result.reason,
                    order_id=str(order.id),
                    order_status=str(order.status),
                    client_order_id=risk_result.client_order_id,
                    qty=risk_result.qty,
                    position_qty_before=position_qty,
                )
            )
            return 0
    except BrokerError as error:
        logger.error("Alpaca order submission failed: %s", error)
        trade_logger.record(
            TradeLogEntry(
                symbol=settings.symbol,
                decision=decision,
                status="FAILED",
                risk_reason=risk_result.reason,
                client_order_id=risk_result.client_order_id,
                qty=risk_result.qty,
                notional=risk_result.notional,
                position_qty_before=position_qty,
                error_message=str(error),
            )
        )
        return 1

    logger.info("Decision: unsupported action %s, so no order was placed.", decision.action.value)
    return 0


if __name__ == "__main__":
    sys.exit(run_bot())
