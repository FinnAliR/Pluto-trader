"""Entry point for paper execution modes."""

from __future__ import annotations

import sys
from dataclasses import replace
from decimal import Decimal

from trading_bot.core.config import EXECUTION_MODE_LIVE_PAPER, SettingsError, load_settings
from trading_bot.execution.broker import BrokerError, create_broker
from trading_bot.market_data.client import MarketDataError, create_data_client, get_price_data
from trading_bot.core.logger import setup_logger
from trading_bot.execution.risk import RiskManager
from trading_bot.strategies.base import StrategyDecision, TradeAction
from trading_bot.strategies.registry import create_strategy
from trading_bot.execution.trade_log import TradeLogEntry, TradeLogger


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
    strategy = create_strategy(settings.strategy_name, settings=settings)

    logger.info("Starting %s paper execution bot in %s mode.", settings.symbol, settings.execution_mode)
    if settings.execution_mode == EXECUTION_MODE_LIVE_PAPER:
        logger.warning("LIVE PAPER MODE is ON: fills are simulated locally; Alpaca orders will not be submitted.")
        logger.info("Live-paper state file: %s", settings.live_paper_state_path)
    else:
        logger.info("Safety check: Alpaca broker is hard-coded to paper=True. Live trading is not supported.")
    logger.info("Configured symbol=%s timeframe=%s strategy=%s", settings.symbol, settings.timeframe, strategy.display_name)
    logger.info(
        "Configured risk limits: max_daily_trades=%d max_position_size=%.0f%% of equity",
        settings.max_daily_trades,
        settings.max_position_size_fraction * 100,
    )
    if settings.trial_mode:
        logger.warning("TRIAL MODE is ON: buy orders are capped at $%.2f.", settings.trial_max_notional)

    broker = create_broker(settings=settings, logger=logger)
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
        logger.error("Paper broker error: %s", error)
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

    try:
        if mark_broker_to_market_if_supported(
            broker=broker,
            latest_close=latest_close_from_price_data(price_data),
        ):
            account = broker.get_account()
    except BrokerError as error:
        logger.error("Broker mark-to-market failed: %s", error)
        return 1

    decision = strategy.make_decision(
        price_data=price_data,
        owns_position=owns_position,
        logger=logger,
    )
    decision = resolve_target_rebalance_decision(
        decision=decision,
        account=account,
        position_qty=position_qty,
        settings=settings,
        logger=logger,
    )

    logger.info("Strategy decision: %s. Reason: %s", decision.action.value, decision.reason)

    if decision.action == TradeAction.HOLD:
        logger.info("Decision: no order needed.")
        return 0

    local_trades_today = trade_logger.count_submitted_trades_today()
    try:
        broker_orders_today = broker.count_orders_submitted_today(settings.symbol)
    except BrokerError as error:
        logger.error("Broker order-history check failed: %s", error)
        return 1

    submitted_trades_today = max(local_trades_today, broker_orders_today)
    logger.info(
        "Submitted trades today: local_csv=%d broker=%d effective=%d/%d",
        local_trades_today,
        broker_orders_today,
        submitted_trades_today,
        settings.max_daily_trades,
    )

    risk_result = risk_manager.evaluate(
        decision=decision,
        account=account,
        position_qty=position_qty,
        has_open_order=has_open_order,
        submitted_trades_today=submitted_trades_today,
    )

    if not risk_result.approved:
        logger.warning("Risk check blocked order: %s", risk_result.reason)
        trade_logger.record(
            TradeLogEntry(
                symbol=settings.symbol,
                decision=decision,
                execution_mode=settings.execution_mode,
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
                    execution_mode=settings.execution_mode,
                    status=order_trade_log_status(settings.execution_mode),
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
                    execution_mode=settings.execution_mode,
                    status=order_trade_log_status(settings.execution_mode),
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
        logger.error("Paper order submission failed: %s", error)
        trade_logger.record(
            TradeLogEntry(
                symbol=settings.symbol,
                decision=decision,
                execution_mode=settings.execution_mode,
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


def latest_close_from_price_data(price_data) -> float | None:
    """Return the latest valid close from a price dataframe."""
    if price_data is None or price_data.empty or "close" not in price_data.columns:
        return None

    close_prices = price_data["close"].dropna()
    if close_prices.empty:
        return None

    return float(close_prices.iloc[-1])


def mark_broker_to_market_if_supported(broker, latest_close: float | None) -> bool:
    """Mark local simulated brokers to market before risk sizing."""
    mark_to_market = getattr(broker, "mark_to_market", None)
    if not callable(mark_to_market):
        return False

    if latest_close is None or latest_close <= 0:
        return False

    mark_to_market(latest_close)
    return True


def order_trade_log_status(execution_mode: str) -> str:
    """Use FILLED for local simulations and SUBMITTED for broker-backed paper orders."""
    if execution_mode == EXECUTION_MODE_LIVE_PAPER:
        return "FILLED"
    return "SUBMITTED"


def resolve_target_rebalance_decision(
    decision: StrategyDecision,
    account,
    position_qty: float,
    settings,
    logger,
) -> StrategyDecision:
    """Turn a target-aware HOLD into a BUY or SELL rebalance when needed."""
    if decision.action != TradeAction.HOLD:
        return decision

    if decision.target_fraction is None:
        return decision

    if decision.latest_close is None or decision.latest_close <= 0:
        return decision

    target_fraction = Decimal(str(decision.target_fraction))
    if not Decimal("0") <= target_fraction <= Decimal("1"):
        return decision

    latest_price = Decimal(str(decision.latest_close))
    equity = Decimal(str(account.equity))
    if equity <= 0:
        return decision

    max_position_value = equity * Decimal(str(settings.max_position_size_fraction))
    target_position_value = max_position_value * target_fraction
    current_position_value = Decimal(str(position_qty)) * latest_price
    value_gap = target_position_value - current_position_value
    min_order_notional = Decimal(str(settings.min_order_notional))

    logger.info(
        "Target allocation check: current SPY value=$%.2f target=$%.2f gap=$%.2f",
        current_position_value,
        target_position_value,
        value_gap,
    )

    if abs(value_gap) < min_order_notional:
        return replace(
            decision,
            reason=(
                f"{decision.reason} Current allocation is within the "
                f"${settings.min_order_notional:.2f} minimum rebalance threshold."
            ),
        )

    if value_gap > 0:
        return replace(
            decision,
            action=TradeAction.BUY,
            reason=f"{decision.reason} Current allocation is below target, so a buy rebalance is requested.",
        )

    return replace(
        decision,
        action=TradeAction.SELL,
        reason=f"{decision.reason} Current allocation is above target, so a sell rebalance is requested.",
    )


if __name__ == "__main__":
    sys.exit(run_bot())

