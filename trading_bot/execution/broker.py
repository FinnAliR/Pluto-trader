"""Broker wrappers for safe paper execution."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from logging import Logger
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from requests.exceptions import RequestException

from trading_bot.core.config import Settings


NEW_YORK_TIME = ZoneInfo("America/New_York")


class BrokerError(RuntimeError):
    """Raised when Alpaca trading API calls fail."""


def create_broker(settings: Settings, logger: Logger):
    """Create the configured paper-execution broker."""
    if settings.execution_mode == "live_paper":
        from trading_bot.execution.live_paper import LivePaperBroker

        return LivePaperBroker(settings=settings, logger=logger)

    return AlpacaBroker(settings=settings, logger=logger)


class AlpacaBroker:
    """Small wrapper around the official alpaca-py TradingClient.

    The TradingClient is always created with paper=True. There is no setting in
    this project that can switch it to live trading.
    """

    def __init__(self, settings: Settings, logger: Logger) -> None:
        if not settings.paper_trading:
            raise ValueError("Paper trading must be enabled. Live trading is blocked.")

        self.settings = settings
        self.logger = logger
        self.client = TradingClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            paper=True,
        )

    def get_account(self):
        """Fetch account details and log key safety fields."""
        try:
            account = self.client.get_account()
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to fetch Alpaca account details: {error}") from error

        self.logger.info("Account status: %s", account.status)
        self.logger.info("Account trading blocked: %s", account.trading_blocked)
        self.logger.info("Buying power: %s", account.buying_power)
        self.logger.info("Cash: %s", account.cash)
        return account

    def is_market_open(self) -> bool:
        """Return True only when Alpaca says the market is open."""
        try:
            clock = self.client.get_clock()
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to fetch Alpaca market clock: {error}") from error

        self.logger.info("Market timestamp: %s", clock.timestamp)
        self.logger.info("Market is open: %s", clock.is_open)
        self.logger.info("Next market open: %s", clock.next_open)
        self.logger.info("Next market close: %s", clock.next_close)
        return bool(clock.is_open)

    def get_position_qty(self, symbol: str) -> float:
        """Return the current long quantity for a symbol, or 0 if not held."""
        try:
            positions = self.client.get_all_positions()
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to fetch Alpaca positions: {error}") from error

        for position in positions:
            if position.symbol.upper() == symbol.upper():
                return float(position.qty)

        return 0.0

    def has_open_order(self, symbol: str) -> bool:
        """Check for any open order for the symbol to avoid duplicates."""
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
            limit=50,
        )
        try:
            open_orders = self.client.get_orders(filter=request)
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to fetch Alpaca open orders: {error}") from error

        if open_orders:
            for order in open_orders:
                self.logger.warning(
                    "Existing open order found: id=%s symbol=%s side=%s qty=%s notional=%s",
                    order.id,
                    order.symbol,
                    order.side,
                    order.qty,
                    order.notional,
                )
            return True

        return False

    def count_orders_submitted_today(self, symbol: str) -> int:
        """Count Alpaca orders submitted during the current New York market date."""
        now_new_york = datetime.now(timezone.utc).astimezone(NEW_YORK_TIME)
        start_new_york = datetime.combine(now_new_york.date(), time.min, tzinfo=NEW_YORK_TIME)
        end_new_york = start_new_york + timedelta(days=1)

        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            symbols=[symbol],
            after=start_new_york.astimezone(timezone.utc),
            until=end_new_york.astimezone(timezone.utc),
            limit=500,
        )

        try:
            orders = self.client.get_orders(filter=request)
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to fetch Alpaca order history: {error}") from error

        count = sum(1 for order in (orders or []) if order.symbol.upper() == symbol.upper())
        self.logger.info("Alpaca orders submitted today for %s: %d", symbol, count)
        return count

    def submit_market_buy(self, symbol: str, notional: float, client_order_id: str):
        """Submit a paper market buy order by notional value."""
        self._validate_symbol(symbol)

        self.logger.warning(
            "WARNING: About to place PAPER BUY order for %s with notional $%.2f.",
            symbol,
            notional,
        )

        order_request = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            extended_hours=False,
            client_order_id=client_order_id,
        )
        try:
            order = self.client.submit_order(order_data=order_request)
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to submit paper buy order: {error}") from error

        self.logger.info(
            "ORDER CONFIRMATION: paper BUY submitted. id=%s client_order_id=%s status=%s symbol=%s notional=%.2f",
            order.id,
            client_order_id,
            order.status,
            symbol,
            notional,
        )
        return order

    def submit_market_sell(self, symbol: str, qty: float, client_order_id: str):
        """Submit a paper market sell order for the held SPY quantity only."""
        self._validate_symbol(symbol)

        self.logger.warning(
            "WARNING: About to place PAPER SELL order for %s with quantity %.6f.",
            symbol,
            qty,
        )

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            extended_hours=False,
            client_order_id=client_order_id,
        )
        try:
            order = self.client.submit_order(order_data=order_request)
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to submit paper sell order: {error}") from error

        self.logger.info(
            "ORDER CONFIRMATION: paper SELL submitted. id=%s client_order_id=%s status=%s symbol=%s qty=%.6f",
            order.id,
            client_order_id,
            order.status,
            symbol,
            qty,
        )
        return order

    def _validate_symbol(self, symbol: str) -> None:
        if symbol.upper() != self.settings.symbol:
            raise ValueError(f"Unexpected symbol {symbol}. This bot only trades {self.settings.symbol}.")

