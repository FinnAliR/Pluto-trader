"""Alpaca paper-trading broker wrapper."""

from __future__ import annotations

from logging import Logger

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from config import Settings


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
        account = self.client.get_account()
        self.logger.info("Account status: %s", account.status)
        self.logger.info("Account trading blocked: %s", account.trading_blocked)
        self.logger.info("Buying power: %s", account.buying_power)
        self.logger.info("Cash: %s", account.cash)
        return account

    def is_market_open(self) -> bool:
        """Return True only when Alpaca says the market is open."""
        clock = self.client.get_clock()
        self.logger.info("Market timestamp: %s", clock.timestamp)
        self.logger.info("Market is open: %s", clock.is_open)
        self.logger.info("Next market open: %s", clock.next_open)
        self.logger.info("Next market close: %s", clock.next_close)
        return bool(clock.is_open)

    def get_position_qty(self, symbol: str) -> float:
        """Return the current long quantity for a symbol, or 0 if not held."""
        positions = self.client.get_all_positions()

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
        open_orders = self.client.get_orders(filter=request)

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
        order = self.client.submit_order(order_data=order_request)
        self.logger.info("Submitted paper buy order: id=%s status=%s", order.id, order.status)
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
        order = self.client.submit_order(order_data=order_request)
        self.logger.info("Submitted paper sell order: id=%s status=%s", order.id, order.status)
        return order

    def _validate_symbol(self, symbol: str) -> None:
        if symbol.upper() != self.settings.symbol:
            raise ValueError(f"Unexpected symbol {symbol}. This bot only trades {self.settings.symbol}.")
