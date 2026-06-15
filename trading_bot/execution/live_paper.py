"""Local live-paper broker that simulates fills against live market prices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from logging import Logger
from uuid import uuid4

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from requests.exceptions import RequestException

from trading_bot.core.config import Settings
from trading_bot.execution.broker import BrokerError, NEW_YORK_TIME


CENT = Decimal("0.01")
SHARE_QUANTITY = Decimal("0.000001")


@dataclass(frozen=True)
class LivePaperAccount:
    """Minimal account shape required by RiskManager."""

    status: str
    trading_blocked: bool
    buying_power: str
    cash: str
    equity: str


@dataclass(frozen=True)
class LivePaperOrder:
    """Minimal order shape written to the normal trade log."""

    id: str
    symbol: str
    side: str
    qty: str
    notional: str
    status: str
    client_order_id: str
    filled_avg_price: str


class LivePaperBroker:
    """Broker-compatible simulator using a local JSON account ledger.

    Market data and market-clock checks still come from Alpaca. Order state,
    cash, and position quantity are local only.
    """

    def __init__(self, settings: Settings, logger: Logger) -> None:
        if not settings.paper_trading:
            raise ValueError("Paper trading must be enabled. Live trading is blocked.")

        self.settings = settings
        self.logger = logger
        self.state_path = settings.live_paper_state_path
        self.clock_client = TradingClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            paper=True,
        )
        self.state = self._load_state()

    def is_market_open(self) -> bool:
        """Return True only when Alpaca says the market is open."""
        try:
            clock = self.clock_client.get_clock()
        except (APIError, RequestException) as error:
            raise BrokerError(f"Failed to fetch Alpaca market clock: {error}") from error

        self.logger.info("Market timestamp: %s", clock.timestamp)
        self.logger.info("Market is open: %s", clock.is_open)
        self.logger.info("Next market open: %s", clock.next_open)
        self.logger.info("Next market close: %s", clock.next_close)
        return bool(clock.is_open)

    def mark_to_market(self, latest_price: float) -> None:
        """Update local equity calculations to the latest live close."""
        price = Decimal(str(latest_price))
        if price <= 0:
            raise BrokerError("Cannot mark live-paper account with a non-positive price.")

        self.state["last_price"] = _format_decimal(price)
        self._save_state()
        self.logger.info(
            "Live-paper mark: %s last price set to $%s.",
            self.settings.symbol,
            _format_money(price),
        )

    def get_account(self) -> LivePaperAccount:
        """Return a local no-margin paper account."""
        cash = self._cash()
        equity = cash + self._position_value()
        account = LivePaperAccount(
            status="ACTIVE",
            trading_blocked=False,
            buying_power=_format_money(cash),
            cash=_format_money(cash),
            equity=_format_money(equity),
        )
        self.logger.info("Live-paper account status: %s", account.status)
        self.logger.info("Live-paper trading blocked: %s", account.trading_blocked)
        self.logger.info("Live-paper buying power: %s", account.buying_power)
        self.logger.info("Live-paper cash: %s", account.cash)
        self.logger.info("Live-paper equity: %s", account.equity)
        return account

    def get_position_qty(self, symbol: str) -> float:
        """Return the current simulated long quantity for a symbol."""
        self._validate_symbol(symbol)
        return float(self._position_qty())

    def has_open_order(self, symbol: str) -> bool:
        """Live-paper orders fill immediately, so there are no open orders."""
        self._validate_symbol(symbol)
        return False

    def count_orders_submitted_today(self, symbol: str) -> int:
        """Count local live-paper fills during the current New York market date."""
        self._validate_symbol(symbol)
        today = datetime.now(timezone.utc).astimezone(NEW_YORK_TIME).date()
        count = 0

        for order in self.state.get("orders", []):
            if str(order.get("symbol", "")).upper() != symbol.upper():
                continue

            submitted_at = _parse_utc_timestamp(str(order.get("submitted_at", "")))
            if submitted_at.astimezone(NEW_YORK_TIME).date() == today:
                count += 1

        self.logger.info("Live-paper orders submitted today for %s: %d", symbol, count)
        return count

    def submit_market_buy(self, symbol: str, notional: float, client_order_id: str) -> LivePaperOrder:
        """Simulate an immediately filled market buy by notional value."""
        self._validate_symbol(symbol)
        fill_price = self._require_last_price()
        requested_notional = Decimal(str(notional)).quantize(CENT, rounding=ROUND_DOWN)
        cash = self._cash()

        if requested_notional <= 0:
            raise BrokerError("Live-paper buy notional must be positive.")

        if requested_notional > cash:
            raise BrokerError(
                f"Live-paper buy notional ${requested_notional:.2f} exceeds cash ${cash:.2f}."
            )

        qty = (requested_notional / fill_price).quantize(SHARE_QUANTITY, rounding=ROUND_DOWN)
        if qty <= 0:
            raise BrokerError("Live-paper buy quantity rounds down to zero.")

        fill_notional = (qty * fill_price).quantize(CENT, rounding=ROUND_DOWN)
        self.state["cash"] = _format_money(cash - fill_notional)
        self.state["position_qty"] = _format_qty(self._position_qty() + qty)

        order = self._new_order(
            symbol=symbol,
            side="BUY",
            qty=qty,
            notional=fill_notional,
            fill_price=fill_price,
            client_order_id=client_order_id,
        )
        self._record_order(order)

        self.logger.warning(
            "LIVE PAPER FILL: simulated BUY for %s qty=%s notional=$%s price=$%s.",
            symbol,
            order.qty,
            order.notional,
            order.filled_avg_price,
        )
        return order

    def submit_market_sell(self, symbol: str, qty: float, client_order_id: str) -> LivePaperOrder:
        """Simulate an immediately filled market sell for the held quantity."""
        self._validate_symbol(symbol)
        fill_price = self._require_last_price()
        requested_qty = Decimal(str(qty)).quantize(SHARE_QUANTITY, rounding=ROUND_DOWN)
        held_qty = self._position_qty()

        if requested_qty <= 0:
            raise BrokerError("Live-paper sell quantity must be positive.")

        if requested_qty > held_qty:
            raise BrokerError(
                f"Live-paper sell quantity {requested_qty} exceeds held quantity {held_qty}."
            )

        fill_notional = (requested_qty * fill_price).quantize(CENT, rounding=ROUND_DOWN)
        self.state["cash"] = _format_money(self._cash() + fill_notional)
        self.state["position_qty"] = _format_qty(held_qty - requested_qty)

        order = self._new_order(
            symbol=symbol,
            side="SELL",
            qty=requested_qty,
            notional=fill_notional,
            fill_price=fill_price,
            client_order_id=client_order_id,
        )
        self._record_order(order)

        self.logger.warning(
            "LIVE PAPER FILL: simulated SELL for %s qty=%s notional=$%s price=$%s.",
            symbol,
            order.qty,
            order.notional,
            order.filled_avg_price,
        )
        return order

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BrokerError(f"Failed to read live-paper state file: {error}") from error
        else:
            loaded = {}

        state = {
            "cash": _format_money(_decimal_value(loaded.get("cash"), self.settings.live_paper_initial_cash)),
            "position_qty": _format_qty(_decimal_value(loaded.get("position_qty"), 0)),
            "last_price": loaded.get("last_price") or "",
            "orders": loaded.get("orders") if isinstance(loaded.get("orders"), list) else [],
        }
        self._write_state(state)
        return state

    def _save_state(self) -> None:
        self._write_state(self.state)

    def _write_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _cash(self) -> Decimal:
        return _decimal_value(self.state.get("cash"), self.settings.live_paper_initial_cash)

    def _position_qty(self) -> Decimal:
        return _decimal_value(self.state.get("position_qty"), 0)

    def _last_price(self) -> Decimal:
        return _decimal_value(self.state.get("last_price"), 0)

    def _position_value(self) -> Decimal:
        last_price = self._last_price()
        if last_price <= 0:
            return Decimal("0")
        return (self._position_qty() * last_price).quantize(CENT, rounding=ROUND_DOWN)

    def _require_last_price(self) -> Decimal:
        last_price = self._last_price()
        if last_price <= 0:
            raise BrokerError("Live-paper broker has no latest market price for fills.")
        return last_price

    def _new_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        notional: Decimal,
        fill_price: Decimal,
        client_order_id: str,
    ) -> LivePaperOrder:
        return LivePaperOrder(
            id=f"live-paper-{uuid4().hex}",
            symbol=symbol.upper(),
            side=side,
            qty=_format_qty(qty),
            notional=_format_money(notional),
            status="FILLED",
            client_order_id=client_order_id,
            filled_avg_price=_format_money(fill_price),
        )

    def _record_order(self, order: LivePaperOrder) -> None:
        orders = self.state.setdefault("orders", [])
        orders.append(
            {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "notional": order.notional,
                "status": order.status,
                "client_order_id": order.client_order_id,
                "filled_avg_price": order.filled_avg_price,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_state()

    def _validate_symbol(self, symbol: str) -> None:
        if symbol.upper() != self.settings.symbol:
            raise ValueError(f"Unexpected symbol {symbol}. This bot only trades {self.settings.symbol}.")


def _decimal_value(value, default: float | int | str) -> Decimal:
    if value in (None, ""):
        return Decimal(str(default))
    try:
        return Decimal(str(value))
    except Exception as error:  # noqa: BLE001 - convert malformed state into a broker error.
        raise BrokerError(f"Invalid decimal value in live-paper state: {value!r}") from error


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _format_money(value: Decimal) -> str:
    return f"{value.quantize(CENT):.2f}"


def _format_qty(value: Decimal) -> str:
    return f"{value.quantize(SHARE_QUANTITY, rounding=ROUND_DOWN):.6f}"
