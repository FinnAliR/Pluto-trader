"""Risk controls that must pass before any order is submitted."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from logging import Logger

from config import Settings
from strategy import StrategyDecision, TradeAction


@dataclass(frozen=True)
class RiskResult:
    """Final risk approval or rejection."""

    approved: bool
    reason: str
    notional: float = 0.0
    qty: float = 0.0
    client_order_id: str = ""


class RiskManager:
    """Applies beginner-friendly risk rules before order submission."""

    def __init__(self, settings: Settings, logger: Logger) -> None:
        self.settings = settings
        self.logger = logger

    def evaluate(
        self,
        decision: StrategyDecision,
        account,
        position_qty: float,
        has_open_order: bool,
    ) -> RiskResult:
        """Approve or reject the proposed strategy action."""
        if decision.action == TradeAction.HOLD:
            return RiskResult(approved=False, reason="No trade action requested.")

        if account.trading_blocked:
            return RiskResult(approved=False, reason="Alpaca account is blocked from trading.")

        if has_open_order:
            return RiskResult(
                approved=False,
                reason="An open SPY order already exists, so a duplicate order is blocked.",
            )

        if decision.latest_close is None or decision.latest_close <= 0:
            return RiskResult(approved=False, reason="Latest price is missing or invalid.")

        if decision.action == TradeAction.BUY:
            return self._evaluate_buy(account=account, position_qty=position_qty)

        if decision.action == TradeAction.SELL:
            return self._evaluate_sell(position_qty=position_qty)

        return RiskResult(approved=False, reason=f"Unsupported action: {decision.action.value}")

    def _evaluate_buy(self, account, position_qty: float) -> RiskResult:
        if position_qty > 0:
            return RiskResult(
                approved=False,
                reason="SPY is already held. Only one open SPY position is allowed.",
            )

        buying_power = Decimal(str(account.buying_power))
        cash = Decimal(str(account.cash))

        if buying_power <= 0:
            return RiskResult(approved=False, reason="Buying power is not positive.")

        if cash <= 0:
            return RiskResult(approved=False, reason="Cash is not positive. No-leverage rule blocks buys.")

        # Buying power can include margin. Capping by cash enforces the no-leverage rule.
        usable_capital = min(buying_power, cash)
        notional = usable_capital * Decimal(str(self.settings.buying_power_fraction))
        notional = notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        self.logger.info("Risk: buying power=%s cash=%s usable capital=%s", buying_power, cash, usable_capital)
        self.logger.info("Risk: max trade notional at %.0f%% = $%s", self.settings.buying_power_fraction * 100, notional)

        if notional < Decimal(str(self.settings.min_order_notional)):
            return RiskResult(
                approved=False,
                reason=(
                    f"Trade notional ${notional} is below minimum "
                    f"${self.settings.min_order_notional:.2f}."
                ),
            )

        return RiskResult(
            approved=True,
            reason="Buy risk checks passed.",
            notional=float(notional),
            client_order_id=self._client_order_id(TradeAction.BUY),
        )

    def _evaluate_sell(self, position_qty: float) -> RiskResult:
        if position_qty <= 0:
            return RiskResult(
                approved=False,
                reason="No SPY position is held. Sell is blocked to prevent shorting.",
            )

        return RiskResult(
            approved=True,
            reason="Sell risk checks passed.",
            qty=position_qty,
            client_order_id=self._client_order_id(TradeAction.SELL),
        )

    def _client_order_id(self, action: TradeAction) -> str:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        timeframe = self.settings.timeframe.lower().replace(" ", "")
        return f"{self.settings.client_order_id_prefix}-{self.settings.symbol.lower()}-{action.value.lower()}-{timeframe}-{today}"
