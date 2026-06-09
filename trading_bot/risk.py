"""Risk controls that must pass before any order is submitted."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from logging import Logger

from config import Settings
from strategy import StrategyDecision, TradeAction


CENT = Decimal("0.01")
SHARE_QUANTITY = Decimal("0.000001")
MAX_CLIENT_ORDER_ID_LENGTH = 48


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
        submitted_trades_today: int,
    ) -> RiskResult:
        """Approve or reject the proposed strategy action."""
        if decision.action == TradeAction.HOLD:
            return RiskResult(approved=False, reason="No trade action requested.")

        if not self.settings.paper_trading:
            return RiskResult(approved=False, reason="PAPER_TRADING is not true.")

        if submitted_trades_today >= self.settings.max_daily_trades:
            return RiskResult(
                approved=False,
                reason=(
                    f"Daily trade limit reached: {submitted_trades_today}/"
                    f"{self.settings.max_daily_trades} submitted trades today."
                ),
            )

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
            return self._evaluate_buy(
                account=account,
                position_qty=position_qty,
                latest_close=decision.latest_close,
                target_fraction=decision.target_fraction,
            )

        if decision.action == TradeAction.SELL:
            return self._evaluate_sell(
                account=account,
                position_qty=position_qty,
                latest_close=decision.latest_close,
                target_fraction=decision.target_fraction,
            )

        return RiskResult(approved=False, reason=f"Unsupported action: {decision.action.value}")

    def _evaluate_buy(
        self,
        account,
        position_qty: float,
        latest_close: float,
        target_fraction: float | None,
    ) -> RiskResult:
        try:
            strategy_target_fraction = _target_fraction_or_default(target_fraction, default=1.0)
        except ValueError as error:
            return RiskResult(approved=False, reason=str(error))

        if strategy_target_fraction <= 0:
            return RiskResult(
                approved=False,
                reason="Buy target fraction is 0%, so no buy is needed.",
            )

        buying_power = Decimal(str(account.buying_power))
        cash = Decimal(str(account.cash))
        equity = Decimal(str(account.equity))
        latest_price = Decimal(str(latest_close))

        if buying_power <= 0:
            return RiskResult(approved=False, reason="Buying power is not positive.")

        if cash <= 0:
            return RiskResult(approved=False, reason="Cash is not positive. No-leverage rule blocks buys.")

        if equity <= 0:
            return RiskResult(approved=False, reason="Account equity is not positive.")

        # Buying power can include margin. Capping by cash enforces the no-leverage rule.
        usable_capital = min(buying_power, cash)
        trade_notional_cap = usable_capital * Decimal(str(self.settings.buying_power_fraction))
        max_position_value = equity * Decimal(str(self.settings.max_position_size_fraction))
        target_position_value = max_position_value * strategy_target_fraction
        current_position_value = Decimal(str(position_qty)) * latest_price
        notional_needed = target_position_value - current_position_value

        if notional_needed <= 0:
            return RiskResult(
                approved=False,
                reason=(
                    f"Target SPY allocation already reached. Current SPY value ${current_position_value:.2f} "
                    f"is at or above target ${target_position_value:.2f}."
                ),
            )

        notional = min(trade_notional_cap, notional_needed)

        if self.settings.trial_mode:
            trial_cap = Decimal(str(self.settings.trial_max_notional))
            notional = min(notional, trial_cap)
            self.logger.warning(
                "TRIAL MODE: buy notional capped at $%.2f before order submission.",
                trial_cap,
            )

        notional = notional.quantize(CENT, rounding=ROUND_DOWN)

        self.logger.info("Risk: buying power=%s cash=%s usable capital=%s", buying_power, cash, usable_capital)
        self.logger.info(
            "Risk: max trade notional at %.0f%% of usable capital = $%.2f",
            self.settings.buying_power_fraction * 100,
            trade_notional_cap,
        )
        self.logger.info(
            "Risk: max SPY position value at %.0f%% of equity = $%.2f",
            self.settings.max_position_size_fraction * 100,
            max_position_value,
        )
        self.logger.info("Risk: strategy target fraction = %.0f%%", strategy_target_fraction * 100)
        self.logger.info("Risk: target SPY position value = $%.2f", target_position_value)
        self.logger.info("Risk: current SPY position value = $%.2f", current_position_value)
        self.logger.info("Risk: buy notional needed to reach target = $%.2f", notional_needed)
        self.logger.info("Risk: approved buy notional after all caps = $%s", notional)

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

    def _evaluate_sell(
        self,
        account,
        position_qty: float,
        latest_close: float,
        target_fraction: float | None,
    ) -> RiskResult:
        if position_qty <= 0:
            return RiskResult(
                approved=False,
                reason="No SPY position is held. Sell is blocked to prevent shorting.",
            )

        try:
            strategy_target_fraction = _target_fraction_or_default(target_fraction, default=0.0)
        except ValueError as error:
            return RiskResult(approved=False, reason=str(error))

        equity = Decimal(str(account.equity))
        latest_price = Decimal(str(latest_close))
        current_qty = Decimal(str(position_qty))

        if equity <= 0:
            return RiskResult(approved=False, reason="Account equity is not positive.")

        max_position_value = equity * Decimal(str(self.settings.max_position_size_fraction))
        target_position_value = max_position_value * strategy_target_fraction
        current_position_value = current_qty * latest_price

        if strategy_target_fraction == 0:
            qty = current_qty.quantize(SHARE_QUANTITY, rounding=ROUND_DOWN)
            self.logger.info("Risk: full exit requested, approved sell quantity = %s", qty)
        else:
            notional_to_sell = current_position_value - target_position_value
            if notional_to_sell <= 0:
                return RiskResult(
                    approved=False,
                    reason=(
                        f"Target SPY allocation already reached. Current SPY value ${current_position_value:.2f} "
                        f"is at or below target ${target_position_value:.2f}."
                    ),
                )

            qty = min(current_qty, notional_to_sell / latest_price)
            qty = qty.quantize(SHARE_QUANTITY, rounding=ROUND_DOWN)

            if qty <= 0:
                return RiskResult(approved=False, reason="Calculated sell quantity rounds down to zero.")

            sell_notional = qty * latest_price
            if sell_notional < Decimal(str(self.settings.min_order_notional)):
                return RiskResult(
                    approved=False,
                    reason=(
                        f"Rebalance sell value ${sell_notional:.2f} is below minimum "
                        f"${self.settings.min_order_notional:.2f}."
                    ),
                )

            self.logger.info("Risk: strategy target fraction = %.0f%%", strategy_target_fraction * 100)
            self.logger.info("Risk: target SPY position value = $%.2f", target_position_value)
            self.logger.info("Risk: current SPY position value = $%.2f", current_position_value)
            self.logger.info("Risk: sell notional needed to reach target = $%.2f", notional_to_sell)
            self.logger.info("Risk: approved sell quantity after rounding = %s", qty)

        return RiskResult(
            approved=True,
            reason="Sell risk checks passed.",
            qty=float(qty),
            client_order_id=self._client_order_id(TradeAction.SELL),
        )

    def _client_order_id(self, action: TradeAction) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        timeframe = self.settings.timeframe.lower().replace(" ", "")
        suffix = f"{self.settings.symbol.lower()}-{action.value.lower()}-{timeframe}-{timestamp}"
        prefix_length = MAX_CLIENT_ORDER_ID_LENGTH - len(suffix) - 1
        prefix = self.settings.client_order_id_prefix[: max(prefix_length, 0)].strip("-")

        if not prefix:
            return suffix[:MAX_CLIENT_ORDER_ID_LENGTH]

        return f"{prefix}-{suffix}"[:MAX_CLIENT_ORDER_ID_LENGTH]


def _target_fraction_or_default(value: float | None, default: float) -> Decimal:
    raw_value = default if value is None else value
    target_fraction = Decimal(str(raw_value))

    if not Decimal("0") <= target_fraction <= Decimal("1"):
        raise ValueError(
            f"Strategy target fraction must be between 0 and 1, got {raw_value}."
        )

    return target_fraction
