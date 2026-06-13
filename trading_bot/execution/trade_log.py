"""CSV audit log for every buy/sell decision."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.execution.strategy import StrategyDecision


NEW_YORK_TIME = ZoneInfo("America/New_York")

FIELD_NAMES = [
    "timestamp_utc",
    "market_date",
    "symbol",
    "action",
    "status",
    "decision_reason",
    "risk_reason",
    "order_id",
    "order_status",
    "client_order_id",
    "qty",
    "notional",
    "position_qty_before",
    "latest_close",
    "fast_ma",
    "slow_ma",
    "indicator_name",
    "indicator_value",
    "target_fraction",
    "error_message",
]


@dataclass(frozen=True)
class TradeLogEntry:
    """One row in trades.csv."""

    symbol: str
    decision: StrategyDecision
    status: str
    risk_reason: str = ""
    order_id: str = ""
    order_status: str = ""
    client_order_id: str = ""
    qty: float = 0.0
    notional: float = 0.0
    position_qty_before: float = 0.0
    error_message: str = ""


class TradeLogger:
    """Writes and reads the beginner-friendly trades.csv file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._ensure_file_exists()

    def count_submitted_trades_today(self) -> int:
        """Count submitted orders for the current New York market date."""
        today = _market_date(datetime.now(timezone.utc))
        count = 0

        with self.path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("market_date") == today and row.get("status") == "SUBMITTED":
                    count += 1

        return count

    def record(self, entry: TradeLogEntry) -> None:
        """Append one buy/sell decision row to trades.csv."""
        now_utc = datetime.now(timezone.utc)

        row = {
            "timestamp_utc": now_utc.isoformat(),
            "market_date": _market_date(now_utc),
            "symbol": entry.symbol,
            "action": entry.decision.action.value,
            "status": entry.status,
            "decision_reason": entry.decision.reason,
            "risk_reason": entry.risk_reason,
            "order_id": entry.order_id,
            "order_status": entry.order_status,
            "client_order_id": entry.client_order_id,
            "qty": _format_number(entry.qty),
            "notional": _format_number(entry.notional),
            "position_qty_before": _format_number(entry.position_qty_before),
            "latest_close": _format_optional_number(entry.decision.latest_close),
            "fast_ma": _format_optional_number(entry.decision.fast_ma),
            "slow_ma": _format_optional_number(entry.decision.slow_ma),
            "indicator_name": entry.decision.indicator_name,
            "indicator_value": _format_optional_number(entry.decision.indicator_value),
            "target_fraction": _format_optional_number(entry.decision.target_fraction),
            "error_message": entry.error_message,
        }

        with self.path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
            writer.writerow(row)

    def _ensure_file_exists(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists() and self.path.stat().st_size > 0:
            self._upgrade_schema_if_needed()
            return

        with self.path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
            writer.writeheader()

    def _upgrade_schema_if_needed(self) -> None:
        with self.path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fields = reader.fieldnames or []
            rows = list(reader)

        if existing_fields == FIELD_NAMES:
            return

        if all(field in existing_fields for field in FIELD_NAMES):
            return

        with self.path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in FIELD_NAMES})


def _market_date(timestamp_utc: datetime) -> str:
    return timestamp_utc.astimezone(NEW_YORK_TIME).date().isoformat()


def _format_number(value: float) -> str:
    return f"{value:.6f}" if value else ""


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"

