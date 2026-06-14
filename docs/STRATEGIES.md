# Strategy Authoring Guide

Pluto Trader discovers strategies from Python files in `trading_bot/strategies/`.

## How Discovery Works

On startup, the GUI, CLI tools, preflight, and paper-trading entry point scan `trading_bot/strategies/` for classes that:

- subclass `trading_bot.strategies.base.Strategy`
- are defined in a non-private module, such as `my_strategy.py`
- are not in a module whose file name starts with `_`
- define a unique `name`, such as `name = "my_strategy"`

Restart the GUI after adding or editing a strategy file. Broken strategy files are skipped so the app can still open; the GUI logs skipped modules in the Run Log tab.

## Editable Parameters

Constructor arguments become editable GUI fields when they have default values.

```python
def __init__(self, lookback: int = 20, threshold: float = 0.03, enabled: bool = True) -> None:
    ...
```

Supported editable types are `int`, `float`, `bool`, and `str`. Add type hints for clarity. Required constructor arguments are ignored because the GUI cannot safely invent values for them.

Add optional `parameter_metadata` to improve field labels, descriptions, and validation:

```python
parameter_metadata = {
    "lookback": {
        "label": "Lookback",
        "minimum": 2,
        "description": "Number of bars used for the rolling high.",
    },
    "threshold": {
        "label": "Drop threshold",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Required pullback from the recent high; 0.03 means 3%.",
    },
}
```

If you need full control, define a classmethod named `parameter_specs` that returns `StrategyParameterSpec` objects from `trading_bot.strategies.registry`.

## Required Methods

Every strategy must implement:

- `generate_signals(price_data)`: returns a pandas `Series` indexed like `price_data`; values are target exposure from `0.0` to `1.0`.
- `make_decision(price_data, owns_position, logger=None)`: returns `StrategyDecision` with `BUY`, `SELL`, or `HOLD`.

The backtester uses `generate_signals`. The paper-trading decision loop uses `make_decision`.

## Minimal Example

Create `trading_bot/strategies/my_pullback.py`:

```python
from __future__ import annotations

from logging import Logger

import pandas as pd

from trading_bot.strategies.base import Strategy, StrategyDecision, TradeAction


class MyPullbackStrategy(Strategy):
    """Buy after a pullback from a recent high."""

    name = "my_pullback"
    parameter_metadata = {
        "lookback": {
            "label": "Lookback",
            "minimum": 2,
            "description": "Number of bars used for the rolling high.",
        },
        "drop_threshold": {
            "label": "Drop threshold",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Required pullback from the recent high; 0.03 means 3%.",
        },
    }

    def __init__(self, lookback: int = 20, drop_threshold: float = 0.03) -> None:
        self.lookback = lookback
        self.drop_threshold = drop_threshold
        self.display_name = f"{lookback}-bar pullback {drop_threshold:.1%}"

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return pd.Series(dtype=float)

        close = price_data["close"].dropna()
        recent_high = close.rolling(self.lookback).max()
        pulled_back = close <= recent_high * (1 - self.drop_threshold)

        signal = pd.Series(0.0, index=close.index)
        signal.loc[pulled_back.fillna(False)] = 1.0
        return signal.reindex(price_data.index).fillna(0.0)

    def make_decision(
        self,
        price_data: pd.DataFrame,
        owns_position: bool,
        logger: Logger | None = None,
    ) -> StrategyDecision:
        if price_data is None or price_data.empty:
            return self._empty_data_decision()
        if "close" not in price_data.columns:
            return self._missing_close_decision()

        signals = self.generate_signals(price_data)
        if signals.empty:
            return self._empty_data_decision()

        latest_close = float(price_data["close"].dropna().iloc[-1])
        wants_position = bool(signals.iloc[-1] > 0)

        if wants_position and not owns_position:
            return StrategyDecision(
                action=TradeAction.BUY,
                reason="Pullback condition is active and no position is currently held.",
                latest_close=latest_close,
            )

        if not wants_position and owns_position:
            return StrategyDecision(
                action=TradeAction.SELL,
                reason="Pullback condition is no longer active.",
                latest_close=latest_close,
            )

        return StrategyDecision(
            action=TradeAction.HOLD,
            reason="Pullback signal does not require a position change.",
            latest_close=latest_close,
        )
```

After restart, `my_pullback` appears in the strategy checkbox list, primary-strategy dropdown, CLI help text, and `.env` `STRATEGY` validation. Its `lookback` and `drop_threshold` fields appear in the Strategy Parameters editor.

Run `Validate Strategies` in the GUI after adding or editing a strategy. It checks discovery, editable parameters, `generate_signals`, and `make_decision` using generated sample OHLC data without contacting Alpaca.

## Common Failures

- Duplicate `name` values are not allowed; the later duplicate is skipped and logged.
- Files named `_example.py` or classes not subclassing `Strategy` are ignored.
- Import-time errors hide that file from the strategy list and are logged in the GUI.
- Constructor parameters without defaults are not shown in the GUI.
- Complex parameter types such as lists and dictionaries are not editable by the default GUI editor.
