# Pluto Trader

Alpaca paper-trading bot for SPY. Live trading is not enabled.

## Quick Start

Run these commands from the repo root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r trading_bot\requirements.txt
Copy-Item trading_bot\.env.example trading_bot\.env
```

Edit `trading_bot\.env` and add Alpaca paper-trading keys.

```powershell
python trading_bot\preflight.py
python trading_bot\backtest.py --strategy ma_crossover
python trading_bot\compare_strategies.py
python trading_bot\gui.py
python trading_bot\main.py
```

## Normal Workflow

1. Run `python trading_bot\preflight.py` before using the bot.
2. Backtest the configured strategy with `python trading_bot\backtest.py`.
3. Compare all strategies with `python trading_bot\compare_strategies.py`.
4. Optionally open the data-only GUI with `python trading_bot\gui.py`.
5. Run one paper-trading decision with `python trading_bot\main.py`.

Backtest plots are saved as PNG files by default. Add `--show` if you want a chart window to open.

## GUI

Run the desktop evaluation GUI:

```powershell
python trading_bot\gui.py
```

Equivalent package entry point:

```powershell
python -m trading_bot.ui.gui
```

The GUI can run selected strategies, compare strategies, and generate trade-analysis CSVs. It is intentionally data-only and does not expose live or paper order submission.

GUI features:

- Scrollable control sidebar so action and save buttons remain reachable on smaller screens.
- Embedded equity-curve chart with matplotlib navigation controls.
- Drawdown subplot below the equity curve.
- Mouse-wheel zoom, recent-year focus, full-range reset, and clickable data-point annotations.
- Save buttons for plot images, summary CSVs, diagnostic CSVs, and the run log.
- Strategy checklist drives the main run button: one checked strategy runs a single backtest, while multiple checked strategies run a comparison.
- Strategy Parameters section lets each strategy keep its own editable parameter set for backtests, comparisons, regime scorecards, and diagnostics.
- Market dropdown lets data-only evaluations run on built-in symbols or any typed Alpaca-supported stock/ETF symbol; enter multiple comma-separated symbols and use Run Market Matrix to compare best strategies by market.
- Hover/focus descriptions are available on the main controls, tables, plot area, and action buttons.
- Date presets for full-history, crash, bear-market, and recovery tests.
- Regime scorecard that compares strategies across multiple market periods.
- Summary table with a practical score column.
- Diagnostics tabs for trades, missed best days, whipsaws, and suspicious daily moves.

## Module Layout

Root-level scripts such as `trading_bot\gui.py`, `backtest.py`, `compare_strategies.py`, `trade_analysis.py`, `preflight.py`, and `main.py` are compatibility wrappers. The implementation is organized into packages:

- `trading_bot\ui\` contains the Tkinter GUI.
- `trading_bot\cli\` contains command-line backtest, comparison, and trade-analysis entry points.
- `trading_bot\backtesting\` contains the reusable backtest engine, summary generation, and plot construction.
- `trading_bot\services\` contains reusable workflows shared by the CLI and GUI.
- `trading_bot\market_data\` contains Alpaca historical and live market-data helpers.
- `trading_bot\execution\` contains paper-trading broker, risk, live decision-cycle, and trade-log code.
- `trading_bot\diagnostics\` builds trade, missed-day, whipsaw, and suspicious-move reports.
- `trading_bot\core\` contains configuration and logging helpers.

You can keep using the root wrappers or run the package modules directly:

```powershell
python -m trading_bot.cli.backtest --strategy ma_crossover
python -m trading_bot.cli.compare_strategies
python -m trading_bot.cli.trade_analysis --strategy ma_crossover
python -m trading_bot.execution.preflight
python -m trading_bot.execution.main
```

Data-only CLI market override example:

```powershell
python -m trading_bot.cli.compare_strategies --symbol QQQ
```

## Safety Defaults

- The broker is hard-coded to Alpaca paper trading.
- The bot refuses to run unless `PAPER_TRADING=True` and `ALPACA_PAPER=True`.
- The symbol is locked to `SPY`.
- GUI/CLI backtests can override the historical-data symbol for research, but live/paper execution remains locked to the configured `SPY` safety check.
- New buys are capped by `BUYING_POWER_FRACTION` and `MAX_POSITION_SIZE_FRACTION`.
- `TRIAL_MODE=true` caps paper buy orders to `TRIAL_MAX_NOTIONAL`.

## Strategy Choices

Set `STRATEGY` in `trading_bot\.env`.

```text
ma_crossover
core_tactical_ma
rebound_reentry_ma
rsi
breakout
candle_pattern_jpy_session
buy_and_hold
```

`core_tactical_ma` can use fractional target exposure. The live bot now rebalances toward that target by buying or trimming only the difference.

`candle_pattern_jpy_session` is a long/flat adaptation of the referenced candle-pattern notebook. The original notebook used long and short stop orders in `backtesting.py`; this app's backtester is long-only, so bullish candle signals enter/hold exposure and bearish candle signals exit to cash.
