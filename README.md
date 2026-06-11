# Pluto Trader

Beginner-friendly Alpaca paper-trading bot for SPY. Live trading is not enabled.

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

The GUI can run single backtests, compare strategies, and generate trade-analysis CSVs. It is intentionally data-only and does not expose live or paper order submission.

## Module Layout

- `trading_bot\services\backtest_service.py` contains reusable single-backtest and strategy-comparison workflows.
- `trading_bot\services\trade_analysis_service.py` contains reusable trade-diagnostic workflow logic.
- `trading_bot\trade_diagnostics.py` builds trade, missed-day, whipsaw, and suspicious-move reports.
- CLI files such as `backtest.py`, `compare_strategies.py`, and `trade_analysis.py` are now thin wrappers around those services.

## Safety Defaults

- The broker is hard-coded to Alpaca paper trading.
- The bot refuses to run unless `PAPER_TRADING=True` and `ALPACA_PAPER=True`.
- The symbol is locked to `SPY`.
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
buy_and_hold
```

`core_tactical_ma` can use fractional target exposure. The live bot now rebalances toward that target by buying or trimming only the difference.
