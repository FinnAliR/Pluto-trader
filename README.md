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
python trading_bot\main.py
```

## Normal Workflow

1. Run `python trading_bot\preflight.py` before using the bot.
2. Backtest the configured strategy with `python trading_bot\backtest.py`.
3. Compare all strategies with `python trading_bot\compare_strategies.py`.
4. Run one paper-trading decision with `python trading_bot\main.py`.

Backtest plots are saved as PNG files by default. Add `--show` if you want a chart window to open.

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
