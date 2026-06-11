"""Tkinter GUI for data-only backtests and strategy diagnostics."""

from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import pandas as pd

from backtester import plot_equity_curves, save_summary_csv
from services.backtest_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    BacktestRequest,
    StrategyComparisonRequest,
    run_single_backtest,
    run_strategy_comparison,
)
from services.trade_analysis_service import (
    DEFAULT_SHORT_TRADE_DAYS,
    DEFAULT_SUSPICIOUS_MOVE_THRESHOLD,
    DEFAULT_TOP_DAYS,
    TradeAnalysisRequest,
    run_trade_analysis,
)
from strategies.registry import available_strategy_names
from trade_diagnostics import AnalysisFiles, save_reports


BOT_DIR = Path(__file__).resolve().parent
BACKTEST_PLOT_FILE_NAME = "backtest_portfolio_value.png"
COMPARISON_RESULTS_FILE_NAME = "backtest_results.csv"
COMPARISON_PLOT_FILE_NAME = "backtest_equity_curves.png"


class PlutoTraderGui(tk.Tk):
    """Small desktop UI for safe, data-only evaluation workflows."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Pluto Trader Evaluation")
        self.geometry("1120x760")
        self.minsize(940, 620)

        self._messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self._buttons: list[ttk.Button] = []

        self.strategy_names = available_strategy_names()
        self.strategy_var = tk.StringVar(value="ma_crossover")
        self.strategies_var = tk.StringVar(value=" ".join(self.strategy_names))
        self.start_var = tk.StringVar(value=DEFAULT_START_DATE)
        self.end_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="2022-01-01")
        self.initial_cash_var = tk.StringVar(value=f"{DEFAULT_INITIAL_CASH:.2f}")
        self.exposure_var = tk.StringVar(value=f"{DEFAULT_EXPOSURE:.2f}")
        self.transaction_cost_var = tk.StringVar(value="1")
        self.slippage_var = tk.StringVar(value="2")
        self.top_days_var = tk.StringVar(value=str(DEFAULT_TOP_DAYS))
        self.short_trade_days_var = tk.StringVar(value=str(DEFAULT_SHORT_TRADE_DAYS))
        self.suspicious_move_var = tk.StringVar(value=str(DEFAULT_SUSPICIOUS_MOVE_THRESHOLD))
        self.status_var = tk.StringVar(value="Ready. Data-only tools; no orders can be submitted from this GUI.")

        self._build_layout()
        self.after(100, self._poll_messages)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=16)
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Pluto Trader", font=("Segoe UI", 18, "bold")).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 4),
        )
        ttk.Label(
            controls,
            text="Backtests and diagnostics only. Live trading is not exposed here.",
            wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 18))

        row = 2
        row = self._add_strategy_fields(controls, row)
        row = self._add_backtest_fields(controls, row)
        row = self._add_analysis_fields(controls, row)

        actions = ttk.LabelFrame(controls, text="Actions", padding=12)
        actions.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        actions.columnconfigure(0, weight=1)

        self._add_button(actions, "Run Single Backtest", self._run_backtest).grid(row=0, column=0, sticky="ew", pady=3)
        self._add_button(actions, "Compare Strategies", self._run_comparison).grid(row=1, column=0, sticky="ew", pady=3)
        self._add_button(actions, "Analyze Trades", self._run_trade_analysis).grid(row=2, column=0, sticky="ew", pady=3)
        self._add_button(actions, "Clear Output", self._clear_output).grid(row=3, column=0, sticky="ew", pady=(12, 3))

        right = ttk.Frame(self, padding=(0, 16, 16, 16))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, textvariable=self.status_var).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.output = scrolledtext.ScrolledText(right, wrap=tk.NONE, font=("Consolas", 10))
        self.output.grid(row=1, column=0, sticky="nsew")
        self.output.insert(
            tk.END,
            "Choose an action on the left. Results, saved file paths, and errors appear here.\n",
        )
        self.output.configure(state=tk.DISABLED)

    def _add_strategy_fields(self, parent: ttk.Frame, row: int) -> int:
        frame = ttk.LabelFrame(parent, text="Strategies", padding=12)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Single strategy").grid(row=0, column=0, sticky="w", pady=3)
        strategy_box = ttk.Combobox(
            frame,
            textvariable=self.strategy_var,
            values=self.strategy_names,
            state="readonly",
        )
        strategy_box.grid(row=0, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Compare list").grid(row=1, column=0, sticky="nw", pady=3)
        ttk.Entry(frame, textvariable=self.strategies_var, width=34).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(
            frame,
            text="Space-separated strategy names.",
            foreground="#555555",
        ).grid(row=2, column=1, sticky="w", pady=(0, 3))
        return row + 1

    def _add_backtest_fields(self, parent: ttk.Frame, row: int) -> int:
        frame = ttk.LabelFrame(parent, text="Backtest Inputs", padding=12)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        frame.columnconfigure(1, weight=1)

        fields = [
            ("Start date", self.start_var),
            ("End date", self.end_var),
            ("Split date", self.split_var),
            ("Initial cash", self.initial_cash_var),
            ("Exposure", self.exposure_var),
            ("Transaction bps", self.transaction_cost_var),
            ("Slippage bps", self.slippage_var),
        ]

        for index, (label, variable) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
            ttk.Entry(frame, textvariable=variable, width=18).grid(row=index, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Leave end or split date blank to skip it.", foreground="#555555").grid(
            row=len(fields),
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 0),
        )
        return row + 1

    def _add_analysis_fields(self, parent: ttk.Frame, row: int) -> int:
        frame = ttk.LabelFrame(parent, text="Trade Analysis", padding=12)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        frame.columnconfigure(1, weight=1)

        fields = [
            ("Top up days", self.top_days_var),
            ("Whipsaw days", self.short_trade_days_var),
            ("Suspicious move", self.suspicious_move_var),
        ]
        for index, (label, variable) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
            ttk.Entry(frame, textvariable=variable, width=18).grid(row=index, column=1, sticky="ew", pady=3)

        return row + 1

    def _add_button(self, parent: ttk.Frame, text: str, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self._buttons.append(button)
        return button

    def _run_backtest(self) -> None:
        try:
            request = self._backtest_request()
        except ValueError as error:
            self._show_validation_error(error)
            return

        self._run_in_background(
            "Running single backtest...",
            lambda: self._format_backtest_result(run_single_backtest(request)),
        )

    def _run_comparison(self) -> None:
        try:
            request = self._comparison_request()
        except ValueError as error:
            self._show_validation_error(error)
            return

        self._run_in_background(
            "Comparing strategies...",
            lambda: self._format_comparison_result(run_strategy_comparison(request)),
        )

    def _run_trade_analysis(self) -> None:
        try:
            request = self._trade_analysis_request()
        except ValueError as error:
            self._show_validation_error(error)
            return

        self._run_in_background(
            "Analyzing trades...",
            lambda: self._format_trade_analysis_result(run_trade_analysis(request)),
        )

    def _run_in_background(self, status: str, worker) -> None:
        self._set_buttons_enabled(False)
        self.status_var.set(status)
        self._write_output(f"{status}\n")

        def target() -> None:
            try:
                self._messages.put(("result", worker()))
            except Exception as error:  # noqa: BLE001 - GUI must surface unexpected failures cleanly.
                details = "".join(traceback.format_exception_only(type(error), error)).strip()
                self._messages.put(("error", details))

        threading.Thread(target=target, daemon=True).start()

    def _poll_messages(self) -> None:
        try:
            kind, message = self._messages.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_messages)
            return

        if kind == "result":
            self._write_output(message)
            self.status_var.set("Ready.")
        else:
            self._write_output(f"ERROR: {message}\n")
            self.status_var.set("Failed. Check the output pane.")
            messagebox.showerror("Pluto Trader", message)

        self._set_buttons_enabled(True)
        self.after(100, self._poll_messages)

    def _backtest_request(self) -> BacktestRequest:
        return BacktestRequest(
            strategy_name=self.strategy_var.get(),
            start=self.start_var.get().strip(),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
        )

    def _comparison_request(self) -> StrategyComparisonRequest:
        strategy_names = self.strategies_var.get().split()
        if not strategy_names:
            raise ValueError("Compare list must include at least one strategy.")

        return StrategyComparisonRequest(
            strategy_names=strategy_names,
            start=self.start_var.get().strip(),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            split_date=_optional_text(self.split_var),
        )

    def _trade_analysis_request(self) -> TradeAnalysisRequest:
        return TradeAnalysisRequest(
            strategy_name=self.strategy_var.get(),
            start=self.start_var.get().strip(),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            top_days=_int_value(self.top_days_var, "Top up days"),
            short_trade_days=_int_value(self.short_trade_days_var, "Whipsaw days"),
            suspicious_move_threshold=_float_value(self.suspicious_move_var, "Suspicious move"),
        )

    def _format_backtest_result(self, result) -> str:
        plot_path = BOT_DIR / BACKTEST_PLOT_FILE_NAME
        plot_equity_curves([result.result], path=plot_path, show_plot=False)

        return (
            "\nSingle Backtest\n"
            "===============\n"
            f"Symbol: {result.settings.symbol}\n"
            f"Strategy: {result.result.display_name}\n"
            f"Start: {result.start_date.date()}\n"
            f"End: {result.end_date.date() if result.end_date else 'latest available'}\n"
            f"Saved plot: {plot_path}\n\n"
            f"{_format_summary_frame(result.summary)}\n"
        )

    def _format_comparison_result(self, result) -> str:
        results_path = BOT_DIR / COMPARISON_RESULTS_FILE_NAME
        plot_path = BOT_DIR / COMPARISON_PLOT_FILE_NAME
        save_summary_csv(result.full_summary, results_path)
        plot_equity_curves(result.full_results, path=plot_path, show_plot=False)

        chunks = [
            "\nStrategy Comparison\n",
            "===================\n",
            f"Symbol: {result.settings.symbol}\n",
            f"Strategies: {', '.join(result.strategy_names)}\n",
            f"Start: {result.start_date.date()}\n",
            f"End: {result.end_date.date() if result.end_date else 'latest available'}\n",
            f"Saved CSV: {results_path}\n",
            f"Saved plot: {plot_path}\n\n",
            "Full Period\n",
            "-----------\n",
            _format_summary_frame(result.full_summary),
            "\n",
        ]

        if result.split_date:
            if result.train_summary is None or result.test_summary is None:
                chunks.append("\nTrain/test split skipped because the split date leaves one side empty.\n")
            else:
                chunks.extend(
                    [
                        f"\nTrain Through {result.split_date.date()}\n",
                        "----------------\n",
                        _format_summary_frame(result.train_summary),
                        f"\n\nTest After {result.split_date.date()}\n",
                        "----------\n",
                        _format_summary_frame(result.test_summary),
                        "\n",
                    ]
                )

        return "".join(chunks)

    def _format_trade_analysis_result(self, result) -> str:
        files = AnalysisFiles(
            trades=BOT_DIR / f"trade_analysis_{result.strategy_name}_trades.csv",
            missed_days=BOT_DIR / f"trade_analysis_{result.strategy_name}_missed_best_days.csv",
            whipsaws=BOT_DIR / f"trade_analysis_{result.strategy_name}_whipsaws.csv",
            suspicious_moves=BOT_DIR / f"trade_analysis_{result.strategy_name}_suspicious_moves.csv",
        )
        save_reports(
            trades=result.trades,
            missed_days=result.missed_days,
            whipsaws=result.whipsaws,
            suspicious_moves=result.suspicious_moves,
            files=files,
        )

        return (
            "\nTrade Analysis\n"
            "==============\n"
            f"Symbol: {result.settings.symbol}\n"
            f"Strategy: {result.result.display_name}\n"
            f"Saved trades: {files.trades}\n"
            f"Saved missed days: {files.missed_days}\n"
            f"Saved whipsaws: {files.whipsaws}\n"
            f"Saved suspicious moves: {files.suspicious_moves}\n\n"
            f"{_format_trade_summary(result)}"
        )

    def _write_output(self, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        if not text.endswith("\n"):
            self.output.insert(tk.END, "\n")
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _clear_output(self) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.configure(state=tk.DISABLED)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self._buttons:
            button.configure(state=state)

    def _show_validation_error(self, error: ValueError) -> None:
        message = str(error)
        self.status_var.set("Input error.")
        self._write_output(f"INPUT ERROR: {message}\n")
        messagebox.showerror("Invalid input", message)


def _format_summary_frame(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No results.\n"

    display = summary.copy()
    percent_columns = [
        "exposure",
        "total_return",
        "buy_hold_same_exposure_return",
        "buy_hold_full_return",
        "buy_hold_return",
        "cagr",
        "max_drawdown",
        "win_rate",
        "average_trade_return",
        "time_in_market",
        "average_position",
    ]
    money_columns = ["final_value"]
    integer_columns = ["trades", "missed_top_20_up_days"]

    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.2%}")

    for column in money_columns:
        if column in display:
            display[column] = display[column].map(lambda value: f"${value:,.2f}")

    for column in integer_columns:
        if column in display:
            display[column] = display[column].map(lambda value: f"{int(value)}")

    if "sharpe" in display:
        display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")

    return display.to_string(index=False) + "\n"


def _format_trade_summary(result) -> str:
    data = result.result.data
    total_return = (float(data["portfolio_value"].iloc[-1]) / float(data["portfolio_value"].iloc[0])) - 1
    same_exposure_return = (
        float(data["buy_hold_same_exposure_value"].iloc[-1])
        / float(data["buy_hold_same_exposure_value"].iloc[0])
    ) - 1
    missed_count = int(result.missed_days["missed_by_strategy"].sum()) if not result.missed_days.empty else 0

    if result.trades.empty:
        completed_trades = result.trades
        has_open_trade = False
    else:
        open_trade = result.trades["open_trade"].fillna(False).astype(bool)
        completed_trades = result.trades[~open_trade]
        has_open_trade = bool(open_trade.any())

    lines = [
        "Summary\n",
        "-------\n",
        f"Strategy return:            {total_return:.2%}\n",
        f"Same-exposure buy-hold:     {same_exposure_return:.2%}\n",
        f"Completed trades:           {len(completed_trades)}\n",
        f"Open trade at end:          {has_open_trade}\n",
        f"Missed top up days:         {missed_count}/{len(result.missed_days)}\n",
        f"Short losing trades:        {len(result.whipsaws)}\n",
        f"Suspicious daily moves:     {len(result.suspicious_moves)}\n",
    ]

    if not completed_trades.empty:
        lines.extend(
            [
                f"Best trade:                 {completed_trades['trade_return'].max():.2%}\n",
                f"Worst trade:                {completed_trades['trade_return'].min():.2%}\n",
                f"Average days held:          {completed_trades['days_held'].mean():.1f}\n",
            ]
        )

    lines.extend(
        [
            "\nWorst Trades\n",
            "------------\n",
            _format_frame_or_none(completed_trades.sort_values("trade_return").head(5)),
            "\nMissed Best SPY Days\n",
            "--------------------\n",
            _format_frame_or_none(result.missed_days[result.missed_days["missed_by_strategy"]].head(5)),
            "\nWorst Short Whipsaws\n",
            "--------------------\n",
            _format_frame_or_none(result.whipsaws.head(5)),
            "\nSuspicious Daily Moves\n",
            "----------------------\n",
            _format_frame_or_none(result.suspicious_moves.head(5)),
        ]
    )
    return "".join(lines)


def _format_frame_or_none(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None\n"
    return frame.to_string(index=False, max_cols=8) + "\n"


def _float_value(variable: tk.StringVar, label: str) -> float:
    raw_value = variable.get().strip()
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{label} must be a number.") from error


def _int_value(variable: tk.StringVar, label: str) -> int:
    raw_value = variable.get().strip()
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer.") from error


def _optional_text(variable: tk.StringVar) -> str | None:
    value = variable.get().strip()
    return value or None


def main() -> int:
    app = PlutoTraderGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
