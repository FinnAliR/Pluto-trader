"""Tkinter dashboard for data-only backtests and strategy diagnostics."""

from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from backtester import BacktestResult, create_equity_curve_figure, save_summary_csv
from services.backtest_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    BacktestRequest,
    StrategyComparisonRequest,
    StrategyComparisonResult,
    run_single_backtest,
    run_strategy_comparison,
)
from services.trade_analysis_service import (
    DEFAULT_SHORT_TRADE_DAYS,
    DEFAULT_SUSPICIOUS_MOVE_THRESHOLD,
    DEFAULT_TOP_DAYS,
    TradeAnalysisRequest,
    TradeAnalysisResult,
    run_trade_analysis,
)
from strategies.registry import available_strategy_names
from trade_diagnostics import AnalysisFiles, save_reports


BOT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DATE = "2022-01-01"

DATE_PRESETS = {
    "Custom": (DEFAULT_START_DATE, ""),
    "Full history": ("2018-01-01", ""),
    "COVID crash": ("2020-02-20", "2020-04-30"),
    "2022 bear market": ("2022-01-01", "2022-12-31"),
    "Recovery trend": ("2023-01-01", ""),
}

REGIME_PACK = [
    ("Full history", "2018-01-01", None),
    ("COVID crash", "2020-02-20", "2020-04-30"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("Recovery trend", "2023-01-01", None),
]

SUMMARY_COLUMN_LABELS = {
    "period": "Period",
    "score": "Score",
    "strategy": "Strategy",
    "display_name": "Display Name",
    "total_return": "Return",
    "buy_hold_same_exposure_return": "B&H Same",
    "buy_hold_full_return": "B&H Full",
    "excess_vs_same_exposure": "Excess",
    "cagr": "CAGR",
    "max_drawdown": "Max DD",
    "sharpe": "Sharpe",
    "trades": "Trades",
    "win_rate": "Win %",
    "time_in_market": "In Market",
    "average_position": "Avg Pos",
    "missed_top_20_up_days": "Miss20",
    "final_value": "Final Value",
}

SUMMARY_COLUMN_ORDER = [
    "period",
    "score",
    "strategy",
    "total_return",
    "buy_hold_same_exposure_return",
    "excess_vs_same_exposure",
    "cagr",
    "max_drawdown",
    "sharpe",
    "trades",
    "win_rate",
    "time_in_market",
    "missed_top_20_up_days",
    "final_value",
]

PERCENT_COLUMNS = {
    "exposure",
    "total_return",
    "buy_hold_same_exposure_return",
    "buy_hold_full_return",
    "buy_hold_return",
    "excess_vs_same_exposure",
    "cagr",
    "max_drawdown",
    "win_rate",
    "average_trade_return",
    "time_in_market",
    "average_position",
    "spy_daily_return",
    "strategy_position",
    "strategy_return",
    "trade_return",
}

MONEY_COLUMNS = {"final_value", "entry_close", "exit_close", "close"}
INTEGER_COLUMNS = {"trades", "missed_top_20_up_days", "trade_number", "days_held"}


@dataclass(frozen=True)
class TaskResult:
    """Result passed from the worker thread back to the Tk event loop."""

    kind: str
    payload: Any


class PlutoTraderGui(tk.Tk):
    """Desktop dashboard for safe, data-only evaluation workflows."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Pluto Trader Evaluation Dashboard")
        self.geometry("1360x860")
        self.minsize(960, 620)

        self._messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._action_buttons: list[ttk.Button] = []
        self._export_buttons: list[ttk.Button] = []

        self.strategy_names = available_strategy_names()
        self.strategy_var = tk.StringVar(value="ma_crossover")
        self.strategy_checks = {name: tk.BooleanVar(value=True) for name in self.strategy_names}
        self.preset_var = tk.StringVar(value="Full history")
        self.start_var = tk.StringVar(value=DEFAULT_START_DATE)
        self.end_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value=DEFAULT_SPLIT_DATE)
        self.initial_cash_var = tk.StringVar(value=f"{DEFAULT_INITIAL_CASH:.2f}")
        self.exposure_var = tk.StringVar(value=f"{DEFAULT_EXPOSURE:.2f}")
        self.transaction_cost_var = tk.StringVar(value="1")
        self.slippage_var = tk.StringVar(value="2")
        self.top_days_var = tk.StringVar(value=str(DEFAULT_TOP_DAYS))
        self.short_trade_days_var = tk.StringVar(value=str(DEFAULT_SHORT_TRADE_DAYS))
        self.suspicious_move_var = tk.StringVar(value=str(DEFAULT_SUSPICIOUS_MOVE_THRESHOLD))
        self.status_var = tk.StringVar(value="Ready. Data-only tools; no orders can be submitted from this GUI.")
        self.metric_vars = {
            "best": tk.StringVar(value="-"),
            "score": tk.StringVar(value="-"),
            "return": tk.StringVar(value="-"),
            "drawdown": tk.StringVar(value="-"),
            "sharpe": tk.StringVar(value="-"),
            "trades": tk.StringVar(value="-"),
        }

        self.current_results: list[BacktestResult] = []
        self.current_summary: pd.DataFrame | None = None
        self.current_analysis: TradeAnalysisResult | None = None
        self.current_figure: Figure | None = None
        self.plot_canvas: FigureCanvasTkAgg | None = None
        self.plot_toolbar: NavigationToolbar2Tk | None = None
        self.plot_annotation = None
        self.plot_marker = None
        self.plot_original_limits: dict[Any, tuple[tuple[float, float], tuple[float, float]]] = {}
        self.plot_connections: list[int] = []
        self.diagnostic_trees: dict[str, ttk.Frame] = {}

        self._configure_style()
        self._build_layout()
        self._set_export_buttons_enabled(False)
        self.after(100, self._poll_messages)

    def _configure_style(self) -> None:
        self.configure(background="#edf2f7")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#edf2f7")
        style.configure("Panel.TFrame", background="#f8fafc")
        style.configure("Side.TFrame", background="#111827")
        style.configure("Side.TLabel", background="#111827", foreground="#e5e7eb")
        style.configure("Muted.Side.TLabel", background="#111827", foreground="#9ca3af")
        style.configure("Title.Side.TLabel", background="#111827", foreground="#f8fafc", font=("Segoe UI Semibold", 22))
        style.configure("Metric.TFrame", background="#ffffff", relief="flat")
        style.configure("MetricName.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9))
        style.configure("MetricValue.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI Semibold", 16))
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(10, 7))
        style.configure("TButton", padding=(8, 5))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#0f172a")])

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar_shell = ttk.Frame(self, style="Side.TFrame")
        sidebar_shell.grid(row=0, column=0, sticky="ns")
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)
        sidebar = self._create_scrollable_sidebar(sidebar_shell)
        self._build_sidebar(sidebar)

        main = ttk.Frame(self, style="App.TFrame", padding=(14, 14, 14, 14))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        ttk.Label(main, textvariable=self.status_var, background="#edf2f7", foreground="#334155").grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._build_dashboard_tab()
        self._build_summary_tab()
        self._build_diagnostics_tab()
        self._build_log_tab()

    def _create_scrollable_sidebar(self, parent: ttk.Frame) -> ttk.Frame:
        canvas = tk.Canvas(parent, width=360, background="#111827", highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, style="Side.TFrame", padding=18)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        content.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_width)
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content.columnconfigure(0, weight=1)
        return content

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Pluto Trader", style="Title.Side.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Evaluate SPY strategies with backtests, scorecards, diagnostics, and exportable charts.",
            style="Muted.Side.TLabel",
            wraplength=300,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 18))

        strategy_frame = ttk.LabelFrame(parent, text="Strategies", padding=10)
        strategy_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        strategy_frame.columnconfigure(0, weight=1)
        strategy_frame.columnconfigure(1, weight=1)
        ttk.Label(strategy_frame, text="Single strategy").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            values=self.strategy_names,
            state="readonly",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 8))

        ttk.Label(strategy_frame, text="Compare set").grid(row=2, column=0, columnspan=2, sticky="w")
        for index, strategy_name in enumerate(self.strategy_names):
            ttk.Checkbutton(
                strategy_frame,
                text=strategy_name,
                variable=self.strategy_checks[strategy_name],
            ).grid(row=3 + index // 2, column=index % 2, sticky="w", pady=1)

        button_row = 3 + (len(self.strategy_names) + 1) // 2
        ttk.Button(strategy_frame, text="All", command=lambda: self._set_strategy_checks(True)).grid(
            row=button_row,
            column=0,
            sticky="ew",
            pady=(8, 0),
            padx=(0, 3),
        )
        ttk.Button(strategy_frame, text="None", command=lambda: self._set_strategy_checks(False)).grid(
            row=button_row,
            column=1,
            sticky="ew",
            pady=(8, 0),
            padx=(3, 0),
        )

        dates_frame = ttk.LabelFrame(parent, text="Dates", padding=10)
        dates_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        dates_frame.columnconfigure(1, weight=1)
        ttk.Label(dates_frame, text="Preset").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(
            dates_frame,
            textvariable=self.preset_var,
            values=list(DATE_PRESETS),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(dates_frame, text="Apply", command=self._apply_date_preset).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        self._add_labeled_entry(dates_frame, 1, "Start", self.start_var)
        self._add_labeled_entry(dates_frame, 2, "End", self.end_var)
        self._add_labeled_entry(dates_frame, 3, "Split", self.split_var)

        params_frame = ttk.LabelFrame(parent, text="Model Inputs", padding=10)
        params_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(params_frame, 0, "Initial cash", self.initial_cash_var)
        self._add_labeled_entry(params_frame, 1, "Exposure", self.exposure_var)
        self._add_labeled_entry(params_frame, 2, "Transaction bps", self.transaction_cost_var)
        self._add_labeled_entry(params_frame, 3, "Slippage bps", self.slippage_var)

        diagnostics_frame = ttk.LabelFrame(parent, text="Diagnostics", padding=10)
        diagnostics_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        diagnostics_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(diagnostics_frame, 0, "Top up days", self.top_days_var)
        self._add_labeled_entry(diagnostics_frame, 1, "Whipsaw days", self.short_trade_days_var)
        self._add_labeled_entry(diagnostics_frame, 2, "Large move", self.suspicious_move_var)

        actions_frame = ttk.LabelFrame(parent, text="Run", padding=10)
        actions_frame.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        actions_frame.columnconfigure(0, weight=1)
        self._add_action_button(actions_frame, "Run Single Backtest", self._run_backtest, primary=True).grid(row=0, column=0, sticky="ew", pady=3)
        self._add_action_button(actions_frame, "Compare Checked Strategies", self._run_comparison).grid(row=1, column=0, sticky="ew", pady=3)
        self._add_action_button(actions_frame, "Run Regime Scorecard", self._run_regime_pack).grid(row=2, column=0, sticky="ew", pady=3)
        self._add_action_button(actions_frame, "Analyze Single Strategy", self._run_trade_analysis).grid(row=3, column=0, sticky="ew", pady=3)

        exports_frame = ttk.LabelFrame(parent, text="Save", padding=10)
        exports_frame.grid(row=7, column=0, sticky="ew")
        exports_frame.columnconfigure(0, weight=1)
        self.save_plot_button = self._add_export_button(exports_frame, "Save Plot PNG", self._save_plot)
        self.save_plot_button.grid(row=0, column=0, sticky="ew", pady=3)
        self.save_summary_button = self._add_export_button(exports_frame, "Save Summary CSV", self._save_summary)
        self.save_summary_button.grid(row=1, column=0, sticky="ew", pady=3)
        self.save_reports_button = self._add_export_button(exports_frame, "Save Diagnostics CSVs", self._save_diagnostics)
        self.save_reports_button.grid(row=2, column=0, sticky="ew", pady=3)
        ttk.Button(exports_frame, text="Save Log", command=self._save_log).grid(row=3, column=0, sticky="ew", pady=(10, 3))

    def _build_dashboard_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Dashboard")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        metric_row = ttk.Frame(tab, style="App.TFrame")
        metric_row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for index in range(6):
            metric_row.columnconfigure(index, weight=1)

        cards = [
            ("Best Strategy", "best"),
            ("Score", "score"),
            ("Return", "return"),
            ("Max Drawdown", "drawdown"),
            ("Sharpe", "sharpe"),
            ("Trades", "trades"),
        ]
        for index, (label, key) in enumerate(cards):
            card = ttk.Frame(metric_row, style="Metric.TFrame", padding=12)
            card.grid(row=0, column=index, sticky="nsew", padx=4)
            ttk.Label(card, text=label, style="MetricName.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=self.metric_vars[key], style="MetricValue.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 0))

        plot_shell = ttk.Frame(tab, style="Panel.TFrame", padding=12)
        plot_shell.grid(row=1, column=0, sticky="nsew")
        plot_shell.columnconfigure(0, weight=1)
        plot_shell.rowconfigure(1, weight=1)

        header = ttk.Frame(plot_shell, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Equity Curve", background="#f8fafc", foreground="#0f172a", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Scroll to zoom, click a line to inspect values, or use the toolbar for pan/box zoom.",
            background="#f8fafc",
            foreground="#64748b",
        ).grid(row=1, column=0, sticky="w")
        ttk.Button(header, text="Last 1Y", command=lambda: self._focus_recent_plot(252)).grid(row=0, column=1, rowspan=2, padx=(8, 3))
        ttk.Button(header, text="Full Range", command=self._reset_plot_view).grid(row=0, column=2, rowspan=2, padx=3)
        ttk.Button(header, text="Clear Marker", command=self._clear_plot_marker).grid(row=0, column=3, rowspan=2, padx=(3, 0))

        self.plot_frame = ttk.Frame(plot_shell, style="Panel.TFrame")
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self.toolbar_frame = ttk.Frame(plot_shell, style="Panel.TFrame")
        self.toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._clear_plot()

    def _build_summary_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Summary")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        header = ttk.Frame(tab, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Strategy Scorecard", background="#edf2f7", foreground="#0f172a", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Score is a heuristic: excess return, Sharpe, drawdown, missed rallies, and turnover.",
            background="#edf2f7",
            foreground="#64748b",
        ).grid(row=1, column=0, sticky="w")

        self.summary_tree = self._create_tree(tab)
        self.summary_tree.grid(row=1, column=0, sticky="nsew")

    def _build_diagnostics_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Diagnostics")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        diagnostics_notebook = ttk.Notebook(tab)
        diagnostics_notebook.grid(row=0, column=0, sticky="nsew")

        for key, title in [
            ("trades", "Trades"),
            ("missed_days", "Missed Best Days"),
            ("whipsaws", "Whipsaws"),
            ("suspicious_moves", "Suspicious Moves"),
        ]:
            frame = ttk.Frame(diagnostics_notebook, style="App.TFrame", padding=8)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            diagnostics_notebook.add(frame, text=title)
            tree_container = self._create_tree(frame)
            tree_container.grid(row=0, column=0, sticky="nsew")
            self.diagnostic_trees[key] = tree_container

    def _build_log_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Run Log")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.output = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=("Consolas", 10), height=14)
        self.output.grid(row=0, column=0, sticky="nsew")
        self.output.insert(tk.END, "Ready. Results, save paths, and errors appear here.\n")
        self.output.configure(state=tk.DISABLED)

    def _add_labeled_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=1, sticky="ew", pady=3)

    def _add_action_button(self, parent: ttk.Frame, text: str, command: Callable[[], None], primary: bool = False) -> ttk.Button:
        style = "Primary.TButton" if primary else "TButton"
        button = ttk.Button(parent, text=text, command=command, style=style)
        self._action_buttons.append(button)
        return button

    def _add_export_button(self, parent: ttk.Frame, text: str, command: Callable[[], None]) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self._export_buttons.append(button)
        return button

    def _create_tree(self, parent: ttk.Frame) -> ttk.Frame:
        container = ttk.Frame(parent)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        tree = ttk.Treeview(container, show="headings")
        y_scroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        return container

    def _set_strategy_checks(self, checked: bool) -> None:
        for variable in self.strategy_checks.values():
            variable.set(checked)

    def _apply_date_preset(self) -> None:
        start, end = DATE_PRESETS[self.preset_var.get()]
        self.start_var.set(start)
        self.end_var.set(end)
        self._log(f"Applied date preset: {self.preset_var.get()} ({start} to {end or 'latest'}).")

    def _run_backtest(self) -> None:
        try:
            request = self._backtest_request()
        except ValueError as error:
            self._show_validation_error(error)
            return
        self._run_in_background("Running single backtest...", "backtest", lambda: run_single_backtest(request))

    def _run_comparison(self) -> None:
        try:
            request = self._comparison_request(include_split=True)
        except ValueError as error:
            self._show_validation_error(error)
            return
        self._run_in_background("Comparing checked strategies...", "comparison", lambda: run_strategy_comparison(request))

    def _run_regime_pack(self) -> None:
        try:
            base_request = self._comparison_request(include_split=False)
        except ValueError as error:
            self._show_validation_error(error)
            return
        self._run_in_background("Running regime scorecard...", "regime", lambda: self._run_regime_worker(base_request))

    def _run_trade_analysis(self) -> None:
        try:
            request = self._trade_analysis_request()
        except ValueError as error:
            self._show_validation_error(error)
            return
        self._run_in_background("Analyzing trades...", "analysis", lambda: run_trade_analysis(request))

    def _run_regime_worker(self, base_request: StrategyComparisonRequest) -> dict[str, Any]:
        summaries = []
        comparisons: list[StrategyComparisonResult] = []

        for period_name, start, end in REGIME_PACK:
            request = StrategyComparisonRequest(
                strategy_names=base_request.strategy_names,
                start=start,
                end=end,
                initial_cash=base_request.initial_cash,
                exposure=base_request.exposure,
                transaction_cost_bps=base_request.transaction_cost_bps,
                slippage_bps=base_request.slippage_bps,
                split_date=None,
            )
            comparison = run_strategy_comparison(request)
            comparisons.append(comparison)
            summary = _summary_with_score(comparison.full_summary)
            summary.insert(0, "period", period_name)
            summaries.append(summary)

        combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        return {"summary": combined, "comparisons": comparisons}

    def _run_in_background(self, status: str, kind: str, worker: Callable[[], Any]) -> None:
        self._set_action_buttons_enabled(False)
        self.status_var.set(status)
        self._log(status)

        def target() -> None:
            try:
                self._messages.put(("result", TaskResult(kind=kind, payload=worker())))
            except Exception as error:  # noqa: BLE001 - GUI must surface unexpected failures cleanly.
                details = "".join(traceback.format_exception_only(type(error), error)).strip()
                self._messages.put(("error", details))

        threading.Thread(target=target, daemon=True).start()

    def _poll_messages(self) -> None:
        try:
            kind, payload = self._messages.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_messages)
            return

        if kind == "result":
            self._handle_task_result(payload)
            self.status_var.set("Ready.")
        else:
            self._log(f"ERROR: {payload}")
            self.status_var.set("Failed. Check the Run Log tab.")
            messagebox.showerror("Pluto Trader", str(payload))

        self._set_action_buttons_enabled(True)
        self.after(100, self._poll_messages)

    def _handle_task_result(self, task_result: TaskResult) -> None:
        if task_result.kind == "backtest":
            self._handle_backtest(task_result.payload)
            return
        if task_result.kind == "comparison":
            self._handle_comparison(task_result.payload)
            return
        if task_result.kind == "analysis":
            self._handle_analysis(task_result.payload)
            return
        if task_result.kind == "regime":
            self._handle_regime(task_result.payload)
            return
        self._log(f"Unknown task result type: {task_result.kind}")

    def _handle_backtest(self, result) -> None:
        self.current_results = [result.result]
        self.current_summary = _summary_with_score(result.summary)
        self.current_analysis = None
        self._render_plot(self.current_results)
        self._populate_summary(self.current_summary)
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log(
            f"Backtest complete: {result.result.display_name}, {result.start_date.date()} to "
            f"{result.end_date.date() if result.end_date else 'latest available'}."
        )
        self.notebook.select(0)

    def _handle_comparison(self, result: StrategyComparisonResult) -> None:
        self.current_results = result.full_results
        self.current_summary = _summary_with_score(result.full_summary)
        self.current_analysis = None
        self._render_plot(self.current_results)
        self._populate_summary(self.current_summary)
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log(
            f"Comparison complete: {len(result.strategy_names)} strategies, {result.start_date.date()} to "
            f"{result.end_date.date() if result.end_date else 'latest available'}."
        )

        if result.split_date:
            if result.train_summary is None or result.test_summary is None:
                self._log("Train/test split skipped because the split date leaves one side empty.")
            else:
                train = _summary_with_score(result.train_summary)
                test = _summary_with_score(result.test_summary)
                best_train = train.sort_values("score", ascending=False).iloc[0]
                best_test = test.sort_values("score", ascending=False).iloc[0]
                self._log(
                    f"Split check: train best={best_train['strategy']} ({best_train['score']:.1f}), "
                    f"test best={best_test['strategy']} ({best_test['score']:.1f})."
                )
        self.notebook.select(0)

    def _handle_regime(self, payload: dict[str, Any]) -> None:
        self.current_results = []
        self.current_summary = payload["summary"]
        self.current_analysis = None
        self._clear_plot("Regime scorecard complete. Run a backtest or comparison to render a specific equity curve.")
        self._populate_summary(self.current_summary)
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log("Regime scorecard complete.")

        if self.current_summary is not None and not self.current_summary.empty:
            for period, group in self.current_summary.groupby("period", sort=False):
                best = group.sort_values("score", ascending=False).iloc[0]
                self._log(f"{period}: best={best['strategy']} score={best['score']:.1f} return={best['total_return']:.2%}.")
        self.notebook.select(1)

    def _handle_analysis(self, result: TradeAnalysisResult) -> None:
        self.current_analysis = result
        self._populate_diagnostics(result)
        analysis_summary = _summary_with_score(pd.DataFrame([{"strategy": result.strategy_name, **result.result.metrics}]))
        self.current_summary = analysis_summary
        self.current_results = [result.result]
        self._render_plot(self.current_results)
        self._populate_summary(analysis_summary)
        self._update_metric_cards(analysis_summary)
        self._set_export_buttons_enabled(True)
        self._log(
            f"Trade analysis complete: {result.result.display_name}. "
            f"Trades={len(result.trades)}, whipsaws={len(result.whipsaws)}."
        )
        self.notebook.select(2)

    def _backtest_request(self) -> BacktestRequest:
        return BacktestRequest(
            strategy_name=self.strategy_var.get(),
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
        )

    def _comparison_request(self, include_split: bool) -> StrategyComparisonRequest:
        strategy_names = [name for name, variable in self.strategy_checks.items() if variable.get()]
        if not strategy_names:
            raise ValueError("Select at least one strategy for comparison.")

        return StrategyComparisonRequest(
            strategy_names=strategy_names,
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            split_date=_optional_text(self.split_var) if include_split else None,
        )

    def _trade_analysis_request(self) -> TradeAnalysisRequest:
        return TradeAnalysisRequest(
            strategy_name=self.strategy_var.get(),
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            top_days=_int_value(self.top_days_var, "Top up days"),
            short_trade_days=_int_value(self.short_trade_days_var, "Whipsaw days"),
            suspicious_move_threshold=_float_value(self.suspicious_move_var, "Large move"),
        )

    def _render_plot(self, results: list[BacktestResult]) -> None:
        if not results:
            self._clear_plot("No results available to plot.")
            return

        self._clear_plot()
        figure = create_equity_curve_figure(results)
        self.current_figure = figure
        self.plot_original_limits = {}
        self._clear_plot_marker()
        self.plot_canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas, self.toolbar_frame, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.grid(row=0, column=0, sticky="w")
        self._remember_plot_limits()
        self._connect_plot_events()
        self._focus_recent_plot(252, log_action=False)
        self.save_plot_button.configure(state=tk.NORMAL)

    def _clear_plot(self, message: str = "No plot yet. Run a backtest or strategy comparison.") -> None:
        if hasattr(self, "plot_canvas") and self.plot_canvas is not None:
            for connection_id in self.plot_connections:
                self.plot_canvas.mpl_disconnect(connection_id)
            self.plot_connections = []
        if hasattr(self, "plot_canvas") and self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        if hasattr(self, "toolbar_frame"):
            for child in self.toolbar_frame.winfo_children():
                child.destroy()
            self.plot_toolbar = None
        if self.current_figure is not None:
            self.current_figure.clear()
            self.current_figure = None
        self.plot_annotation = None
        self.plot_marker = None
        self.plot_original_limits = {}
        if hasattr(self, "plot_frame"):
            for child in self.plot_frame.winfo_children():
                child.destroy()
            self.plot_placeholder = ttk.Label(
                self.plot_frame,
                text=message,
                background="#f8fafc",
                foreground="#64748b",
                anchor="center",
            )
            self.plot_placeholder.grid(row=0, column=0, sticky="nsew")
        if hasattr(self, "save_plot_button"):
            self.save_plot_button.configure(state=tk.DISABLED)

    def _connect_plot_events(self) -> None:
        if self.plot_canvas is None:
            return

        self.plot_connections = [
            self.plot_canvas.mpl_connect("button_press_event", self._on_plot_click),
            self.plot_canvas.mpl_connect("scroll_event", self._on_plot_scroll),
        ]

    def _remember_plot_limits(self) -> None:
        if self.current_figure is None:
            return

        self.plot_original_limits = {
            axis: (axis.get_xlim(), axis.get_ylim())
            for axis in self.current_figure.axes
        }

    def _reset_plot_view(self) -> None:
        if self.current_figure is None or not self.plot_original_limits:
            self._show_info("No plot is available to reset yet.")
            return

        for axis, (x_limits, y_limits) in self.plot_original_limits.items():
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)

        self._clear_plot_marker(redraw=False)
        if self.plot_canvas is not None:
            self.plot_canvas.draw_idle()
        self._log("Reset plot to full range.")

    def _focus_recent_plot(self, days: int, log_action: bool = True) -> None:
        if self.current_figure is None or not self.current_results:
            if log_action:
                self._show_info("No plotted backtest is available to focus yet.")
            return

        index = self.current_results[0].data.index
        if index.empty:
            return

        start_index = max(len(index) - days, 0)
        start = index[start_index]
        end = index[-1]
        for axis in self.current_figure.axes:
            axis.set_xlim(start, end)

        self._autoscale_visible_y()
        self._clear_plot_marker(redraw=False)
        if self.plot_canvas is not None:
            self.plot_canvas.draw_idle()
        if log_action:
            self._log(f"Focused plot on the most recent {days} trading days.")

    def _autoscale_visible_y(self) -> None:
        if self.current_figure is None:
            return

        for axis in self.current_figure.axes:
            x_min, x_max = axis.get_xlim()
            visible_values = []
            for line in axis.get_lines():
                if line.get_label().startswith("_"):
                    continue

                x_values = _line_x_values(line)
                y_values = np.asarray(line.get_ydata(orig=False), dtype=float)
                if len(x_values) != len(y_values):
                    continue

                mask = (
                    (x_values >= min(x_min, x_max))
                    & (x_values <= max(x_min, x_max))
                    & np.isfinite(y_values)
                )
                if mask.any():
                    visible_values.extend(y_values[mask].tolist())

            if not visible_values:
                continue

            y_min = min(visible_values)
            y_max = max(visible_values)
            if y_min == y_max:
                margin = abs(y_min) * 0.05 or 1.0
            else:
                margin = (y_max - y_min) * 0.08

            if axis.get_ylabel() == "Drawdown":
                axis.set_ylim(min(y_min - margin, -0.01), max(y_max + margin, 0.02))
            else:
                axis.set_ylim(y_min - margin, y_max + margin)

    def _on_plot_scroll(self, event) -> None:
        if self.current_figure is None or event.inaxes is None or event.xdata is None:
            return
        if self.plot_toolbar is not None and self.plot_toolbar.mode:
            return

        scale = 0.80 if event.button == "up" else 1.25
        reference_axis = self.current_figure.axes[0]
        x_min, x_max = reference_axis.get_xlim()
        full_limits = self.plot_original_limits.get(reference_axis, ((x_min, x_max), (0, 1)))[0]
        full_min, full_max = min(full_limits), max(full_limits)
        center = event.xdata

        new_min = center - (center - x_min) * scale
        new_max = center + (x_max - center) * scale
        if new_max - new_min < 5:
            return

        new_min = max(new_min, full_min)
        new_max = min(new_max, full_max)
        if new_max <= new_min:
            return

        for axis in self.current_figure.axes:
            axis.set_xlim(new_min, new_max)

        self._autoscale_visible_y()
        self._clear_plot_marker(redraw=False)
        if self.plot_canvas is not None:
            self.plot_canvas.draw_idle()

    def _on_plot_click(self, event) -> None:
        if self.current_figure is None or self.plot_canvas is None or event.inaxes is None:
            return
        if self.plot_toolbar is not None and self.plot_toolbar.mode:
            return
        if event.x is None or event.y is None:
            return

        nearest = self._nearest_plot_point(event.inaxes, event.x, event.y)
        if nearest is None:
            return

        self._clear_plot_marker(redraw=False)
        axis = event.inaxes
        color = nearest["line"].get_color()
        (self.plot_marker,) = axis.plot(
            [nearest["x"]],
            [nearest["y"]],
            marker="o",
            markersize=7,
            markerfacecolor="#f97316",
            markeredgecolor="#111827",
            zorder=20,
        )
        self.plot_annotation = axis.annotate(
            _format_plot_annotation(axis, nearest),
            xy=(nearest["x"], nearest["y"]),
            xytext=(14, 16),
            textcoords="offset points",
            fontsize=9,
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.45", "fc": "#ffffff", "ec": color, "alpha": 0.96},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            zorder=21,
        )
        self.plot_canvas.draw_idle()

    def _nearest_plot_point(self, axis, event_x: float, event_y: float) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_distance = float("inf")

        for line in axis.get_lines():
            label = line.get_label()
            if not label or label.startswith("_"):
                continue

            x_values = _line_x_values(line)
            y_values = np.asarray(line.get_ydata(orig=False), dtype=float)
            if len(x_values) != len(y_values) or len(x_values) == 0:
                continue

            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if not finite.any():
                continue

            points = axis.transData.transform(np.column_stack([x_values[finite], y_values[finite]]))
            distances = ((points[:, 0] - event_x) ** 2) + ((points[:, 1] - event_y) ** 2)
            nearest_index = int(np.argmin(distances))
            distance = float(distances[nearest_index])
            if distance < best_distance:
                visible_x = x_values[finite]
                visible_y = y_values[finite]
                best_distance = distance
                best = {
                    "line": line,
                    "label": label,
                    "x": float(visible_x[nearest_index]),
                    "y": float(visible_y[nearest_index]),
                    "first_y": _first_finite(y_values),
                }

        return best

    def _clear_plot_marker(self, redraw: bool = True) -> None:
        if self.plot_annotation is not None:
            try:
                self.plot_annotation.remove()
            except ValueError:
                pass
            self.plot_annotation = None

        if self.plot_marker is not None:
            try:
                self.plot_marker.remove()
            except ValueError:
                pass
            self.plot_marker = None

        if redraw and self.plot_canvas is not None:
            self.plot_canvas.draw_idle()

    def _populate_summary(self, summary: pd.DataFrame | None) -> None:
        if summary is None or summary.empty:
            self._populate_tree(self.summary_tree, pd.DataFrame())
            return

        columns = [column for column in SUMMARY_COLUMN_ORDER if column in summary.columns]
        extra_columns = [column for column in summary.columns if column not in columns and column != "display_name"]
        display = summary[columns + extra_columns].copy()
        self._populate_tree(self.summary_tree, display)

    def _populate_diagnostics(self, result: TradeAnalysisResult) -> None:
        self._populate_tree(self.diagnostic_trees["trades"], result.trades)
        self._populate_tree(self.diagnostic_trees["missed_days"], result.missed_days)
        self._populate_tree(self.diagnostic_trees["whipsaws"], result.whipsaws)
        self._populate_tree(self.diagnostic_trees["suspicious_moves"], result.suspicious_moves)

    def _clear_diagnostics(self) -> None:
        for tree_container in self.diagnostic_trees.values():
            self._populate_tree(tree_container, pd.DataFrame())

    def _populate_tree(self, tree_container: ttk.Frame, frame: pd.DataFrame) -> None:
        tree = _tree_from_container(tree_container)
        tree.delete(*tree.get_children())

        if frame is None or frame.empty:
            tree.configure(columns=("message",))
            tree.heading("message", text="Message")
            tree.column("message", width=360, anchor="w")
            tree.insert("", tk.END, values=("No data available.",))
            return

        columns = [str(column) for column in frame.columns]
        tree.configure(columns=columns)
        for column in columns:
            label = SUMMARY_COLUMN_LABELS.get(column, column.replace("_", " ").title())
            anchor = "w" if column in {"period", "strategy", "display_name", "date"} else "e"
            tree.heading(column, text=label)
            tree.column(column, width=_column_width(column), minwidth=70, anchor=anchor, stretch=True)

        for _, row in frame.iterrows():
            values = [_format_cell(column, row[column]) for column in frame.columns]
            tree.insert("", tk.END, values=values)

    def _update_metric_cards(self, summary: pd.DataFrame | None) -> None:
        if summary is None or summary.empty:
            for variable in self.metric_vars.values():
                variable.set("-")
            return

        best = summary.sort_values("score", ascending=False).iloc[0] if "score" in summary.columns else summary.iloc[0]
        self.metric_vars["best"].set(str(best.get("strategy", "-")))
        self.metric_vars["score"].set(_format_cell("score", best.get("score", "-")))
        self.metric_vars["return"].set(_format_cell("total_return", best.get("total_return", "-")))
        self.metric_vars["drawdown"].set(_format_cell("max_drawdown", best.get("max_drawdown", "-")))
        self.metric_vars["sharpe"].set(_format_cell("sharpe", best.get("sharpe", "-")))
        self.metric_vars["trades"].set(_format_cell("trades", best.get("trades", "-")))

    def _save_plot(self) -> None:
        if self.current_figure is None:
            self._show_info("No plot is available to save yet.")
            return

        path = filedialog.asksaveasfilename(
            title="Save equity curve",
            initialdir=str(BOT_DIR),
            initialfile="pluto_equity_curve.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return

        self.current_figure.savefig(path, dpi=160, bbox_inches="tight")
        self._log(f"Saved plot: {path}")

    def _save_summary(self) -> None:
        if self.current_summary is None or self.current_summary.empty:
            self._show_info("No summary is available to save yet.")
            return

        path = filedialog.asksaveasfilename(
            title="Save summary CSV",
            initialdir=str(BOT_DIR),
            initialfile="pluto_strategy_summary.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        save_summary_csv(self.current_summary, Path(path))
        self._log(f"Saved summary: {path}")

    def _save_diagnostics(self) -> None:
        if self.current_analysis is None:
            self._show_info("Run trade analysis before saving diagnostics.")
            return

        directory = filedialog.askdirectory(title="Choose diagnostics output folder", initialdir=str(BOT_DIR))
        if not directory:
            return

        output_dir = Path(directory)
        strategy_name = self.current_analysis.strategy_name
        files = AnalysisFiles(
            trades=output_dir / f"trade_analysis_{strategy_name}_trades.csv",
            missed_days=output_dir / f"trade_analysis_{strategy_name}_missed_best_days.csv",
            whipsaws=output_dir / f"trade_analysis_{strategy_name}_whipsaws.csv",
            suspicious_moves=output_dir / f"trade_analysis_{strategy_name}_suspicious_moves.csv",
        )
        save_reports(
            trades=self.current_analysis.trades,
            missed_days=self.current_analysis.missed_days,
            whipsaws=self.current_analysis.whipsaws,
            suspicious_moves=self.current_analysis.suspicious_moves,
            files=files,
        )
        self._log(f"Saved diagnostics to: {output_dir}")

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save run log",
            initialdir=str(BOT_DIR),
            initialfile="pluto_gui_log.txt",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        text = self.output.get("1.0", tk.END)
        Path(path).write_text(text, encoding="utf-8")
        self._log(f"Saved log: {path}")

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self._action_buttons:
            button.configure(state=state)

    def _set_export_buttons_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self._export_buttons:
            button.configure(state=state)
        if self.current_figure is None:
            self.save_plot_button.configure(state=tk.DISABLED)
        if self.current_analysis is None:
            self.save_reports_button.configure(state=tk.DISABLED)

    def _show_validation_error(self, error: ValueError) -> None:
        message = str(error)
        self.status_var.set("Input error.")
        self._log(f"INPUT ERROR: {message}")
        messagebox.showerror("Invalid input", message)

    def _show_info(self, message: str) -> None:
        self._log(message)
        messagebox.showinfo("Pluto Trader", message)

    def _log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, f"[{timestamp}] {text}\n")
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)


def _tree_from_container(tree_container: ttk.Frame) -> ttk.Treeview:
    for child in tree_container.winfo_children():
        if isinstance(child, ttk.Treeview):
            return child
    raise RuntimeError("Treeview container does not contain a tree.")


def _line_x_values(line) -> np.ndarray:
    x_data = line.get_xdata(orig=False)
    if len(x_data) == 0:
        return np.asarray([], dtype=float)

    first_value = x_data[0]
    if isinstance(first_value, (datetime, date, pd.Timestamp, np.datetime64)):
        return np.asarray(mdates.date2num(pd.to_datetime(x_data).to_pydatetime()), dtype=float)

    try:
        return np.asarray(x_data, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(mdates.date2num(pd.to_datetime(x_data).to_pydatetime()), dtype=float)


def _first_finite(values: np.ndarray) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(finite_values[0])


def _format_plot_annotation(axis, point: dict[str, Any]) -> str:
    date_text = mdates.num2date(point["x"]).date().isoformat()
    label = point["label"]
    value = point["y"]

    if axis.get_ylabel() == "Drawdown":
        return f"{label}\n{date_text}\nDrawdown: {value:.2%}"

    first_y = point.get("first_y")
    return_text = ""
    if first_y and first_y > 0:
        return_text = f"\nReturn: {(value / first_y) - 1:.2%}"

    return f"{label}\n{date_text}\nValue: ${value:,.2f}{return_text}"


def _summary_with_score(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()

    scored = summary.copy()
    scored["excess_vs_same_exposure"] = (
        scored["total_return"] - scored["buy_hold_same_exposure_return"]
        if "buy_hold_same_exposure_return" in scored.columns
        else 0.0
    )
    scored["score"] = scored.apply(_score_row, axis=1)
    return scored.sort_values("score", ascending=False)


def _score_row(row: pd.Series) -> float:
    excess = float(row.get("excess_vs_same_exposure", 0.0) or 0.0)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    max_drawdown = float(row.get("max_drawdown", 0.0) or 0.0)
    trades = float(row.get("trades", 0.0) or 0.0)
    missed_days = float(row.get("missed_top_20_up_days", 0.0) or 0.0)

    return_points = 35 + _clamp(excess * 100, -20, 25)
    sharpe_points = _clamp(sharpe * 12, -10, 20)
    drawdown_points = _clamp((0.35 + max_drawdown) / 0.35 * 25, 0, 25)
    trade_penalty = min(trades * 0.08, 8)
    missed_penalty = min(missed_days * 0.75, 12)
    score = return_points + sharpe_points + drawdown_points - trade_penalty - missed_penalty
    return round(_clamp(score, 0, 100), 1)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _format_cell(column: str, value: Any) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, bool):
        return "yes" if value else "no"

    if column in PERCENT_COLUMNS:
        return f"{float(value):.2%}"

    if column in MONEY_COLUMNS:
        return f"${float(value):,.2f}"

    if column in INTEGER_COLUMNS:
        return f"{int(float(value))}"

    if column == "score":
        return f"{float(value):.1f}"

    if column == "sharpe":
        return f"{float(value):.2f}"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def _column_width(column: str) -> int:
    widths = {
        "period": 150,
        "display_name": 220,
        "strategy": 150,
        "score": 80,
        "final_value": 120,
        "date": 110,
        "entry_signal_date": 130,
        "entry_active_date": 130,
        "exit_signal_date": 130,
        "exit_active_date": 130,
        "last_backtest_date": 140,
    }
    return widths.get(column, 110)


def _required_text(variable: tk.StringVar, label: str) -> str:
    value = variable.get().strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def _optional_text(variable: tk.StringVar) -> str | None:
    value = variable.get().strip()
    return value or None


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


def main() -> int:
    app = PlutoTraderGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
