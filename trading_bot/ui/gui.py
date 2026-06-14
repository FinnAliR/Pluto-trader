"""Tkinter dashboard for data-only backtests and strategy diagnostics."""

from __future__ import annotations

import calendar
import json
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

from trading_bot.backtesting.engine import BacktestResult, create_equity_curve_figure, save_summary_csv
from trading_bot.services.backtest_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    BacktestRequest,
    StrategyComparisonRequest,
    StrategyComparisonResult,
    normalize_symbol,
    run_single_backtest,
    run_strategy_comparison,
)
from trading_bot.services.edge_analysis_service import EdgeAnalysisRequest, EdgeAnalysisResult, run_edge_analysis
from trading_bot.services.trade_analysis_service import (
    DEFAULT_SHORT_TRADE_DAYS,
    DEFAULT_SUSPICIOUS_MOVE_THRESHOLD,
    DEFAULT_TOP_DAYS,
    TradeAnalysisRequest,
    TradeAnalysisResult,
    run_trade_analysis,
)
from trading_bot.strategies.registry import (
    available_strategy_names,
    normalize_strategy_params,
    strategy_discovery_errors,
    strategy_parameter_specs,
)
from trading_bot.strategies.validation import validate_all_strategies, validation_results_to_frame
from trading_bot.diagnostics.reports import AnalysisFiles, save_reports


BOT_DIR = Path(__file__).resolve().parents[1]
RUN_PRESETS_DIR = BOT_DIR.parent / "presets"
RUN_PRESET_VERSION = 1
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

RUN_ASPECTS = [
    (
        "selected",
        "Backtest / Compare",
        "Run one checked strategy as a single backtest; compare strategies when multiple are checked.",
    ),
    (
        "regime",
        "Regime Scorecard",
        "Compare checked strategies across full-history, crash, bear-market, and recovery periods.",
    ),
    (
        "market",
        "Market Matrix",
        "Compare checked strategies across comma-separated symbols from the Market field.",
    ),
    (
        "edge",
        "Edge Lab",
        "Score checked strategies by out-of-sample behavior, market breadth, regime consistency, cost stress, and parameter stability.",
    ),
    (
        "analysis",
        "Trade Diagnostics",
        "Build trade diagnostics for the primary strategy, or the only checked strategy when exactly one is selected.",
    ),
]

MARKET_CATALOG = [
    ("SPY", "S&P 500 ETF, broad US large-cap equities"),
    ("QQQ", "Nasdaq-100 ETF, US large-cap growth and technology tilt"),
    ("IWM", "Russell 2000 ETF, US small-cap equities"),
    ("DIA", "Dow Jones Industrial Average ETF, US blue-chip equities"),
    ("VTI", "Total US stock market ETF"),
    ("RSP", "Equal-weight S&P 500 ETF"),
    ("XLK", "US technology sector ETF"),
    ("XLF", "US financials sector ETF"),
    ("XLE", "US energy sector ETF"),
    ("XLV", "US healthcare sector ETF"),
    ("XLY", "US consumer discretionary sector ETF"),
    ("XLP", "US consumer staples sector ETF"),
    ("GLD", "Gold trust ETF"),
    ("SLV", "Silver trust ETF"),
    ("TLT", "20+ year US Treasury bond ETF"),
    ("IEF", "7-10 year US Treasury bond ETF"),
    ("HYG", "High-yield corporate bond ETF"),
    ("LQD", "Investment-grade corporate bond ETF"),
    ("USO", "US oil fund ETF"),
    ("UNG", "US natural gas fund ETF"),
    ("EEM", "Emerging markets equity ETF"),
    ("EFA", "Developed international equity ETF"),
    ("FXI", "China large-cap equity ETF"),
    ("EWJ", "Japan equity ETF"),
    ("AAPL", "Apple common stock"),
    ("MSFT", "Microsoft common stock"),
    ("NVDA", "NVIDIA common stock"),
    ("AMZN", "Amazon common stock"),
    ("META", "Meta Platforms common stock"),
    ("TSLA", "Tesla common stock"),
    ("JPM", "JPMorgan Chase common stock"),
    ("XOM", "Exxon Mobil common stock"),
]
MARKET_SYMBOLS = [symbol for symbol, _description in MARKET_CATALOG]
MARKET_DESCRIPTION_TEXT = "\n".join(f"{symbol}: {description}" for symbol, description in MARKET_CATALOG)

SUMMARY_COLUMN_LABELS = {
    "status": "Status",
    "symbol": "Market",
    "period": "Period",
    "score": "Score",
    "edge_score": "Edge Score",
    "verdict": "Verdict",
    "strategy": "Strategy",
    "display_name": "Display Name",
    "primary_symbol": "Primary",
    "full_excess": "Full Excess",
    "train_excess": "Train Excess",
    "test_excess": "Test Excess",
    "test_sharpe": "Test Sharpe",
    "test_max_drawdown": "Test Max DD",
    "market_pass_rate": "Market Pass",
    "positive_markets": "Positive Markets",
    "markets_tested": "Markets",
    "regime_pass_rate": "Regime Pass",
    "regime_periods_tested": "Regimes",
    "cost_resilience": "Cost Resilience",
    "stressed_excess": "Stressed Excess",
    "parameter_stability": "Param Stability",
    "parameter_variants_tested": "Param Variants",
    "notes": "Notes",
    "variant": "Variant",
    "changed_param": "Changed Param",
    "param_value": "Param Value",
    "passes_edge_gate": "Passes Gate",
    "cost_case": "Cost Case",
    "editable_parameters": "Editable Params",
    "checks": "Checks",
    "warnings": "Warnings",
    "errors": "Errors",
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
    "status",
    "symbol",
    "period",
    "edge_score",
    "verdict",
    "score",
    "strategy",
    "display_name",
    "primary_symbol",
    "full_excess",
    "train_excess",
    "test_excess",
    "test_sharpe",
    "test_max_drawdown",
    "market_pass_rate",
    "positive_markets",
    "markets_tested",
    "regime_pass_rate",
    "regime_periods_tested",
    "cost_resilience",
    "stressed_excess",
    "parameter_stability",
    "parameter_variants_tested",
    "notes",
    "total_return",
    "buy_hold_same_exposure_return",
    "excess_vs_same_exposure",
    "full_excess",
    "train_excess",
    "test_excess",
    "test_max_drawdown",
    "market_pass_rate",
    "regime_pass_rate",
    "cost_resilience",
    "stressed_excess",
    "parameter_stability",
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
INTEGER_COLUMNS = {
    "trades",
    "missed_top_20_up_days",
    "trade_number",
    "days_held",
    "positive_markets",
    "markets_tested",
    "regime_periods_tested",
    "parameter_variants_tested",
}


@dataclass(frozen=True)
class TaskResult:
    """Result passed from the worker thread back to the Tk event loop."""

    kind: str
    payload: Any


class WidgetDescription:
    """Attach a keyboard/focus reachable description to a Tk widget."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<FocusIn>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<FocusOut>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None or not self.text:
            return

        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 16
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self._tip,
            text=self.text,
            background="#111827",
            foreground="#f8fafc",
            relief="solid",
            borderwidth=1,
            padding=(8, 5),
            wraplength=360,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class DatePickerPopup:
    """Small dependency-free calendar popup for YYYY-MM-DD fields."""

    def __init__(self, parent: tk.Widget, variable: tk.StringVar, allow_blank: bool = False) -> None:
        self.parent = parent
        self.variable = variable
        self.allow_blank = allow_blank

        initial_date = _parse_picker_date(variable.get()) or date.today()
        self.year = initial_date.year
        self.month = initial_date.month
        self.window = tk.Toplevel(parent)
        self.window.title("Select date")
        self.window.transient(parent.winfo_toplevel())
        self.window.resizable(False, False)
        self.window.grab_set()

        self.body = ttk.Frame(self.window, padding=8)
        self.body.grid(row=0, column=0, sticky="nsew")
        self._render()

        x = parent.winfo_rootx()
        y = parent.winfo_rooty() + parent.winfo_height() + 4
        self.window.geometry(f"+{x}+{y}")
        self.window.bind("<Escape>", lambda _event: self.window.destroy())

    def _render(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

        header = ttk.Frame(self.body)
        header.grid(row=0, column=0, columnspan=7, sticky="ew", pady=(0, 6))
        header.columnconfigure(1, weight=1)

        ttk.Button(header, text="<", width=3, command=self._previous_month).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=f"{calendar.month_name[self.month]} {self.year}", anchor="center").grid(
            row=0,
            column=1,
            sticky="ew",
            padx=8,
        )
        ttk.Button(header, text=">", width=3, command=self._next_month).grid(row=0, column=2, sticky="e")

        for column, day_name in enumerate(calendar.day_abbr):
            ttk.Label(self.body, text=day_name, anchor="center", width=4).grid(row=1, column=column, padx=1, pady=1)

        month_calendar = calendar.Calendar(firstweekday=0).monthdayscalendar(self.year, self.month)
        for row, week in enumerate(month_calendar, start=2):
            for column, day_number in enumerate(week):
                if day_number == 0:
                    ttk.Label(self.body, text="", width=4).grid(row=row, column=column, padx=1, pady=1)
                    continue

                button = ttk.Button(
                    self.body,
                    text=str(day_number),
                    width=4,
                    command=lambda day=day_number: self._select_day(day),
                )
                button.grid(row=row, column=column, padx=1, pady=1)

        footer_row = len(month_calendar) + 2
        ttk.Button(self.body, text="Today", command=self._select_today).grid(
            row=footer_row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(6, 0),
        )
        if self.allow_blank:
            ttk.Button(self.body, text="Clear", command=self._clear).grid(
                row=footer_row,
                column=4,
                columnspan=3,
                sticky="ew",
                pady=(6, 0),
            )

    def _previous_month(self) -> None:
        if self.month == 1:
            self.month = 12
            self.year -= 1
        else:
            self.month -= 1
        self._render()

    def _next_month(self) -> None:
        if self.month == 12:
            self.month = 1
            self.year += 1
        else:
            self.month += 1
        self._render()

    def _select_day(self, day_number: int) -> None:
        self.variable.set(date(self.year, self.month, day_number).isoformat())
        self.window.destroy()

    def _select_today(self) -> None:
        self.variable.set(date.today().isoformat())
        self.window.destroy()

    def _clear(self) -> None:
        self.variable.set("")
        self.window.destroy()


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
        if not self.strategy_names:
            raise RuntimeError("No valid strategies were discovered in trading_bot.strategies.")

        self.strategy_discovery_errors = strategy_discovery_errors()
        default_strategy = "ma_crossover" if "ma_crossover" in self.strategy_names else self.strategy_names[0]
        self.strategy_var = tk.StringVar(value=default_strategy)
        self.strategy_checks = {name: tk.BooleanVar(value=True) for name in self.strategy_names}
        self.market_var = tk.StringVar(value="SPY")
        self.run_aspect_vars = {
            key: tk.BooleanVar(value=(key == "selected"))
            for key, _label, _description in RUN_ASPECTS
        }
        self.strategy_param_edit_var = tk.StringVar(value=self.strategy_names[0])
        self.strategy_param_vars = {
            strategy_name: {
                spec.name: tk.StringVar(value=_format_param_value(spec.default))
                for spec in strategy_parameter_specs(strategy_name)
            }
            for strategy_name in self.strategy_names
        }
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
        self.edge_trees: dict[str, ttk.Frame] = {}
        self.diagnostic_trees: dict[str, ttk.Frame] = {}
        self.strategy_param_fields_frame: ttk.Frame | None = None

        self._configure_style()
        self._build_layout()
        self._set_export_buttons_enabled(False)
        self._log_strategy_discovery_errors()
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

    def _describe(self, widget: tk.Widget, text: str) -> tk.Widget:
        if not self._should_describe_widget(widget):
            return widget
        setattr(widget, "_pluto_alt_text", text)
        setattr(widget, "_pluto_description", WidgetDescription(widget, text))
        return widget

    def _describe_widget(self, widget: tk.Widget, text: str) -> None:
        self._describe(widget, text)

    def _should_describe_widget(self, widget: tk.Widget) -> bool:
        return isinstance(
            widget,
            (
                tk.Button,
                tk.Checkbutton,
                tk.Entry,
                tk.Text,
                ttk.Button,
                ttk.Checkbutton,
                ttk.Entry,
                ttk.Combobox,
            ),
        )

    def _build_strategy_parameter_editor(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        label = ttk.Label(parent, text="Edit strategy")
        label.grid(row=0, column=0, sticky="w", pady=(0, 6))
        selector = ttk.Combobox(
            parent,
            textvariable=self.strategy_param_edit_var,
            values=self.strategy_names,
            state="readonly",
        )
        selector.grid(row=0, column=1, sticky="ew", pady=(0, 6), padx=(6, 0))
        selector.bind("<<ComboboxSelected>>", lambda _event: self._render_strategy_param_fields())
        self._describe_widget(selector, "Choose which strategy parameter set to edit.")

        self.strategy_param_fields_frame = ttk.Frame(parent)
        self.strategy_param_fields_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.strategy_param_fields_frame.columnconfigure(1, weight=1)
        self._render_strategy_param_fields()

    def _render_strategy_param_fields(self) -> None:
        if self.strategy_param_fields_frame is None:
            return

        for child in self.strategy_param_fields_frame.winfo_children():
            child.destroy()

        strategy_name = self.strategy_param_edit_var.get()
        specs = strategy_parameter_specs(strategy_name)
        if not specs:
            ttk.Label(self.strategy_param_fields_frame, text="No editable parameters.").grid(
                row=0,
                column=0,
                columnspan=2,
                sticky="w",
            )
            return

        for row, spec in enumerate(specs):
            label = ttk.Label(self.strategy_param_fields_frame, text=spec.label)
            label.grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(
                self.strategy_param_fields_frame,
                textvariable=self.strategy_param_vars[strategy_name][spec.name],
                width=12,
            )
            entry.grid(row=row, column=1, sticky="ew", pady=2, padx=(6, 0))
            description = (
                f"{strategy_name} parameter '{spec.name}'. {spec.description} "
                f"Default: {_format_param_value(spec.default)}."
            )
            self._describe_widget(entry, description)

        reset_button = ttk.Button(
            self.strategy_param_fields_frame,
            text="Reset This Strategy",
            command=lambda name=strategy_name: self._reset_strategy_params(name),
        )
        reset_button.grid(row=len(specs), column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._describe_widget(reset_button, f"Reset {strategy_name} parameters to defaults.")

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

        status_label = ttk.Label(main, textvariable=self.status_var, background="#edf2f7", foreground="#334155")
        status_label.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )
        self._describe_widget(status_label, "Current application status and the latest running task.")

        self.notebook = ttk.Notebook(main)
        self._describe_widget(self.notebook, "Main result area with dashboard chart, summary table, diagnostics, and run log tabs.")
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._build_dashboard_tab()
        self._build_summary_tab()
        self._build_edge_tab()
        self._build_diagnostics_tab()
        self._build_log_tab()

    def _log_strategy_discovery_errors(self) -> None:
        if not self.strategy_discovery_errors:
            return

        self.status_var.set("Ready. Some custom strategies failed to load; see Run Log.")
        self._log("Some strategy files could not be loaded and were skipped:")
        for module_name, error in sorted(self.strategy_discovery_errors.items()):
            self._log(f"{module_name}: {type(error).__name__}: {error}")

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
        title_label = ttk.Label(parent, text="Pluto Trader", style="Title.Side.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        subtitle_label = ttk.Label(
            parent,
            text="Evaluate market strategies with backtests, scorecards, diagnostics, and exportable charts.",
            style="Muted.Side.TLabel",
            wraplength=300,
        )
        subtitle_label.grid(row=1, column=0, sticky="ew", pady=(4, 18))
        self._describe_widget(title_label, "Pluto Trader dashboard title.")
        self._describe_widget(subtitle_label, "Dashboard purpose: evaluate strategies across market symbols with data-only tools and exports.")

        strategy_frame = ttk.LabelFrame(parent, text="Strategies", padding=10)
        strategy_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        strategy_frame.columnconfigure(0, weight=1)
        strategy_frame.columnconfigure(1, weight=1)
        self._describe_widget(strategy_frame, "Strategy controls for choosing selected strategies and a primary diagnostic strategy.")
        primary_label = ttk.Label(strategy_frame, text="Primary strategy")
        primary_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self._describe_widget(primary_label, "Primary strategy used for trade analysis when multiple strategies are checked.")
        primary_selector = ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            values=self.strategy_names,
            state="readonly",
        )
        primary_selector.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 8))
        self._describe_widget(primary_selector, "Primary strategy used for diagnostics; if exactly one checkbox is selected, it is synced here.")

        selected_label = ttk.Label(strategy_frame, text="Selected strategies")
        selected_label.grid(row=2, column=0, columnspan=2, sticky="w")
        self._describe_widget(selected_label, "Checked strategies used by Run Selected Strategies and the regime scorecard.")
        for index, strategy_name in enumerate(self.strategy_names):
            checkbox = ttk.Checkbutton(
                strategy_frame,
                text=strategy_name,
                variable=self.strategy_checks[strategy_name],
                command=self._sync_primary_strategy_from_checks,
            )
            checkbox.grid(row=3 + index // 2, column=index % 2, sticky="w", pady=1)
            self._describe_widget(checkbox, f"Include {strategy_name} in selected-strategy runs.")

        button_row = 3 + (len(self.strategy_names) + 1) // 2
        all_button = ttk.Button(strategy_frame, text="All", command=lambda: self._set_strategy_checks(True))
        all_button.grid(
            row=button_row,
            column=0,
            sticky="ew",
            pady=(8, 0),
            padx=(0, 3),
        )
        self._describe_widget(all_button, "Select every strategy checkbox.")
        none_button = ttk.Button(strategy_frame, text="None", command=lambda: self._set_strategy_checks(False))
        none_button.grid(
            row=button_row,
            column=1,
            sticky="ew",
            pady=(8, 0),
            padx=(3, 0),
        )
        self._describe_widget(none_button, "Clear every strategy checkbox.")

        market_frame = ttk.LabelFrame(parent, text="Market", padding=10)
        market_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        market_frame.columnconfigure(1, weight=1)
        self._describe_widget(market_frame, "Market symbol controls for data-only backtests and comparisons.")
        market_label = ttk.Label(market_frame, text="Symbol(s)")
        market_label.grid(row=0, column=0, sticky="w", pady=2)
        market_selector = ttk.Combobox(
            market_frame,
            textvariable=self.market_var,
            values=MARKET_SYMBOLS,
            state="normal",
        )
        market_selector.grid(row=0, column=1, sticky="ew", pady=2, padx=(6, 0))
        market_description = (
            "Choose a built-in market symbol or type any Alpaca-supported stock/ETF symbol. "
            "For Run Market Matrix, type comma-separated symbols such as SPY,QQQ,GLD.\n\n"
            f"{MARKET_DESCRIPTION_TEXT}"
        )
        self._describe_widget(market_label, market_description)
        self._describe_widget(market_selector, market_description)

        strategy_params_frame = ttk.LabelFrame(parent, text="Strategy Parameters", padding=10)
        strategy_params_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        strategy_params_frame.columnconfigure(0, weight=1)
        self._describe_widget(strategy_params_frame, "Editable parameters used for each strategy when it is selected for a run.")
        self._build_strategy_parameter_editor(strategy_params_frame)

        dates_frame = ttk.LabelFrame(parent, text="Dates", padding=10)
        dates_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        dates_frame.columnconfigure(1, weight=1)
        self._describe_widget(dates_frame, "Date-range controls for backtests, comparisons, and diagnostics.")
        preset_label = ttk.Label(dates_frame, text="Preset")
        preset_label.grid(row=0, column=0, sticky="w", pady=2)
        preset_selector = ttk.Combobox(
            dates_frame,
            textvariable=self.preset_var,
            values=list(DATE_PRESETS),
            state="readonly",
        )
        preset_selector.grid(row=0, column=1, sticky="ew", pady=2)
        apply_button = ttk.Button(dates_frame, text="Apply", command=self._apply_date_preset)
        apply_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))
        self._describe_widget(preset_label, "Named date-range presets for common market periods.")
        self._describe_widget(preset_selector, "Choose a preset date range to copy into the date fields.")
        self._describe_widget(apply_button, "Apply the selected date preset to the start and end fields.")
        self._add_date_selector(dates_frame, 1, "Start", self.start_var, "Backtest start date in YYYY-MM-DD format.", allow_blank=False)
        self._add_date_selector(dates_frame, 2, "End", self.end_var, "Optional backtest end date in YYYY-MM-DD format; blank means latest available.", allow_blank=True)
        self._add_date_selector(dates_frame, 3, "Split", self.split_var, "Optional train/test split date used for multi-strategy comparisons.", allow_blank=True)

        params_frame = ttk.LabelFrame(parent, text="Model Inputs", padding=10)
        params_frame.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        self._describe_widget(params_frame, "Backtest model inputs for capital, exposure, transaction costs, and slippage.")
        self._add_labeled_entry(params_frame, 0, "Initial cash", self.initial_cash_var, "Starting portfolio value for each backtest.")
        self._add_labeled_entry(params_frame, 1, "Exposure", self.exposure_var, "Portfolio exposure while a strategy is long; use 0.25 for twenty-five percent.")
        self._add_labeled_entry(params_frame, 2, "Transaction bps", self.transaction_cost_var, "Estimated transaction cost in basis points per position change.")
        self._add_labeled_entry(params_frame, 3, "Slippage bps", self.slippage_var, "Estimated slippage in basis points per position change.")

        diagnostics_frame = ttk.LabelFrame(parent, text="Diagnostics", padding=10)
        diagnostics_frame.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        diagnostics_frame.columnconfigure(1, weight=1)
        self._describe_widget(diagnostics_frame, "Trade diagnostic parameters for missed rallies, whipsaws, and suspicious moves.")
        self._add_labeled_entry(diagnostics_frame, 0, "Top up days", self.top_days_var, "Number of strongest market days to inspect for missed rallies.")
        self._add_labeled_entry(diagnostics_frame, 1, "Whipsaw days", self.short_trade_days_var, "Maximum holding days for a losing trade to count as a whipsaw.")
        self._add_labeled_entry(diagnostics_frame, 2, "Large move", self.suspicious_move_var, "Absolute daily market move threshold for suspicious-move diagnostics.")

        actions_frame = ttk.LabelFrame(parent, text="Run", padding=10)
        actions_frame.grid(row=8, column=0, sticky="ew", pady=(0, 10))
        actions_frame.columnconfigure(0, weight=1)
        actions_frame.columnconfigure(1, weight=1)
        self._describe_widget(actions_frame, "Toggle which run aspects should execute, then run all enabled aspects together.")

        for row, (key, label, description) in enumerate(RUN_ASPECTS):
            checkbox = ttk.Checkbutton(
                actions_frame,
                text=label,
                variable=self.run_aspect_vars[key],
            )
            checkbox.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            self._describe_widget(checkbox, description)

        self._add_action_button(
            actions_frame,
            "Run Enabled Aspects",
            self._run_enabled_aspects,
            "Run every checked aspect in sequence for the selected strategy set.",
            primary=True,
        ).grid(row=len(RUN_ASPECTS), column=0, columnspan=2, sticky="ew", pady=(8, 3))

        all_aspects_button = ttk.Button(actions_frame, text="All Aspects", command=lambda: self._set_run_aspects(True))
        all_aspects_button.grid(row=len(RUN_ASPECTS) + 1, column=0, sticky="ew", pady=3, padx=(0, 3))
        self._describe_widget(all_aspects_button, "Turn on every run aspect toggle.")

        default_aspect_button = ttk.Button(actions_frame, text="Backtest Only", command=self._set_default_run_aspects)
        default_aspect_button.grid(row=len(RUN_ASPECTS) + 1, column=1, sticky="ew", pady=3, padx=(3, 0))
        self._describe_widget(default_aspect_button, "Turn on only the Backtest / Compare aspect.")

        self._add_action_button(
            actions_frame,
            "Validate Strategies",
            self._validate_strategies,
            "Check discovered strategies for load errors, editable parameters, signal output, and decision output.",
        ).grid(row=len(RUN_ASPECTS) + 2, column=0, columnspan=2, sticky="ew", pady=(8, 3))

        self._add_action_button(
            actions_frame,
            "Run Edge Lab",
            self._run_edge_analysis,
            "Score selected strategies for edge evidence using walk-forward, market breadth, regime, cost, and parameter tests.",
        ).grid(row=len(RUN_ASPECTS) + 3, column=0, columnspan=2, sticky="ew", pady=3)

        presets_frame = ttk.LabelFrame(parent, text="Presets", padding=10)
        presets_frame.grid(row=9, column=0, sticky="ew", pady=(0, 10))
        presets_frame.columnconfigure(0, weight=1)
        presets_frame.columnconfigure(1, weight=1)
        self._describe_widget(presets_frame, "Save or load the current GUI run configuration as a JSON preset.")
        save_preset_button = self._add_action_button(
            presets_frame,
            "Save Preset",
            self._save_run_preset,
            "Save selected strategies, strategy params, dates, symbols, run toggles, and model inputs.",
        )
        save_preset_button.grid(row=0, column=0, sticky="ew", pady=3, padx=(0, 3))
        load_preset_button = self._add_action_button(
            presets_frame,
            "Load Preset",
            self._load_run_preset,
            "Load a previously saved run preset JSON file.",
        )
        load_preset_button.grid(row=0, column=1, sticky="ew", pady=3, padx=(3, 0))

        exports_frame = ttk.LabelFrame(parent, text="Save", padding=10)
        exports_frame.grid(row=10, column=0, sticky="ew")
        exports_frame.columnconfigure(0, weight=1)
        self._describe_widget(exports_frame, "Save controls for charts, summaries, diagnostics, and logs.")
        self.save_plot_button = self._add_export_button(exports_frame, "Save Plot PNG", self._save_plot, "Save the current equity-curve chart as PNG, PDF, or SVG.")
        self.save_plot_button.grid(row=0, column=0, sticky="ew", pady=3)
        self.save_summary_button = self._add_export_button(exports_frame, "Save Summary CSV", self._save_summary, "Save the current strategy summary table as a CSV file.")
        self.save_summary_button.grid(row=1, column=0, sticky="ew", pady=3)
        self.save_reports_button = self._add_export_button(exports_frame, "Save Diagnostics CSVs", self._save_diagnostics, "Save trade, missed-day, whipsaw, and suspicious-move diagnostics as CSV files.")
        self.save_reports_button.grid(row=2, column=0, sticky="ew", pady=3)
        save_log_button = ttk.Button(exports_frame, text="Save Log", command=self._save_log)
        save_log_button.grid(row=3, column=0, sticky="ew", pady=(10, 3))
        self._describe_widget(save_log_button, "Save the Run Log tab contents as a text file.")

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
            card_description = f"{label} metric for the latest backtest, comparison, scorecard, or analysis result."
            name_label = ttk.Label(card, text=label, style="MetricName.TLabel")
            name_label.grid(row=0, column=0, sticky="w")
            value_label = ttk.Label(card, textvariable=self.metric_vars[key], style="MetricValue.TLabel")
            value_label.grid(row=1, column=0, sticky="w", pady=(5, 0))
            self._describe_widget(card, card_description)
            self._describe_widget(name_label, card_description)
            self._describe_widget(value_label, card_description)

        plot_shell = ttk.Frame(tab, style="Panel.TFrame", padding=12)
        plot_shell.grid(row=1, column=0, sticky="nsew")
        plot_shell.columnconfigure(0, weight=1)
        plot_shell.rowconfigure(1, weight=1)

        header = ttk.Frame(plot_shell, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        plot_title = ttk.Label(header, text="Equity Curve", background="#f8fafc", foreground="#0f172a", font=("Segoe UI Semibold", 14))
        plot_title.grid(row=0, column=0, sticky="w")
        plot_help = ttk.Label(
            header,
            text="Scroll to zoom, click a line to inspect values, or use the toolbar for pan/box zoom.",
            background="#f8fafc",
            foreground="#64748b",
        )
        plot_help.grid(row=1, column=0, sticky="w")
        last_year_button = ttk.Button(header, text="Last 1Y", command=lambda: self._focus_recent_plot(252))
        last_year_button.grid(row=0, column=1, rowspan=2, padx=(8, 3))
        full_range_button = ttk.Button(header, text="Full Range", command=self._reset_plot_view)
        full_range_button.grid(row=0, column=2, rowspan=2, padx=3)
        clear_marker_button = ttk.Button(header, text="Clear Marker", command=self._clear_plot_marker)
        clear_marker_button.grid(row=0, column=3, rowspan=2, padx=(3, 0))
        self._describe_widget(plot_title, "Equity-curve chart for the latest plotted strategy result.")
        self._describe_widget(plot_help, "Chart instructions: scroll to zoom, click a line to inspect values, or use the Matplotlib toolbar.")
        self._describe_widget(last_year_button, "Zoom the chart to the most recent one trading year.")
        self._describe_widget(full_range_button, "Reset the chart to the full available date range.")
        self._describe_widget(clear_marker_button, "Remove the selected point marker and annotation from the chart.")

        self.plot_frame = ttk.Frame(plot_shell, style="Panel.TFrame")
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self._describe_widget(self.plot_frame, "Chart region displaying portfolio equity curves and drawdown.")
        self.toolbar_frame = ttk.Frame(plot_shell, style="Panel.TFrame")
        self.toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._describe_widget(self.toolbar_frame, "Matplotlib toolbar for navigating and saving the embedded chart.")
        self._clear_plot()

    def _build_summary_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Summary")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        header = ttk.Frame(tab, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        scorecard_title = ttk.Label(header, text="Strategy Scorecard", background="#edf2f7", foreground="#0f172a", font=("Segoe UI Semibold", 14))
        scorecard_title.grid(row=0, column=0, sticky="w")
        scorecard_help = ttk.Label(
            header,
            text="Score is a heuristic: excess return, Sharpe, drawdown, missed rallies, and turnover.",
            background="#edf2f7",
            foreground="#64748b",
        )
        scorecard_help.grid(row=1, column=0, sticky="w")
        self._describe_widget(scorecard_title, "Summary table title for strategy scores and performance metrics.")
        self._describe_widget(scorecard_help, "Score explanation: combines excess return, Sharpe, drawdown, missed rallies, and turnover.")

        self.summary_tree = self._create_tree(tab, "Summary table of strategy performance metrics.")
        self.summary_tree.grid(row=1, column=0, sticky="nsew")

    def _build_edge_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Edge Lab")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        edge_notebook = ttk.Notebook(tab)
        edge_notebook.grid(row=0, column=0, sticky="nsew")
        self._describe_widget(edge_notebook, "Edge Lab detail tabs for market breadth, regime consistency, parameter stability, and cost stress.")

        for key, title, description in [
            ("market", "Markets", "Market-by-market edge evidence for each checked strategy."),
            ("regime", "Regimes", "Regime-by-regime edge evidence on the primary market."),
            ("parameters", "Parameters", "Parameter sensitivity checks around the selected strategy parameter values."),
            ("costs", "Costs", "Configured-cost and stressed-cost backtests on the primary market."),
        ]:
            frame = ttk.Frame(edge_notebook, style="App.TFrame", padding=8)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            edge_notebook.add(frame, text=title)
            tree_container = self._create_tree(frame, description)
            tree_container.grid(row=0, column=0, sticky="nsew")
            self.edge_trees[key] = tree_container
        self._clear_edge_details()

    def _build_diagnostics_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="App.TFrame", padding=14)
        self.notebook.add(tab, text="Diagnostics")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        diagnostics_notebook = ttk.Notebook(tab)
        diagnostics_notebook.grid(row=0, column=0, sticky="nsew")
        self._describe_widget(diagnostics_notebook, "Diagnostics result tabs for trades, missed best days, whipsaws, and suspicious moves.")

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
            tree_container = self._create_tree(frame, f"{title} diagnostics table.")
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
        self._describe_widget(self.output, "Run log showing task progress, save paths, validation errors, and unexpected errors.")

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        description: str,
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=16)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        self._describe_widget(label_widget, description)
        self._describe_widget(entry, description)

    def _add_date_selector(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        description: str,
        allow_blank: bool,
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=3)

        entry = ttk.Entry(parent, textvariable=variable, width=16, state="readonly")
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        self._describe_widget(entry, description)

        pick_button = ttk.Button(
            parent,
            text="Pick",
            command=lambda: self._open_date_picker(pick_button, variable, allow_blank=allow_blank),
        )
        pick_button.grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=3)
        self._describe_widget(pick_button, f"Open calendar picker for {label.lower()} date.")

    def _open_date_picker(self, anchor: tk.Widget, variable: tk.StringVar, allow_blank: bool) -> None:
        DatePickerPopup(anchor, variable, allow_blank=allow_blank)

    def _add_action_button(
        self,
        parent: ttk.Frame,
        text: str,
        command: Callable[[], None],
        description: str,
        primary: bool = False,
    ) -> ttk.Button:
        style = "Primary.TButton" if primary else "TButton"
        button = ttk.Button(parent, text=text, command=command, style=style)
        self._describe_widget(button, description)
        self._action_buttons.append(button)
        return button

    def _add_export_button(self, parent: ttk.Frame, text: str, command: Callable[[], None], description: str) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self._describe_widget(button, description)
        self._export_buttons.append(button)
        return button

    def _create_tree(self, parent: ttk.Frame, description: str) -> ttk.Frame:
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
        self._describe_widget(container, description)
        self._describe_widget(tree, description)
        self._describe_widget(y_scroll, f"Vertical scrollbar for {description.lower()}")
        self._describe_widget(x_scroll, f"Horizontal scrollbar for {description.lower()}")
        return container

    def _set_strategy_checks(self, checked: bool) -> None:
        for variable in self.strategy_checks.values():
            variable.set(checked)
        self._sync_primary_strategy_from_checks()

    def _set_run_aspects(self, checked: bool) -> None:
        for variable in self.run_aspect_vars.values():
            variable.set(checked)

    def _set_default_run_aspects(self) -> None:
        for key, variable in self.run_aspect_vars.items():
            variable.set(key == "selected")

    def _market_symbols(self) -> list[str]:
        raw_symbols = self.market_var.get().replace(";", ",").split(",")
        symbols = [normalize_symbol(symbol) for symbol in raw_symbols if symbol.strip()]
        if not symbols:
            raise ValueError("Enter at least one market symbol.")
        return list(dict.fromkeys(symbols))

    def _primary_market_symbol(self) -> str:
        return self._market_symbols()[0]

    def _reset_strategy_params(self, strategy_name: str) -> None:
        for spec in strategy_parameter_specs(strategy_name):
            self.strategy_param_vars[strategy_name][spec.name].set(_format_param_value(spec.default))
        self._log(f"Reset {strategy_name} parameters to defaults.")

    def _selected_strategy_names(self) -> list[str]:
        return [name for name, variable in self.strategy_checks.items() if variable.get()]

    def _primary_strategy_name(self) -> str:
        strategy_names = self._selected_strategy_names()
        if len(strategy_names) == 1:
            return strategy_names[0]
        return self.strategy_var.get()

    def _sync_primary_strategy_from_checks(self) -> None:
        strategy_names = self._selected_strategy_names()
        if len(strategy_names) == 1:
            self.strategy_var.set(strategy_names[0])

    def _strategy_params_for(self, strategy_name: str) -> dict[str, Any]:
        raw_params = {
            param_name: variable.get().strip()
            for param_name, variable in self.strategy_param_vars.get(strategy_name, {}).items()
        }
        try:
            return normalize_strategy_params(strategy_name, raw_params)
        except ValueError as error:
            raise ValueError(f"{strategy_name}: {error}") from error

    def _strategy_params_by_name(self, strategy_names: list[str]) -> dict[str, dict[str, Any]]:
        return {strategy_name: self._strategy_params_for(strategy_name) for strategy_name in strategy_names}

    def _apply_date_preset(self) -> None:
        start, end = DATE_PRESETS[self.preset_var.get()]
        self.start_var.set(start)
        self.end_var.set(end)
        self._log(f"Applied date preset: {self.preset_var.get()} ({start} to {end or 'latest'}).")

    def _selected_run_aspects(self) -> list[str]:
        return [key for key, variable in self.run_aspect_vars.items() if variable.get()]

    def _run_enabled_aspects(self) -> None:
        try:
            strategy_names = self._selected_strategy_names()
            if not strategy_names:
                raise ValueError("Select at least one strategy to run.")

            selected_aspects = self._selected_run_aspects()
            if not selected_aspects:
                raise ValueError("Select at least one run aspect.")

            run_plan: list[tuple[str, str, Callable[[], Any]]] = []

            if "selected" in selected_aspects:
                if len(strategy_names) == 1:
                    strategy_name = strategy_names[0]
                    self.strategy_var.set(strategy_name)
                    request = self._backtest_request(strategy_name)
                    run_plan.append(
                        (
                            "Backtest / Compare",
                            "backtest",
                            lambda request=request: run_single_backtest(request),
                        )
                    )
                else:
                    request = self._comparison_request(include_split=True, strategy_names=strategy_names)
                    run_plan.append(
                        (
                            "Backtest / Compare",
                            "comparison",
                            lambda request=request: run_strategy_comparison(request),
                        )
                    )

            if "regime" in selected_aspects:
                base_request = self._comparison_request(include_split=False, strategy_names=strategy_names)
                run_plan.append(
                    (
                        "Regime Scorecard",
                        "regime",
                        lambda base_request=base_request: self._run_regime_worker(base_request),
                    )
                )

            if "market" in selected_aspects:
                symbols = self._market_symbols()
                base_request = self._comparison_request(include_split=False, strategy_names=strategy_names)
                run_plan.append(
                    (
                        "Market Matrix",
                        "market_matrix",
                        lambda base_request=base_request, symbols=symbols: self._run_market_matrix_worker(base_request, symbols),
                    )
                )

            if "edge" in selected_aspects:
                request = self._edge_analysis_request(strategy_names)
                run_plan.append(
                    (
                        "Edge Lab",
                        "edge_analysis",
                        lambda request=request: run_edge_analysis(request),
                    )
                )

            if "analysis" in selected_aspects:
                request = self._trade_analysis_request(self._primary_strategy_name())
                run_plan.append(
                    (
                        "Trade Diagnostics",
                        "analysis",
                        lambda request=request: run_trade_analysis(request),
                    )
                )
        except ValueError as error:
            self._show_validation_error(error)
            return

        aspect_text = ", ".join(label for label, _kind, _worker in run_plan)
        self._run_in_background(
            f"Running enabled aspects: {aspect_text}...",
            "run_bundle",
            lambda run_plan=run_plan: self._run_aspect_bundle_worker(run_plan),
        )

    def _validate_strategies(self) -> None:
        self._run_in_background(
            "Validating discovered strategies...",
            "strategy_validation",
            lambda: validation_results_to_frame(validate_all_strategies()),
        )

    def _run_edge_analysis(self) -> None:
        try:
            strategy_names = self._selected_strategy_names()
            if not strategy_names:
                raise ValueError("Select at least one strategy to run.")
            request = self._edge_analysis_request(strategy_names)
        except ValueError as error:
            self._show_validation_error(error)
            return

        self._run_in_background("Running Edge Lab...", "edge_analysis", lambda: run_edge_analysis(request))

    def _run_selected_strategies(self) -> None:
        try:
            strategy_names = self._selected_strategy_names()
            if not strategy_names:
                raise ValueError("Select at least one strategy to run.")

            if len(strategy_names) == 1:
                strategy_name = strategy_names[0]
                self.strategy_var.set(strategy_name)
                request = self._backtest_request(strategy_name)
                self._run_in_background(
                    f"Running {strategy_name} backtest...",
                    "backtest",
                    lambda: run_single_backtest(request),
                )
                return

            request = self._comparison_request(include_split=True, strategy_names=strategy_names)
        except ValueError as error:
            self._show_validation_error(error)
            return

        self._run_in_background(
            f"Comparing {len(request.strategy_names)} selected strategies...",
            "comparison",
            lambda: run_strategy_comparison(request),
        )

    def _run_regime_pack(self) -> None:
        try:
            base_request = self._comparison_request(include_split=False)
        except ValueError as error:
            self._show_validation_error(error)
            return
        self._run_in_background("Running regime scorecard...", "regime", lambda: self._run_regime_worker(base_request))

    def _run_market_matrix(self) -> None:
        try:
            symbols = self._market_symbols()
            strategy_names = self._selected_strategy_names()
            if not strategy_names:
                raise ValueError("Select at least one strategy to run.")
            base_request = self._comparison_request(include_split=False, strategy_names=strategy_names)
        except ValueError as error:
            self._show_validation_error(error)
            return

        self._run_in_background(
            f"Running market matrix: {len(strategy_names)} strategies across {len(symbols)} markets...",
            "market_matrix",
            lambda: self._run_market_matrix_worker(base_request, symbols),
        )

    def _run_trade_analysis(self) -> None:
        try:
            request = self._trade_analysis_request(self._primary_strategy_name())
        except ValueError as error:
            self._show_validation_error(error)
            return
        self._run_in_background("Analyzing trades...", "analysis", lambda: run_trade_analysis(request))

    def _run_aspect_bundle_worker(
        self,
        run_plan: list[tuple[str, str, Callable[[], Any]]],
    ) -> list[TaskResult]:
        results = []
        for label, kind, worker in run_plan:
            try:
                results.append(TaskResult(kind=kind, payload=worker()))
            except Exception as error:  # noqa: BLE001 - keep later selected aspects running.
                details = "".join(traceback.format_exception_only(type(error), error)).strip()
                results.append(
                    TaskResult(
                        kind="aspect_error",
                        payload={"label": label, "error": details},
                    )
                )
        return results

    def _run_regime_worker(self, base_request: StrategyComparisonRequest) -> dict[str, Any]:
        summaries = []
        comparisons: list[StrategyComparisonResult] = []

        for period_name, start, end in REGIME_PACK:
            request = StrategyComparisonRequest(
                strategy_names=base_request.strategy_names,
                symbol=base_request.symbol,
                start=start,
                end=end,
                initial_cash=base_request.initial_cash,
                exposure=base_request.exposure,
                transaction_cost_bps=base_request.transaction_cost_bps,
                slippage_bps=base_request.slippage_bps,
                split_date=None,
                strategy_params_by_name=base_request.strategy_params_by_name,
            )
            comparison = run_strategy_comparison(request)
            comparisons.append(comparison)
            summary = _summary_with_score(comparison.full_summary)
            summary.insert(0, "period", period_name)
            summaries.append(summary)

        combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        return {"summary": combined, "comparisons": comparisons}

    def _run_market_matrix_worker(
        self,
        base_request: StrategyComparisonRequest,
        symbols: list[str],
    ) -> dict[str, Any]:
        summaries = []
        comparisons: list[StrategyComparisonResult] = []

        for symbol in symbols:
            request = StrategyComparisonRequest(
                strategy_names=base_request.strategy_names,
                symbol=symbol,
                start=base_request.start,
                end=base_request.end,
                initial_cash=base_request.initial_cash,
                exposure=base_request.exposure,
                transaction_cost_bps=base_request.transaction_cost_bps,
                slippage_bps=base_request.slippage_bps,
                split_date=None,
                strategy_params_by_name=base_request.strategy_params_by_name,
            )
            comparison = run_strategy_comparison(request)
            comparisons.append(comparison)
            summary = _summary_with_score(comparison.full_summary)
            summary.insert(0, "symbol", symbol)
            summaries.append(summary)

        combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        return {"summary": combined, "comparisons": comparisons, "symbols": symbols}

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
        if task_result.kind == "run_bundle":
            self._handle_run_bundle(task_result.payload)
            return
        if task_result.kind == "aspect_error":
            self._handle_aspect_error(task_result.payload)
            return
        if task_result.kind == "strategy_validation":
            self._handle_strategy_validation(task_result.payload)
            return
        if task_result.kind == "edge_analysis":
            self._handle_edge_analysis(task_result.payload)
            return
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
        if task_result.kind == "market_matrix":
            self._handle_market_matrix(task_result.payload)
            return
        self._log(f"Unknown task result type: {task_result.kind}")

    def _handle_run_bundle(self, results: list[TaskResult]) -> None:
        error_count = sum(1 for result in results if result.kind == "aspect_error")
        for result in results:
            self._handle_task_result(result)
        if error_count:
            self._log(f"Enabled run complete with {error_count} failed aspect(s).")
            messagebox.showwarning("Pluto Trader", f"{error_count} run aspect(s) failed. Check the Run Log tab.")
            return
        self._log(f"Enabled run complete: {len(results)} aspect(s) finished.")

    def _handle_aspect_error(self, payload: dict[str, str]) -> None:
        self._log(f"{payload['label']} failed: {payload['error']}")

    def _handle_strategy_validation(self, report: pd.DataFrame) -> None:
        self.current_results = []
        self.current_summary = report
        self.current_analysis = None
        self._clear_plot("Strategy validation complete. Run a backtest or comparison to render an equity curve.")
        self._populate_summary(report)
        self._clear_edge_details()
        self._clear_diagnostics()
        self._set_export_buttons_enabled(True)

        status_counts = report["status"].value_counts().to_dict() if "status" in report.columns else {}
        passed = int(status_counts.get("pass", 0))
        warned = int(status_counts.get("warn", 0))
        failed = int(status_counts.get("fail", 0))
        total = int(len(report))
        self.metric_vars["best"].set("Validation")
        self.metric_vars["score"].set(f"{passed}/{total} pass")
        self.metric_vars["return"].set(f"{failed} fail")
        self.metric_vars["drawdown"].set(f"{warned} warn")
        self.metric_vars["sharpe"].set("-")
        self.metric_vars["trades"].set(str(total))

        self._log(f"Strategy validation complete: {passed} passed, {warned} warnings, {failed} failed.")
        if failed or warned:
            for _, row in report[report["status"].isin(["fail", "warn"])].iterrows():
                detail = row["errors"] if row["status"] == "fail" else row["warnings"]
                self._log(f"Strategy validation {row['status']}: {row['strategy']}: {detail}")
        if failed:
            messagebox.showwarning("Pluto Trader", f"{failed} strategy validation check(s) failed. See the Summary and Run Log tabs.")
        self.notebook.select(1)

    def _handle_edge_analysis(self, result: EdgeAnalysisResult) -> None:
        self.current_results = []
        self.current_summary = result.summary
        self.current_analysis = None
        self._clear_plot("Edge Lab complete. Run a backtest or comparison to render a specific equity curve.")
        self._populate_summary(result.summary)
        self._populate_edge_details(result)
        self._clear_diagnostics()
        self._set_export_buttons_enabled(True)

        if result.summary.empty:
            self._update_metric_cards(pd.DataFrame())
            self._log("Edge Lab complete: no candidate rows returned.")
            self.notebook.select(1)
            return

        best = result.summary.sort_values("edge_score", ascending=False).iloc[0]
        self.metric_vars["best"].set(str(best.get("strategy", "-")))
        self.metric_vars["score"].set(_format_cell("edge_score", best.get("edge_score", "-")))
        self.metric_vars["return"].set(_format_cell("test_excess", best.get("test_excess", "-")))
        self.metric_vars["drawdown"].set(_format_cell("market_pass_rate", best.get("market_pass_rate", "-")))
        self.metric_vars["sharpe"].set(_format_cell("cost_resilience", best.get("cost_resilience", "-")))
        self.metric_vars["trades"].set(str(best.get("verdict", "-")))
        self._log(
            f"Edge Lab complete: best={best['strategy']} verdict={best['verdict']} "
            f"edge_score={best['edge_score']:.1f} notes={best['notes']}."
        )
        rejects = int((result.summary["verdict"] == "reject").sum()) if "verdict" in result.summary.columns else 0
        candidates = int((result.summary["verdict"] == "candidate").sum()) if "verdict" in result.summary.columns else 0
        self._log(f"Edge Lab verdicts: {candidates} candidate(s), {rejects} reject(s).")
        self.notebook.select(1)

    def _handle_backtest(self, result) -> None:
        self.current_results = [result.result]
        self.current_summary = _summary_with_score(result.summary)
        self.current_analysis = None
        self._render_plot(self.current_results)
        self._populate_summary(self.current_summary)
        self._clear_edge_details()
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log(
            f"Backtest complete: {result.symbol} {result.result.display_name}, {result.start_date.date()} to "
            f"{result.end_date.date() if result.end_date else 'latest available'}."
        )
        self.notebook.select(0)

    def _handle_comparison(self, result: StrategyComparisonResult) -> None:
        self.current_results = result.full_results
        self.current_summary = _summary_with_score(result.full_summary)
        self.current_analysis = None
        self._render_plot(self.current_results)
        self._populate_summary(self.current_summary)
        self._clear_edge_details()
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log(
            f"Comparison complete: {result.symbol}, {len(result.strategy_names)} strategies, {result.start_date.date()} to "
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
                train_label = best_train.get("display_name", best_train["strategy"])
                test_label = best_test.get("display_name", best_test["strategy"])
                self._log(
                    f"Split check: train best={train_label} ({best_train['score']:.1f}), "
                    f"test best={test_label} ({best_test['score']:.1f})."
                )
        self.notebook.select(0)

    def _handle_regime(self, payload: dict[str, Any]) -> None:
        self.current_results = []
        self.current_summary = payload["summary"]
        self.current_analysis = None
        self._clear_plot("Regime scorecard complete. Run a backtest or comparison to render a specific equity curve.")
        self._populate_summary(self.current_summary)
        self._clear_edge_details()
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log("Regime scorecard complete.")

        if self.current_summary is not None and not self.current_summary.empty:
            for period, group in self.current_summary.groupby("period", sort=False):
                best = group.sort_values("score", ascending=False).iloc[0]
                best_label = best.get("display_name", best["strategy"])
                self._log(f"{period}: best={best_label} score={best['score']:.1f} return={best['total_return']:.2%}.")
        self.notebook.select(1)

    def _handle_market_matrix(self, payload: dict[str, Any]) -> None:
        self.current_results = []
        self.current_summary = payload["summary"]
        self.current_analysis = None
        self._clear_plot("Market matrix complete. Select one market and run selected strategies to render an equity curve.")
        self._populate_summary(self.current_summary)
        self._clear_edge_details()
        self._clear_diagnostics()
        self._update_metric_cards(self.current_summary)
        self._set_export_buttons_enabled(True)
        self._log("Market matrix complete.")

        if self.current_summary is not None and not self.current_summary.empty:
            for symbol, group in self.current_summary.groupby("symbol", sort=False):
                best = group.sort_values("score", ascending=False).iloc[0]
                best_label = best.get("display_name", best["strategy"])
                self._log(f"{symbol}: best={best_label} score={best['score']:.1f} return={best['total_return']:.2%}.")
        self.notebook.select(1)

    def _handle_analysis(self, result: TradeAnalysisResult) -> None:
        self.current_analysis = result
        self._populate_diagnostics(result)
        analysis_summary = _summary_with_score(
            pd.DataFrame(
                [
                    {
                        "strategy": result.strategy_name,
                        "display_name": result.result.display_name,
                        **result.result.metrics,
                    }
                ]
            )
        )
        self.current_summary = analysis_summary
        self.current_results = [result.result]
        self._render_plot(self.current_results)
        self._populate_summary(analysis_summary)
        self._clear_edge_details()
        self._update_metric_cards(analysis_summary)
        self._set_export_buttons_enabled(True)
        self._log(
            f"Trade analysis complete: {result.result.display_name}. "
            f"Trades={len(result.trades)}, whipsaws={len(result.whipsaws)}."
        )
        self.notebook.select(2)

    def _backtest_request(self, strategy_name: str | None = None) -> BacktestRequest:
        strategy_name = strategy_name or self.strategy_var.get()
        return BacktestRequest(
            strategy_name=strategy_name,
            symbol=self._primary_market_symbol(),
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            strategy_params=self._strategy_params_for(strategy_name),
        )

    def _comparison_request(self, include_split: bool, strategy_names: list[str] | None = None) -> StrategyComparisonRequest:
        strategy_names = strategy_names if strategy_names is not None else self._selected_strategy_names()
        if not strategy_names:
            raise ValueError("Select at least one strategy to run.")

        return StrategyComparisonRequest(
            strategy_names=strategy_names,
            symbol=self._primary_market_symbol(),
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            split_date=_optional_text(self.split_var) if include_split else None,
            strategy_params_by_name=self._strategy_params_by_name(strategy_names),
        )

    def _edge_analysis_request(self, strategy_names: list[str] | None = None) -> EdgeAnalysisRequest:
        strategy_names = strategy_names if strategy_names is not None else self._selected_strategy_names()
        if not strategy_names:
            raise ValueError("Select at least one strategy to run.")

        return EdgeAnalysisRequest(
            strategy_names=strategy_names,
            symbols=self._market_symbols(),
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            split_date=_optional_text(self.split_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            strategy_params_by_name=self._strategy_params_by_name(strategy_names),
        )

    def _trade_analysis_request(self, strategy_name: str | None = None) -> TradeAnalysisRequest:
        strategy_name = strategy_name or self.strategy_var.get()
        return TradeAnalysisRequest(
            strategy_name=strategy_name,
            symbol=self._primary_market_symbol(),
            start=_required_text(self.start_var, "Start date"),
            end=_optional_text(self.end_var),
            initial_cash=_float_value(self.initial_cash_var, "Initial cash"),
            exposure=_float_value(self.exposure_var, "Exposure"),
            transaction_cost_bps=_float_value(self.transaction_cost_var, "Transaction bps"),
            slippage_bps=_float_value(self.slippage_var, "Slippage bps"),
            top_days=_int_value(self.top_days_var, "Top up days"),
            short_trade_days=_int_value(self.short_trade_days_var, "Whipsaw days"),
            suspicious_move_threshold=_float_value(self.suspicious_move_var, "Large move"),
            strategy_params=self._strategy_params_for(strategy_name),
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
        canvas_widget = self.plot_canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        self._describe_widget(canvas_widget, "Interactive equity-curve and drawdown chart for the current results.")
        self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas, self.toolbar_frame, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.grid(row=0, column=0, sticky="w")
        self._describe_widget(self.plot_toolbar, "Matplotlib navigation toolbar for home, back, forward, pan, zoom, configure, and save actions.")
        self._remember_plot_limits()
        self._connect_plot_events()
        self._focus_recent_plot(252, log_action=False)
        self.save_plot_button.configure(state=tk.NORMAL)

    def _clear_plot(self, message: str = "No plot yet. Run a backtest or strategy comparison.") -> None:
        plot_canvas = getattr(self, "plot_canvas", None)
        if plot_canvas is not None:
            for connection_id in self.plot_connections:
                plot_canvas.mpl_disconnect(connection_id)
            self.plot_connections = []

            # Figure.clear() asks the canvas toolbar to update. Detach it first
            # so Matplotlib does not touch Tk buttons that are being destroyed.
            if hasattr(plot_canvas, "toolbar"):
                plot_canvas.toolbar = None

        self._clear_plot_marker(redraw=False)

        if self.current_figure is not None:
            figure_canvas = getattr(self.current_figure, "canvas", None)
            if figure_canvas is not None and hasattr(figure_canvas, "toolbar"):
                figure_canvas.toolbar = None
            self.current_figure.clear()
            self.current_figure = None

        if plot_canvas is not None:
            try:
                plot_canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
            self.plot_canvas = None

        if self.plot_toolbar is not None:
            try:
                self.plot_toolbar.destroy()
            except tk.TclError:
                pass
            self.plot_toolbar = None

        if hasattr(self, "toolbar_frame"):
            for child in self.toolbar_frame.winfo_children():
                try:
                    child.destroy()
                except tk.TclError:
                    pass
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
            self._describe_widget(self.plot_placeholder, message)
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
        extra_columns = [column for column in summary.columns if column not in columns]
        display = summary[columns + extra_columns].copy()
        self._populate_tree(self.summary_tree, display)

    def _populate_diagnostics(self, result: TradeAnalysisResult) -> None:
        self._populate_tree(self.diagnostic_trees["trades"], result.trades)
        self._populate_tree(self.diagnostic_trees["missed_days"], result.missed_days)
        self._populate_tree(self.diagnostic_trees["whipsaws"], result.whipsaws)
        self._populate_tree(self.diagnostic_trees["suspicious_moves"], result.suspicious_moves)

    def _populate_edge_details(self, result: EdgeAnalysisResult) -> None:
        self._populate_tree(self.edge_trees["market"], result.market_details)
        self._populate_tree(self.edge_trees["regime"], result.regime_details)
        self._populate_tree(self.edge_trees["parameters"], result.parameter_details)
        self._populate_tree(self.edge_trees["costs"], result.cost_details)

    def _clear_edge_details(self) -> None:
        for tree_container in self.edge_trees.values():
            self._populate_tree(tree_container, pd.DataFrame())

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
            anchor = (
                "w"
                if column in {
                    "symbol",
                    "period",
                    "strategy",
                    "display_name",
                    "date",
                    "status",
                    "editable_parameters",
                    "checks",
                    "warnings",
                    "errors",
                    "verdict",
                    "primary_symbol",
                    "notes",
                    "variant",
                    "changed_param",
                    "param_value",
                    "cost_case",
                }
                else "e"
            )
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
        self.metric_vars["best"].set(str(best.get("display_name", best.get("strategy", "-"))))
        self.metric_vars["score"].set(_format_cell("score", best.get("score", "-")))
        self.metric_vars["return"].set(_format_cell("total_return", best.get("total_return", "-")))
        self.metric_vars["drawdown"].set(_format_cell("max_drawdown", best.get("max_drawdown", "-")))
        self.metric_vars["sharpe"].set(_format_cell("sharpe", best.get("sharpe", "-")))
        self.metric_vars["trades"].set(_format_cell("trades", best.get("trades", "-")))

    def _save_run_preset(self) -> None:
        RUN_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save run preset",
            initialdir=str(RUN_PRESETS_DIR),
            initialfile=f"pluto_preset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        payload = self._run_preset_payload()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Saved run preset: {path}")

    def _load_run_preset(self) -> None:
        RUN_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load run preset",
            initialdir=str(RUN_PRESETS_DIR),
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self._apply_run_preset(payload)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self._show_validation_error(ValueError(f"Could not load preset: {error}"))
            return

        self._render_strategy_param_fields()
        self._log(f"Loaded run preset: {path}")

    def _run_preset_payload(self) -> dict[str, Any]:
        return {
            "version": RUN_PRESET_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "primary_strategy": self.strategy_var.get(),
            "strategy_checks": {
                strategy_name: bool(variable.get())
                for strategy_name, variable in self.strategy_checks.items()
            },
            "selected_strategies": self._selected_strategy_names(),
            "market_symbols": self.market_var.get(),
            "run_aspects": {
                aspect: bool(variable.get())
                for aspect, variable in self.run_aspect_vars.items()
            },
            "dates": {
                "preset": self.preset_var.get(),
                "start": self.start_var.get(),
                "end": self.end_var.get(),
                "split": self.split_var.get(),
            },
            "model_inputs": {
                "initial_cash": self.initial_cash_var.get(),
                "exposure": self.exposure_var.get(),
                "transaction_cost_bps": self.transaction_cost_var.get(),
                "slippage_bps": self.slippage_var.get(),
            },
            "diagnostics": {
                "top_days": self.top_days_var.get(),
                "short_trade_days": self.short_trade_days_var.get(),
                "suspicious_move_threshold": self.suspicious_move_var.get(),
            },
            "strategy_params": {
                strategy_name: {
                    param_name: variable.get()
                    for param_name, variable in param_vars.items()
                }
                for strategy_name, param_vars in self.strategy_param_vars.items()
            },
        }

    def _apply_run_preset(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Preset must contain a JSON object.")

        ignored_items: list[str] = []
        version = payload.get("version")
        if version not in {None, RUN_PRESET_VERSION}:
            ignored_items.append(f"version {version}")

        strategy_checks = payload.get("strategy_checks")
        if isinstance(strategy_checks, dict):
            for strategy_name, variable in self.strategy_checks.items():
                variable.set(bool(strategy_checks.get(strategy_name, False)))
            ignored_items.extend(
                f"strategy {strategy_name}"
                for strategy_name in sorted(set(strategy_checks) - set(self.strategy_checks))
            )
        else:
            selected_strategies = payload.get("selected_strategies", [])
            if isinstance(selected_strategies, list):
                selected_set = {str(strategy_name) for strategy_name in selected_strategies}
                for strategy_name, variable in self.strategy_checks.items():
                    variable.set(strategy_name in selected_set)
                ignored_items.extend(
                    f"strategy {strategy_name}"
                    for strategy_name in sorted(selected_set - set(self.strategy_checks))
                )

        primary_strategy = payload.get("primary_strategy")
        if isinstance(primary_strategy, str) and primary_strategy in self.strategy_names:
            self.strategy_var.set(primary_strategy)
        else:
            self._sync_primary_strategy_from_checks()
            if primary_strategy:
                ignored_items.append(f"primary strategy {primary_strategy}")

        market_symbols = payload.get("market_symbols")
        if isinstance(market_symbols, str):
            self.market_var.set(market_symbols)

        run_aspects = payload.get("run_aspects")
        if isinstance(run_aspects, dict):
            for aspect, variable in self.run_aspect_vars.items():
                if aspect in run_aspects:
                    variable.set(bool(run_aspects[aspect]))
            ignored_items.extend(
                f"run aspect {aspect}"
                for aspect in sorted(set(run_aspects) - set(self.run_aspect_vars))
            )

        self._apply_string_fields(
            payload.get("dates"),
            {
                "preset": self.preset_var,
                "start": self.start_var,
                "end": self.end_var,
                "split": self.split_var,
            },
        )
        self._apply_string_fields(
            payload.get("model_inputs"),
            {
                "initial_cash": self.initial_cash_var,
                "exposure": self.exposure_var,
                "transaction_cost_bps": self.transaction_cost_var,
                "slippage_bps": self.slippage_var,
            },
        )
        self._apply_string_fields(
            payload.get("diagnostics"),
            {
                "top_days": self.top_days_var,
                "short_trade_days": self.short_trade_days_var,
                "suspicious_move_threshold": self.suspicious_move_var,
            },
        )

        strategy_params = payload.get("strategy_params")
        if isinstance(strategy_params, dict):
            for strategy_name, params in strategy_params.items():
                if strategy_name not in self.strategy_param_vars:
                    ignored_items.append(f"params for {strategy_name}")
                    continue
                if not isinstance(params, dict):
                    ignored_items.append(f"params for {strategy_name}")
                    continue
                param_vars = self.strategy_param_vars[strategy_name]
                for param_name, value in params.items():
                    if param_name in param_vars:
                        param_vars[param_name].set(str(value))
                    else:
                        ignored_items.append(f"param {strategy_name}.{param_name}")

        if ignored_items:
            self._log("Preset ignored unavailable items: " + ", ".join(ignored_items))

    def _apply_string_fields(self, payload: Any, variables: dict[str, tk.StringVar]) -> None:
        if not isinstance(payload, dict):
            return
        for key, variable in variables.items():
            if key in payload and payload[key] is not None:
                variable.set(str(payload[key]))

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

    if column in {"score", "edge_score"}:
        return f"{float(value):.1f}"

    if column == "sharpe":
        return f"{float(value):.2f}"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _parse_picker_date(value: str) -> date | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return datetime.strptime(stripped, "%Y-%m-%d").date()
    except ValueError:
        return None


def _column_width(column: str) -> int:
    widths = {
        "period": 150,
        "symbol": 90,
        "status": 80,
        "verdict": 90,
        "primary_symbol": 90,
        "display_name": 220,
        "strategy": 150,
        "edge_score": 90,
        "full_excess": 110,
        "train_excess": 110,
        "test_excess": 110,
        "test_sharpe": 100,
        "test_max_drawdown": 120,
        "market_pass_rate": 110,
        "regime_pass_rate": 110,
        "cost_resilience": 120,
        "stressed_excess": 120,
        "parameter_stability": 120,
        "notes": 340,
        "variant": 200,
        "changed_param": 140,
        "param_value": 120,
        "cost_case": 110,
        "editable_parameters": 220,
        "checks": 180,
        "warnings": 320,
        "errors": 380,
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
