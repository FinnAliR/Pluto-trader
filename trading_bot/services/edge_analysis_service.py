"""Edge-analysis workflow for finding robust strategy candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trading_bot.backtesting.engine import results_to_summary_frame, run_backtest
from trading_bot.core.config import Settings, load_settings
from trading_bot.services.backtest_service import (
    DEFAULT_EXPOSURE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    load_daily_price_data,
    normalize_symbol,
    run_strategy_suite,
)
from trading_bot.strategies.registry import (
    create_strategy,
    normalize_strategy_name,
    normalize_strategy_params,
    strategy_parameter_specs,
)


EDGE_REGIME_PERIODS = [
    ("Full history", "2018-01-01", None),
    ("COVID crash", "2020-02-20", "2020-04-30"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("Recovery trend", "2023-01-01", None),
]

DEFAULT_STRESS_TRANSACTION_COST_BPS = 5.0
DEFAULT_STRESS_SLIPPAGE_BPS = 10.0
DEFAULT_PARAMETER_SHOCK = 0.20
MIN_SPLIT_ROWS = 40


@dataclass(frozen=True)
class EdgeAnalysisRequest:
    """Inputs for edge candidate analysis."""

    strategy_names: list[str]
    symbols: list[str] = field(default_factory=lambda: ["SPY"])
    start: str = DEFAULT_START_DATE
    end: str | None = None
    split_date: str | None = None
    initial_cash: float = DEFAULT_INITIAL_CASH
    exposure: float = DEFAULT_EXPOSURE
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    strategy_params_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    stress_transaction_cost_bps: float = DEFAULT_STRESS_TRANSACTION_COST_BPS
    stress_slippage_bps: float = DEFAULT_STRESS_SLIPPAGE_BPS
    parameter_shock: float = DEFAULT_PARAMETER_SHOCK


@dataclass(frozen=True)
class EdgeAnalysisResult:
    """Outputs from edge candidate analysis."""

    summary: pd.DataFrame
    market_details: pd.DataFrame
    regime_details: pd.DataFrame
    parameter_details: pd.DataFrame
    cost_details: pd.DataFrame
    split_date: datetime | None


def run_edge_analysis(
    request: EdgeAnalysisRequest,
    settings: Settings | None = None,
) -> EdgeAnalysisResult:
    """Run walk-forward, market-breadth, cost, and parameter robustness checks."""
    loaded_settings = settings or load_settings()
    strategy_names = [normalize_strategy_name(name) for name in request.strategy_names]
    if not strategy_names:
        raise ValueError("Select at least one strategy for edge analysis.")

    symbols = list(dict.fromkeys(normalize_symbol(symbol) for symbol in request.symbols))
    if not symbols:
        raise ValueError("Select at least one market symbol for edge analysis.")

    start_date = _parse_date(request.start)
    end_date = _parse_date(request.end) if request.end else None
    params_by_name = {
        normalize_strategy_name(name): params
        for name, params in request.strategy_params_by_name.items()
    }

    price_data_by_symbol = {
        symbol: load_daily_price_data(loaded_settings, start_date, end_date, symbol)
        for symbol in symbols
    }
    primary_symbol = symbols[0]
    primary_data = price_data_by_symbol[primary_symbol]
    split_date = _analysis_split_date(primary_data, request.split_date)

    market_details = _market_detail_frame(
        strategy_names=strategy_names,
        symbols=symbols,
        price_data_by_symbol=price_data_by_symbol,
        settings=loaded_settings,
        request=request,
        params_by_name=params_by_name,
    )
    full_primary = market_details[market_details["symbol"] == primary_symbol].copy()
    train_summary, test_summary = _walk_forward_summaries(
        strategy_names=strategy_names,
        price_data=primary_data,
        settings=loaded_settings,
        request=request,
        params_by_name=params_by_name,
        split_date=split_date,
    )
    regime_details = _regime_detail_frame(
        strategy_names=strategy_names,
        price_data=primary_data,
        settings=loaded_settings,
        request=request,
        params_by_name=params_by_name,
    )
    cost_details = _cost_detail_frame(
        strategy_names=strategy_names,
        price_data=primary_data,
        settings=loaded_settings,
        request=request,
        params_by_name=params_by_name,
        primary_symbol=primary_symbol,
    )
    parameter_details = _parameter_detail_frame(
        strategy_names=strategy_names,
        price_data=primary_data,
        settings=loaded_settings,
        request=request,
        params_by_name=params_by_name,
        primary_symbol=primary_symbol,
    )
    summary = _edge_summary_frame(
        strategy_names=strategy_names,
        primary_symbol=primary_symbol,
        full_primary=full_primary,
        train_summary=train_summary,
        test_summary=test_summary,
        market_details=market_details,
        regime_details=regime_details,
        cost_details=cost_details,
        parameter_details=parameter_details,
    )
    return EdgeAnalysisResult(
        summary=summary,
        market_details=market_details,
        regime_details=regime_details,
        parameter_details=parameter_details,
        cost_details=cost_details,
        split_date=split_date,
    )


def _market_detail_frame(
    strategy_names: list[str],
    symbols: list[str],
    price_data_by_symbol: dict[str, pd.DataFrame],
    settings: Settings,
    request: EdgeAnalysisRequest,
    params_by_name: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        summary = _run_summary(
            strategy_names=strategy_names,
            price_data=price_data_by_symbol[symbol],
            settings=settings,
            request=request,
            params_by_name=params_by_name,
        )
        summary.insert(0, "symbol", symbol)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _walk_forward_summaries(
    strategy_names: list[str],
    price_data: pd.DataFrame,
    settings: Settings,
    request: EdgeAnalysisRequest,
    params_by_name: dict[str, dict[str, Any]],
    split_date: datetime | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_date is None:
        return pd.DataFrame(), pd.DataFrame()

    train_data = price_data.loc[price_data.index <= split_date]
    test_data = price_data.loc[price_data.index > split_date]
    if len(train_data) < MIN_SPLIT_ROWS or len(test_data) < MIN_SPLIT_ROWS:
        return pd.DataFrame(), pd.DataFrame()

    train_summary = _run_summary(
        strategy_names=strategy_names,
        price_data=train_data,
        settings=settings,
        request=request,
        params_by_name=params_by_name,
    )
    test_summary = _run_summary(
        strategy_names=strategy_names,
        price_data=test_data,
        settings=settings,
        request=request,
        params_by_name=params_by_name,
    )
    return train_summary, test_summary


def _regime_detail_frame(
    strategy_names: list[str],
    price_data: pd.DataFrame,
    settings: Settings,
    request: EdgeAnalysisRequest,
    params_by_name: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for period_name, start, end in EDGE_REGIME_PERIODS:
        start_date = _parse_date(start)
        end_date = _parse_date(end) if end else None
        period_data = price_data.loc[price_data.index >= start_date]
        if end_date is not None:
            period_data = period_data.loc[period_data.index <= end_date]
        if len(period_data) < MIN_SPLIT_ROWS:
            continue

        summary = _run_summary(
            strategy_names=strategy_names,
            price_data=period_data,
            settings=settings,
            request=request,
            params_by_name=params_by_name,
        )
        summary.insert(0, "period", period_name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _cost_detail_frame(
    strategy_names: list[str],
    price_data: pd.DataFrame,
    settings: Settings,
    request: EdgeAnalysisRequest,
    params_by_name: dict[str, dict[str, Any]],
    primary_symbol: str,
) -> pd.DataFrame:
    normal = _run_summary(
        strategy_names=strategy_names,
        price_data=price_data,
        settings=settings,
        request=request,
        params_by_name=params_by_name,
    )
    normal.insert(0, "cost_case", "configured")
    stressed_request = EdgeAnalysisRequest(
        strategy_names=request.strategy_names,
        symbols=request.symbols,
        start=request.start,
        end=request.end,
        split_date=request.split_date,
        initial_cash=request.initial_cash,
        exposure=request.exposure,
        transaction_cost_bps=request.transaction_cost_bps + request.stress_transaction_cost_bps,
        slippage_bps=request.slippage_bps + request.stress_slippage_bps,
        strategy_params_by_name=request.strategy_params_by_name,
        stress_transaction_cost_bps=request.stress_transaction_cost_bps,
        stress_slippage_bps=request.stress_slippage_bps,
        parameter_shock=request.parameter_shock,
    )
    stressed = _run_summary(
        strategy_names=strategy_names,
        price_data=price_data,
        settings=settings,
        request=stressed_request,
        params_by_name=params_by_name,
    )
    stressed.insert(0, "cost_case", "stressed")
    frame = pd.concat([normal, stressed], ignore_index=True)
    frame.insert(0, "symbol", primary_symbol)
    return frame


def _parameter_detail_frame(
    strategy_names: list[str],
    price_data: pd.DataFrame,
    settings: Settings,
    request: EdgeAnalysisRequest,
    params_by_name: dict[str, dict[str, Any]],
    primary_symbol: str,
) -> pd.DataFrame:
    rows = []
    for strategy_name in strategy_names:
        variants = _parameter_variants(
            strategy_name=strategy_name,
            params=params_by_name.get(strategy_name, {}),
            settings=settings,
            shock=request.parameter_shock,
        )
        if not variants:
            rows.append(
                {
                    "symbol": primary_symbol,
                    "strategy": strategy_name,
                    "variant": "no numeric editable parameters",
                    "changed_param": "",
                    "param_value": "",
                    "excess_vs_same_exposure": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "passes_edge_gate": True,
                }
            )
            continue

        for variant_label, changed_param, param_value, params in variants:
            strategy = create_strategy(strategy_name, settings=settings, params=params)
            result = run_backtest(
                strategy=strategy,
                price_data=price_data,
                initial_cash=request.initial_cash,
                exposure=request.exposure,
                transaction_cost_bps=request.transaction_cost_bps,
                slippage_bps=request.slippage_bps,
            )
            summary = _with_excess(results_to_summary_frame([result]))
            row = summary.iloc[0].to_dict()
            row.update(
                {
                    "symbol": primary_symbol,
                    "variant": variant_label,
                    "changed_param": changed_param,
                    "param_value": param_value,
                    "passes_edge_gate": _passes_edge_gate(pd.Series(row)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _edge_summary_frame(
    strategy_names: list[str],
    primary_symbol: str,
    full_primary: pd.DataFrame,
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    market_details: pd.DataFrame,
    regime_details: pd.DataFrame,
    cost_details: pd.DataFrame,
    parameter_details: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for strategy_name in strategy_names:
        full_row = _row_for_strategy(full_primary, strategy_name)
        train_row = _row_for_strategy(train_summary, strategy_name)
        test_row = _row_for_strategy(test_summary, strategy_name)
        stressed_row = _row_for_strategy(cost_details[cost_details["cost_case"] == "stressed"], strategy_name)
        market_rows = market_details[market_details["strategy"] == strategy_name]
        regime_rows = regime_details[regime_details["strategy"] == strategy_name]
        parameter_rows = parameter_details[parameter_details["strategy"] == strategy_name]

        market_pass_rate = _pass_rate(market_rows)
        regime_pass_rate = _pass_rate(regime_rows)
        parameter_stability = _parameter_stability(parameter_rows)
        full_excess = _float(full_row.get("excess_vs_same_exposure", 0.0))
        stressed_excess = _float(stressed_row.get("excess_vs_same_exposure", 0.0))
        cost_resilience = _cost_resilience(full_excess, stressed_excess)
        oos_score = _quality_score(test_row if not test_row.empty else full_row)
        trade_score = _trade_count_score(_float(full_row.get("trades", 0.0)))

        edge_score = round(
            (oos_score * 0.30)
            + (market_pass_rate * 100 * 0.20)
            + (regime_pass_rate * 100 * 0.15)
            + (cost_resilience * 100 * 0.15)
            + (parameter_stability * 100 * 0.15)
            + (trade_score * 0.05),
            1,
        )
        test_excess = _float(test_row.get("excess_vs_same_exposure", full_excess))
        verdict = _verdict(edge_score, test_excess, market_pass_rate, stressed_excess)
        notes = _notes(
            test_row=test_row,
            test_excess=test_excess,
            market_pass_rate=market_pass_rate,
            regime_pass_rate=regime_pass_rate,
            cost_resilience=cost_resilience,
            parameter_stability=parameter_stability,
            trades=_float(full_row.get("trades", 0.0)),
        )

        rows.append(
            {
                "edge_score": edge_score,
                "verdict": verdict,
                "strategy": strategy_name,
                "display_name": full_row.get("display_name", strategy_name),
                "primary_symbol": primary_symbol,
                "full_excess": full_excess,
                "train_excess": _float(train_row.get("excess_vs_same_exposure", 0.0)),
                "test_excess": test_excess,
                "test_sharpe": _float(test_row.get("sharpe", full_row.get("sharpe", 0.0))),
                "test_max_drawdown": _float(test_row.get("max_drawdown", full_row.get("max_drawdown", 0.0))),
                "market_pass_rate": market_pass_rate,
                "positive_markets": int(sum(_passes_edge_gate(row) for _, row in market_rows.iterrows())),
                "markets_tested": int(len(market_rows)),
                "regime_pass_rate": regime_pass_rate,
                "regime_periods_tested": int(len(regime_rows)),
                "cost_resilience": cost_resilience,
                "stressed_excess": stressed_excess,
                "parameter_stability": parameter_stability,
                "parameter_variants_tested": int(len(parameter_rows)),
                "trades": _float(full_row.get("trades", 0.0)),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows).sort_values("edge_score", ascending=False)


def _run_summary(
    strategy_names: list[str],
    price_data: pd.DataFrame,
    settings: Settings,
    request: EdgeAnalysisRequest,
    params_by_name: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    results = run_strategy_suite(
        strategy_names=strategy_names,
        settings=settings,
        price_data=price_data,
        initial_cash=request.initial_cash,
        exposure=request.exposure,
        transaction_cost_bps=request.transaction_cost_bps,
        slippage_bps=request.slippage_bps,
        strategy_params_by_name=params_by_name,
    )
    return _with_excess(results_to_summary_frame(results))


def _with_excess(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["excess_vs_same_exposure"] = frame["total_return"] - frame["buy_hold_same_exposure_return"]
    return frame


def _parameter_variants(
    strategy_name: str,
    params: dict[str, Any],
    settings: Settings,
    shock: float,
) -> list[tuple[str, str, Any, dict[str, Any]]]:
    base_params = normalize_strategy_params(strategy_name, params, settings=settings)
    variants = []
    for spec in strategy_parameter_specs(strategy_name, settings=settings):
        if spec.value_type not in {int, float}:
            continue
        base_value = base_params.get(spec.name, spec.default)
        if isinstance(base_value, bool):
            continue
        try:
            base_number = float(base_value)
        except (TypeError, ValueError):
            continue

        for direction, label in [(-1, "lower"), (1, "higher")]:
            candidate = _shocked_value(spec, base_number, direction, shock)
            if candidate == base_value:
                continue
            variant_params = dict(base_params)
            variant_params[spec.name] = candidate
            variants.append(
                (
                    f"{spec.name} {label} {shock:.0%}",
                    spec.name,
                    candidate,
                    normalize_strategy_params(strategy_name, variant_params, settings=settings),
                )
            )
    return variants[:16]


def _shocked_value(spec, base_number: float, direction: int, shock: float) -> int | float:
    if base_number == 0:
        candidate = direction * shock
    else:
        candidate = base_number * (1 + (direction * shock))

    if spec.minimum is not None:
        candidate = max(candidate, spec.minimum)
    if spec.maximum is not None:
        candidate = min(candidate, spec.maximum)

    if spec.value_type is int:
        return int(round(candidate))
    return float(candidate)


def _analysis_split_date(price_data: pd.DataFrame, split_date: str | None) -> datetime | None:
    if price_data.empty:
        return None
    if split_date:
        return _parse_date(split_date)
    split_index = int(len(price_data.index) * 0.60)
    split_index = min(max(split_index, 1), len(price_data.index) - 2)
    return pd.Timestamp(price_data.index[split_index]).to_pydatetime()


def _row_for_strategy(frame: pd.DataFrame, strategy_name: str) -> pd.Series:
    if frame is None or frame.empty or "strategy" not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame[frame["strategy"] == strategy_name]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def _pass_rate(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 0.0
    passes = sum(_passes_edge_gate(row) for _, row in frame.iterrows())
    return passes / len(frame)


def _parameter_stability(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 1.0
    if "passes_edge_gate" in frame.columns:
        return float(frame["passes_edge_gate"].fillna(False).astype(bool).mean())
    return _pass_rate(frame)


def _passes_edge_gate(row: pd.Series) -> bool:
    return (
        _float(row.get("excess_vs_same_exposure", 0.0)) > 0
        and _float(row.get("sharpe", 0.0)) > 0
        and _float(row.get("max_drawdown", 0.0)) > -0.45
    )


def _quality_score(row: pd.Series) -> float:
    excess = _float(row.get("excess_vs_same_exposure", 0.0))
    sharpe = _float(row.get("sharpe", 0.0))
    max_drawdown = _float(row.get("max_drawdown", 0.0))
    score = 45 + (excess * 120) + (sharpe * 12) + (((0.35 + max_drawdown) / 0.35) * 20)
    return _clamp(score, 0, 100)


def _trade_count_score(trades: float) -> float:
    if trades < 3:
        return 20
    if trades < 10:
        return 50 + (trades - 3) * 6
    if trades <= 180:
        return 100
    if trades <= 350:
        return 75
    return 50


def _cost_resilience(normal_excess: float, stressed_excess: float) -> float:
    if normal_excess <= 0:
        return 0.0
    return _clamp(stressed_excess / normal_excess, 0.0, 1.0)


def _verdict(edge_score: float, test_excess: float, market_pass_rate: float, stressed_excess: float) -> str:
    if edge_score >= 70 and test_excess > 0 and market_pass_rate >= 0.50 and stressed_excess > 0:
        return "candidate"
    if edge_score >= 55 and test_excess > 0:
        return "watch"
    return "reject"


def _notes(
    test_row: pd.Series,
    test_excess: float,
    market_pass_rate: float,
    regime_pass_rate: float,
    cost_resilience: float,
    parameter_stability: float,
    trades: float,
) -> str:
    notes = []
    if test_row.empty:
        notes.append("no valid train/test split")
    if test_excess <= 0:
        notes.append("no out-of-sample excess")
    if market_pass_rate < 0.50:
        notes.append("weak market breadth")
    if regime_pass_rate < 0.50:
        notes.append("weak regime consistency")
    if cost_resilience < 0.50:
        notes.append("fragile after cost stress")
    if parameter_stability < 0.50:
        notes.append("fragile parameter sensitivity")
    if trades < 3:
        notes.append("too few trades")
    return "; ".join(notes) or "robust enough for paper tracking"


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
