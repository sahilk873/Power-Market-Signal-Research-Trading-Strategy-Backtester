import numpy as np
import pandas as pd

from .engine import BacktestResult


def compute_metrics(result: BacktestResult) -> dict:
    """Compute comprehensive performance metrics from a backtest result."""
    equity = result.equity_curve
    returns = result.returns
    pnl = result.pnl
    exposures = result.exposures
    trades = result.trades

    metrics: dict = {}

    # --- Return & risk ---
    total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
    metrics["total_return_pct"] = round(total_return * 100, 2)

    ann_return = (1 + total_return) ** (8760 / len(returns)) - 1 if len(returns) > 0 else 0.0
    metrics["annualised_return_pct"] = round(ann_return * 100, 2)

    ann_vol = returns.std() * np.sqrt(8760) if len(returns) > 0 else 0.0
    metrics["annualised_vol_pct"] = round(ann_vol * 100, 2)

    metrics["sharpe_ratio"] = (
        round(ann_return / ann_vol, 3) if ann_vol > 0 else 0.0
    )

    # --- Drawdown ---
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    metrics["max_drawdown_pct"] = round(dd.min() * 100, 2)
    metrics["avg_drawdown_pct"] = round(dd.mean() * 100, 2)

    in_dd = dd < 0
    if in_dd.any():
        dd_periods = (~in_dd).cumsum()
        dd_lengths = dd_periods[in_dd].groupby(dd_periods).size()
        metrics["avg_dd_duration_hours"] = round(dd_lengths.mean(), 1) if len(dd_lengths) > 0 else 0.0
        metrics["max_dd_duration_hours"] = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    else:
        metrics["avg_dd_duration_hours"] = 0.0
        metrics["max_dd_duration_hours"] = 0

    # --- Hit rate ---
    positive = (returns > 0).sum()
    total = len(returns)
    metrics["hit_rate_pct"] = round(positive / total * 100, 2) if total > 0 else 0.0

    # --- Profit factor ---
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    metrics["profit_factor"] = (
        round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")
    )

    # --- Turnover ---
    if len(trades) > 0:
        avg_trade_size = trades["size_change"].abs().mean()
        metrics["avg_trade_size_mw"] = round(avg_trade_size, 1)
        metrics["total_trades"] = len(trades)
    else:
        metrics["avg_trade_size_mw"] = 0.0
        metrics["total_trades"] = 0

    avg_exposure = exposures.mean(axis=1).mean()
    metrics["avg_exposure_mw"] = round(avg_exposure, 1)

    # --- Signal decay ---
    metrics["signal_decay"] = _compute_signal_decay(result)

    # --- Calmar ---
    metrics["calmar_ratio"] = (
        round(ann_return / abs(metrics["max_drawdown_pct"] / 100), 3)
        if metrics["max_drawdown_pct"] != 0
        else 0.0
    )

    # --- PnL stats ---
    metrics["total_pnl"] = round(pnl.sum(), 2)
    metrics["avg_hourly_pnl"] = round(pnl.mean(), 2)
    metrics["std_hourly_pnl"] = round(pnl.std(), 2)
    metrics["best_hour_pnl"] = round(pnl.max(), 2)
    metrics["worst_hour_pnl"] = round(pnl.min(), 2)

    # --- Win/Loss ---
    winning_trades = (returns > 0).sum()
    losing_trades = (returns < 0).sum()
    metrics["winning_hours"] = int(winning_trades)
    metrics["losing_hours"] = int(losing_trades)

    return metrics


def _compute_signal_decay(result: BacktestResult) -> dict:
    """Measure how much P&L decays as a function of signal age.

    For each hour after a signal triggers, we average the subsequent
    cumulative P&L.  A monotonically increasing curve means the signal
    has persistent predictive power; a flat or decreasing curve means
    decay.
    """
    strength = result.signal_result.signal_strength
    pnl = result.pnl

    if strength.empty or strength.max().max() == 0:
        return {"decay_profile": [], "decay_half_life_hours": None}

    signal_events = (strength.abs() > 0.1).any(axis=1)
    event_times = signal_events[signal_events].index

    max_lookahead = min(48, len(pnl) // 10)
    decay_profile = []
    for offset in range(max_lookahead):
        future_pnl = pnl.shift(-offset)
        aligned = future_pnl.loc[event_times]
        decay_profile.append(aligned.mean())

    decay_series = pd.Series(decay_profile).ffill()
    if len(decay_series) > 1 and decay_series.iloc[0] != 0:
        half_val = decay_series.iloc[0] / 2
        below_half = (decay_series <= half_val).values
        half_life = int(np.argmax(below_half)) if below_half.any() else None
    else:
        half_life = None

    return {
        "decay_profile": [round(v, 4) for v in decay_profile if not np.isnan(v)],
        "decay_half_life_hours": half_life,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dictionary."""
    sections = {
        "Returns & Risk": [
            "total_return_pct", "annualised_return_pct", "annualised_vol_pct",
            "sharpe_ratio", "calmar_ratio",
        ],
        "Drawdown": [
            "max_drawdown_pct", "avg_drawdown_pct",
            "avg_dd_duration_hours", "max_dd_duration_hours",
        ],
        "Hit Rate & Profit Factor": [
            "hit_rate_pct", "profit_factor",
            "winning_hours", "losing_hours",
        ],
        "PnL": [
            "total_pnl", "avg_hourly_pnl", "std_hourly_pnl",
            "best_hour_pnl", "worst_hour_pnl",
        ],
        "Trading Activity": [
            "total_trades", "avg_trade_size_mw", "avg_exposure_mw",
        ],
        "Signal Decay": ["signal_decay"],
    }

    print("=" * 60)
    print("BACKTEST PERFORMANCE METRICS")
    print("=" * 60)
    for section, keys in sections.items():
        print(f"\n{section}:")
        print("-" * 40)
        for k in keys:
            if k == "signal_decay":
                sd = metrics.get(k, {})
                profile = sd.get("decay_profile", [])
                hl = sd.get("decay_half_life_hours")
                print(f"  decay half-life : {hl} h" if hl else "  decay half-life : N/A")
                print(f"  decay profile   : [{', '.join(f'{v:.3f}' for v in profile[:10])}{', ...' if len(profile) > 10 else ''}]")
            else:
                val = metrics.get(k, "N/A")
                print(f"  {k:.<25s}: {val}")
    print("=" * 60)
