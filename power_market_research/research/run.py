"""End-to-end research workflow: data → features → signals → backtest → attribution."""

import warnings

import pandas as pd

from power_market_research.config import AppConfig
from power_market_research.data.pjm import PJMDataGenerator
from power_market_research.features.engineering import FeatureEngineer
from power_market_research.signals.base import CombinedSignal
from power_market_research.signals.hub_spread import HubSpreadSignal
from power_market_research.signals.zone_spread import ZoneSpreadMeanReversion
from power_market_research.signals.congestion import CongestionRankingSignal
from power_market_research.backtester.engine import BacktestEngine
from power_market_research.backtester.metrics import compute_metrics, print_metrics
from power_market_research.backtester.attribution import PnLAttributor

warnings.filterwarnings("ignore")


def run_research() -> None:
    cfg = AppConfig()

    # ------------------------------------------------------------------ #
    # 1. Generate synthetic PJM market data (substitute real data here)  #
    # ------------------------------------------------------------------ #
    print("Generating synthetic PJM market data ...")
    generator = PJMDataGenerator(seed=cfg.random_seed)
    dates = pd.date_range(cfg.backtest.start_date, cfg.backtest.end_date, freq="h")
    raw = generator.generate_all(dates)
    print(f"  Generated {len(raw):,} hourly records with {len(raw.columns)} columns")

    # ------------------------------------------------------------------ #
    # 2. Feature engineering — all features properly lagged               #
    # ------------------------------------------------------------------ #
    print("\nEngineering features (all lagged to avoid look-ahead bias) ...")
    engineer = FeatureEngineer(cfg.features)
    hub_cols = cfg.pjm.hubs
    zone_cols = cfg.pjm.zones
    features = engineer.compute(raw, hub_cols=hub_cols, zone_cols=zone_cols)
    print(f"  Feature table: {features.shape[0]:,} rows x {features.shape[1]:,} cols")

    # Drop rows that are NaN from lagging (warm-up period)
    features = features.dropna(how="all").fillna(0)

    # Separate price data (non-lagged, for P&L computation)
    price_cols = list(dict.fromkeys(hub_cols + zone_cols))
    price_cols = [c for c in price_cols if c in raw.columns]
    prices = raw[price_cols].reindex(features.index)

    # ------------------------------------------------------------------ #
    # 3. Define & combine trading signals                                 #
    # ------------------------------------------------------------------ #
    print("\nGenerating trading signals ...")
    hub_signal = HubSpreadSignal(
        spread_pairs=cfg.signals.hub_spread_pairs,
        zscore_threshold=cfg.features.zscore_threshold,
        position_scale_mw=100.0,
    )
    zone_signal = ZoneSpreadMeanReversion(
        zones=cfg.pjm.zones,
        lookback=cfg.signals.zone_spread_mean_reversion_lookback,
        zscore_threshold=cfg.features.zscore_threshold,
        position_limit_mw=150.0,
    )
    congestion_signal = CongestionRankingSignal(
        zones=cfg.pjm.zones,
        lookback=cfg.signals.congestion_ranking_lookback,
        top_n=cfg.signals.top_n_zones,
        position_per_zone_mw=50.0,
    )

    combined_signal = CombinedSignal([
        (hub_signal, 0.4),
        (zone_signal, 0.35),
        (congestion_signal, 0.25),
    ])

    # ------------------------------------------------------------------ #
    # 4. Run backtest                                                    #
    # ------------------------------------------------------------------ #
    print("\nRunning backtest ...")
    engine = BacktestEngine(cfg.backtest)
    result = engine.run(combined_signal, features, prices)

    # ------------------------------------------------------------------ #
    # 5. Compute & print metrics                                          #
    # ------------------------------------------------------------------ #
    metrics = compute_metrics(result)
    print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # 6. P&L attribution by feature family                                #
    # ------------------------------------------------------------------ #
    print("\nComputing P&L attribution ...")
    attributor = PnLAttributor()
    contributions = attributor.attribute(
        pnl=result.pnl,
        features=features,
        position_strength=result.signal_result.signal_strength,
    )
    attributor.print_summary(contributions)

    # ------------------------------------------------------------------ #
    # 7. Summary statistics                                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("RESEARCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Period            : {cfg.backtest.start_date} → {cfg.backtest.end_date}")
    print(f"  Total hours       : {len(features):,}")
    print(f"  Feature count     : {features.shape[1]:,}")
    print(f"  Initial capital   : ${cfg.backtest.initial_capital:,.0f}")
    print(f"  Final equity      : ${result.equity_curve.iloc[-1]:,.0f}")
    print(f"  Total P&L         : ${metrics['total_pnl']:,.0f}")
    print(f"  Sharpe (ann.)     : {metrics['sharpe_ratio']}")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']}%")
    print(f"  Hit rate          : {metrics['hit_rate_pct']}%")
    print(f"{'=' * 60}\n")

    print("Run complete. To use real PJM data, supply CSV/Parquet paths to\n"
          "PJMRealDataLoader and re-run.")


if __name__ == "__main__":
    run_research()
