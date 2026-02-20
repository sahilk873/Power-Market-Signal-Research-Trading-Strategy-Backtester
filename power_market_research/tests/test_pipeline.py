import pandas as pd
import numpy as np

from power_market_research.config import AppConfig
from power_market_research.data.pjm import PJMDataGenerator
from power_market_research.features.engineering import FeatureEngineer
from power_market_research.signals.hub_spread import HubSpreadSignal
from power_market_research.signals.zone_spread import ZoneSpreadMeanReversion
from power_market_research.signals.congestion import CongestionRankingSignal
from power_market_research.signals.base import CombinedSignal
from power_market_research.backtester.engine import BacktestEngine
from power_market_research.backtester.metrics import compute_metrics
from power_market_research.backtester.attribution import PnLAttributor


def test_data_generation():
    gen = PJMDataGenerator(seed=42)
    dates = pd.date_range("2023-01-01", "2023-01-08", freq="h")[:7 * 24]
    df = gen.generate_all(dates)
    assert len(df) == 7 * 24
    assert "WEST_HUB" in df.columns
    assert "EAST_HUB" in df.columns
    assert "PJM_RTLMP" in df.columns
    assert "load" in df.columns
    assert "temperature" in df.columns
    assert df["load"].min() > 0
    assert df["WEST_HUB"].min() >= -5.0


def test_feature_lagging():
    cfg = AppConfig()
    gen = PJMDataGenerator(seed=42)
    dates = pd.date_range("2023-01-01", "2023-01-07", freq="h")
    raw = gen.generate_all(dates)
    eng = FeatureEngineer(cfg.features)
    features = eng.compute(raw, hub_cols=cfg.pjm.hubs, zone_cols=cfg.pjm.zones)
    lag_cols = [c for c in features.columns if "_lag_" in c]
    assert len(lag_cols) > 0, "No lagged feature columns found"
    assert "load_surprise" in features.columns
    assert "load_surprise_zscore" in features.columns


def test_signals_produce_positions():
    cfg = AppConfig()
    gen = PJMDataGenerator(seed=42)
    dates = pd.date_range("2023-01-01", "2023-01-07", freq="h")
    raw = gen.generate_all(dates)
    eng = FeatureEngineer(cfg.features)
    features = eng.compute(raw, hub_cols=cfg.pjm.hubs, zone_cols=cfg.pjm.zones)
    features = features.dropna(how="all").fillna(0)

    hub = HubSpreadSignal(cfg.signals.hub_spread_pairs, 2.0, 100.0)
    res = hub.generate(features)
    assert res.positions.abs().sum().sum() > 0, "HubSpread generated zero positions"

    zone = ZoneSpreadMeanReversion(cfg.pjm.zones, 24, 2.0, 150.0)
    res = zone.generate(features)
    assert res.positions.abs().sum().sum() > 0, "ZoneSpread generated zero positions"

    cong = CongestionRankingSignal(cfg.pjm.zones, 6, 3, 50.0)
    res = cong.generate(features)
    assert res.positions.abs().sum().sum() > 0, "CongestionRanking generated zero positions"


def test_backtest_runs():
    cfg = AppConfig()
    cfg.backtest.start_date = "2023-01-01"
    cfg.backtest.end_date = "2023-01-07"
    gen = PJMDataGenerator(seed=42)
    dates = pd.date_range(cfg.backtest.start_date, cfg.backtest.end_date, freq="h")
    raw = gen.generate_all(dates)
    eng = FeatureEngineer(cfg.features)
    features = eng.compute(raw, hub_cols=cfg.pjm.hubs, zone_cols=cfg.pjm.zones)
    features = features.dropna(how="all").fillna(0)
    price_cols = [c for c in raw.columns if c in cfg.pjm.hubs or c in cfg.pjm.zones]
    prices = raw[price_cols].reindex(features.index)

    hub = HubSpreadSignal(cfg.signals.hub_spread_pairs, 2.0, 100.0)
    zone = ZoneSpreadMeanReversion(cfg.pjm.zones, 24, 2.0, 150.0)
    cong = CongestionRankingSignal(cfg.pjm.zones, 6, 3, 50.0)
    combined = CombinedSignal([(hub, 0.4), (zone, 0.35), (cong, 0.25)])

    engine = BacktestEngine(cfg.backtest)
    result = engine.run(combined, features, prices)
    assert result.pnl.sum() != 0
    assert len(result.equity_curve) > 0


def test_metrics_computed():
    cfg = AppConfig()
    cfg.backtest.start_date = "2023-01-01"
    cfg.backtest.end_date = "2023-01-14"
    gen = PJMDataGenerator(seed=42)
    dates = pd.date_range(cfg.backtest.start_date, cfg.backtest.end_date, freq="h")
    raw = gen.generate_all(dates)
    eng = FeatureEngineer(cfg.features)
    features = eng.compute(raw, hub_cols=cfg.pjm.hubs, zone_cols=cfg.pjm.zones)
    features = features.dropna(how="all").fillna(0)
    price_cols = [c for c in raw.columns if c in cfg.pjm.hubs or c in cfg.pjm.zones]
    prices = raw[price_cols].reindex(features.index)

    hub = HubSpreadSignal(cfg.signals.hub_spread_pairs, 2.0, 100.0)
    zone = ZoneSpreadMeanReversion(cfg.pjm.zones, 24, 2.0, 150.0)
    cong = CongestionRankingSignal(cfg.pjm.zones, 6, 3, 50.0)
    combined = CombinedSignal([(hub, 0.4), (zone, 0.35), (cong, 0.25)])

    engine = BacktestEngine(cfg.backtest)
    result = engine.run(combined, features, prices)
    metrics = compute_metrics(result)
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "hit_rate_pct" in metrics
    assert "total_pnl" in metrics
    assert metrics["total_trades"] > 0


def test_attribution():
    cfg = AppConfig()
    cfg.backtest.start_date = "2023-01-01"
    cfg.backtest.end_date = "2023-01-07"
    gen = PJMDataGenerator(seed=42)
    dates = pd.date_range(cfg.backtest.start_date, cfg.backtest.end_date, freq="h")
    raw = gen.generate_all(dates)
    eng = FeatureEngineer(cfg.features)
    features = eng.compute(raw, hub_cols=cfg.pjm.hubs, zone_cols=cfg.pjm.zones)
    features = features.dropna(how="all").fillna(0)
    price_cols = [c for c in raw.columns if c in cfg.pjm.hubs or c in cfg.pjm.zones]
    prices = raw[price_cols].reindex(features.index)

    hub = HubSpreadSignal(cfg.signals.hub_spread_pairs, 2.0, 100.0)
    zone = ZoneSpreadMeanReversion(cfg.pjm.zones, 24, 2.0, 150.0)
    cong = CongestionRankingSignal(cfg.pjm.zones, 6, 3, 50.0)
    combined = CombinedSignal([(hub, 0.4), (zone, 0.35), (cong, 0.25)])

    engine = BacktestEngine(cfg.backtest)
    result = engine.run(combined, features, prices)
    attr = PnLAttributor()
    contrib = attr.attribute(result.pnl, features, result.signal_result.signal_strength)
    families = PnLAttributor.FAMILIES
    for f in families:
        assert f in contrib.columns, f"Missing family {f}"
    assert "unexplained" in contrib.columns
    total = contrib.sum().sum()
    assert abs(total - result.pnl.sum()) < 1.0, "Attribution total != PnL"
