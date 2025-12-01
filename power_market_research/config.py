from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BacktestConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 10_000_000.0
    transaction_cost_per_mwh: float = 2.50
    position_limit_mw: float = 500.0
    volatility_lookback: int = 20
    volatility_target: float = 0.15
    max_portfolio_leverage: float = 2.0
    signal_decay_half_life: int = 5


@dataclass
class FeatureConfig:
    lags: List[int] = field(default_factory=lambda: [1, 24, 168])
    roll_windows: List[int] = field(default_factory=lambda: [24, 168])
    zscore_window: int = 168
    zscore_threshold: float = 2.0


@dataclass
class PJMConfig:
    hubs: List[str] = field(
        default_factory=lambda: [
            "WEST_HUB", "EAST_HUB", "PJM_RTLMP",
        ]
    )
    zones: List[str] = field(
        default_factory=lambda: [
            "AEP", "APS", "ATSI", "BGE", "COMED", "DAY",
            "DEOK", "DOM", "DPL", "DUQ", "EKPC", "JCPL",
            "METED", "PECO", "PENELEC", "PEPCO", "PPL",
            "PSEG", "RECO", "WEST", "EAST",
        ]
    )
    price_col: str = "lmp"
    load_col: str = "load"
    temp_col: str = "temperature"
    wind_col: str = "wind_mw"
    solar_col: str = "solar_mw"


@dataclass
class SignalConfig:
    hub_spread_pairs: List[tuple] = field(
        default_factory=lambda: [
            ("WEST_HUB", "EAST_HUB"),
            ("WEST_HUB", "PJM_RTLMP"),
            ("EAST_HUB", "PJM_RTLMP"),
        ]
    )
    zone_spread_mean_reversion_lookback: int = 24
    congestion_ranking_lookback: int = 6
    top_n_zones: int = 3


@dataclass
class AppConfig:
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    pjm: PJMConfig = field(default_factory=PJMConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    random_seed: int = 42
    n_jobs: int = 4
