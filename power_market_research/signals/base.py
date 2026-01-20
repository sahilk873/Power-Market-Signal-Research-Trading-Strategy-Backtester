from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SignalResult:
    timestamp: pd.DatetimeIndex
    positions: pd.DataFrame  # index=timestamp, columns=instruments (node pairs)
    signal_strength: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class Signal(ABC):
    """Base class for all trading signals."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, features: pd.DataFrame) -> SignalResult:
        ...

    def scale_volatility(
        self,
        raw_positions: pd.DataFrame,
        prices: pd.DataFrame,
        lookback: int = 20,
        target_vol: float = 0.15,
    ) -> pd.DataFrame:
        returns = prices.pct_change()
        vol = returns.rolling(lookback, min_periods=1).std().fillna(0)
        scaling = target_vol / (vol * np.sqrt(8760))
        scaling = scaling.clip(upper=5.0)
        common_cols = raw_positions.columns.intersection(scaling.columns)
        return raw_positions[common_cols] * scaling[common_cols]


class CombinedSignal(Signal):
    """Weighted combination of multiple signals."""

    def __init__(self, signals: list[tuple[Signal, float]]):
        super().__init__("combined")
        self.signals = signals

    def generate(self, features: pd.DataFrame) -> SignalResult:
        results: list[tuple[SignalResult, float, str]] = []
        for signal, weight in self.signals:
            res = signal.generate(features)
            results.append((res, weight, signal.name))
        first = results[0][0]
        all_positions = []
        all_strengths = []
        for res, weight, _ in results:
            all_positions.append(res.positions * weight)
            all_strengths.append(res.signal_strength * weight)
        combined_positions = all_positions[0]
        for p in all_positions[1:]:
            combined_positions = combined_positions.add(p, fill_value=0)
        combined_strength = all_strengths[0]
        for s in all_strengths[1:]:
            combined_strength = combined_strength.add(s, fill_value=0)
        return SignalResult(
            timestamp=first.timestamp,
            positions=combined_positions,
            signal_strength=combined_strength,
            metadata={"components": [(name, w) for _, w, name in results]},
        )
