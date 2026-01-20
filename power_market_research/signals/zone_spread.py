import numpy as np
import pandas as pd

from .base import Signal, SignalResult


class ZoneSpreadMeanReversion(Signal):
    """Mean reversion on zone-to-hub spreads (vectorised).

    For each zone, uses the pre-computed z-score of its spread vs.
    PJM_RTLMP (or the first matching spread_zscore found).  When the
    z-score exceeds the threshold, fade it back toward zero.
    Positions capped at `position_limit_mw`.
    """

    def __init__(
        self,
        zones: list[str],
        lookback: int = 24,
        zscore_threshold: float = 2.0,
        position_limit_mw: float = 200.0,
        name: str = "zone_spread_mr",
    ):
        super().__init__(name)
        self.zones = zones
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.position_limit = position_limit_mw

    def generate(self, features: pd.DataFrame) -> SignalResult:
        timestamps = features.index
        pos_data = {}
        strength_data = {}

        for zone in self.zones:
            zcol = None
            for c in features.columns:
                if c.startswith(f"spread_{zone}_") and c.endswith("_zscore"):
                    zcol = c
                    break
            if zcol is None:
                basis_col = f"basis_{zone}"
                if basis_col in features.columns:
                    z = (
                        features[basis_col]
                        .rolling(self.lookback, min_periods=1)
                        .apply(lambda s: (s.iloc[-1] - s.mean()) / s.std() if s.std() > 0 else 0.0)
                    )
                else:
                    continue
            else:
                z = features[zcol].fillna(0)

            raw = -z / self.zscore_threshold
            raw = raw.clip(-1.0, 1.0)
            col = f"zone_{zone}"
            pos_data[col] = raw * self.position_limit
            strength_data[col] = raw.abs()

        if not pos_data:
            return SignalResult(
                timestamp=timestamps,
                positions=pd.DataFrame(index=timestamps),
                signal_strength=pd.DataFrame(index=timestamps),
            )

        return SignalResult(
            timestamp=timestamps,
            positions=pd.DataFrame(pos_data, index=timestamps),
            signal_strength=pd.DataFrame(strength_data, index=timestamps),
            metadata={
                "lookback": self.lookback,
                "threshold": self.zscore_threshold,
                "limit_mw": self.position_limit,
            },
        )
