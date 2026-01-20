import numpy as np
import pandas as pd

from .base import Signal, SignalResult


class HubSpreadSignal(Signal):
    """Long/short hub spreads based on z-score mean reversion.

    For each hub spread (e.g. WEST_HUB - EAST_HUB):
      - z-score > +threshold  -> short the spread (expect narrowing)
      - z-score < -threshold  -> long the spread  (expect widening)
    Positions fade linearly as z-score approaches zero.
    """

    def __init__(
        self,
        spread_pairs: list[tuple[str, str]],
        zscore_threshold: float = 2.0,
        position_scale_mw: float = 100.0,
        name: str = "hub_spread",
    ):
        super().__init__(name)
        self.spread_pairs = spread_pairs
        self.zscore_threshold = zscore_threshold
        self.position_scale = position_scale_mw

    def generate(self, features: pd.DataFrame) -> SignalResult:
        pos_list = []
        strength_list = []
        timestamps = features.index

        for leg1, leg2 in self.spread_pairs:
            zcol = f"spread_{leg1}_{leg2}_zscore"
            if zcol not in features.columns:
                continue
            zscores = features[zcol].fillna(0)
            raw = np.clip(-zscores / self.zscore_threshold, -1.0, 1.0)
            col_name = f"{leg1}_{leg2}"
            pos_series = pd.Series(
                raw.values * self.position_scale,
                index=timestamps,
                name=col_name,
            )
            pos_list.append(pos_series)
            strength_list.append(
                pd.Series(np.abs(raw.values), index=timestamps, name=col_name)
            )

        if not pos_list:
            return SignalResult(
                timestamp=timestamps,
                positions=pd.DataFrame(index=timestamps),
                signal_strength=pd.DataFrame(index=timestamps),
            )

        positions = pd.concat(pos_list, axis=1)
        strengths = pd.concat(strength_list, axis=1)
        return SignalResult(
            timestamp=timestamps,
            positions=positions,
            signal_strength=strengths,
            metadata={"threshold": self.zscore_threshold, "scale_mw": self.position_scale},
        )
