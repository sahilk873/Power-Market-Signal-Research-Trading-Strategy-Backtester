import numpy as np
import pandas as pd

from .base import Signal, SignalResult


class CongestionRankingSignal(Signal):
    """Rank zones by expected congestion stress and long/short the extremes.

    Congestion stress is proxied by:
      - Recent spread volatility (higher vol → more congestion risk)
      - Load surprise z-score
      - Outage intensity
      - Renewable shortfall

    A composite score is computed; the top N zones are shorted (they are
    expected to cheapen as congestion eases) and the bottom N are bought
    (expected to appreciate).

    Uses vectorised operations per timestamp for efficiency.
    """

    def __init__(
        self,
        zones: list[str],
        lookback: int = 6,
        top_n: int = 3,
        position_per_zone_mw: float = 50.0,
        name: str = "congestion_ranking",
    ):
        super().__init__(name)
        self.zones = zones
        self.lookback = lookback
        self.top_n = top_n
        self.position_per_zone = position_per_zone_mw

    def generate(self, features: pd.DataFrame) -> SignalResult:
        timestamps = features.index
        spread_zcols = [c for c in features.columns if "spread_" in c and c.endswith("_zscore")]

        zone_scores = {z: pd.Series(0.0, index=timestamps) for z in self.zones}

        for zone in self.zones:
            score = zone_scores[zone]

            zone_spread_zcols = [c for c in spread_zcols if c.startswith(f"spread_{zone}_")]
            if zone_spread_zcols:
                recent = features[zone_spread_zcols].abs().mean(axis=1).fillna(0)
                score += recent * 0.3

            load_surprise = features.get("load_surprise_zscore_lag_1", pd.Series(0.0, index=timestamps)).abs().fillna(0)
            score += load_surprise * 0.25

            outage = features.get("outage_zscore_lag_1", pd.Series(0.0, index=timestamps)).abs().fillna(0)
            score += outage * 0.25

            renewable = features.get("renewable_shortfall_total_lag_1", pd.Series(0.0, index=timestamps)).abs().fillna(0) / 1000
            score += renewable * 0.2

            zone_scores[zone] = score

        scores_df = pd.DataFrame(zone_scores, index=timestamps)
        ranks = scores_df.rank(axis=1, method="first")

        n_zones = len(self.zones)
        long_mask = ranks <= self.top_n
        short_mask = ranks > (n_zones - self.top_n)

        pos_data = {}
        strength_data = {}
        for zone in self.zones:
            pos = pd.Series(0.0, index=timestamps)
            strength = pd.Series(0.0, index=timestamps)
            pos[long_mask[zone]] = self.position_per_zone
            strength[long_mask[zone]] = scores_df[zone]
            pos[short_mask[zone]] = -self.position_per_zone
            strength[short_mask[zone]] = scores_df[zone]
            col = f"zone_{zone}"
            pos_data[col] = pos
            strength_data[col] = strength

        positions = pd.DataFrame(pos_data, index=timestamps)
        strengths = pd.DataFrame(strength_data, index=timestamps)

        return SignalResult(
            timestamp=timestamps,
            positions=positions,
            signal_strength=strengths,
            metadata={"top_n": self.top_n, "per_zone_mw": self.position_per_zone},
        )
