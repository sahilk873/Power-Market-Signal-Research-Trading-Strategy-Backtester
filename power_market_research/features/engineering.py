import numpy as np
import pandas as pd

from power_market_research.config import FeatureConfig


def _zscore(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    s = np.nanstd(arr)
    return (arr[-1] - np.nanmean(arr)) / s if s > 0 else 0.0


class FeatureEngineer:
    """Transforms raw market data into a clean, lagged feature table.

    Every feature is *lagged* relative to the prediction timestamp so no
    look-ahead bias is possible.

    Feature families (used for P&L attribution):
        'price_momentum', 'load_surprise', 'temperature',
        'renewable', 'outage', 'congestion', 'reserve_margin', 'gas'
    """

    def __init__(self, config: FeatureConfig):
        self.cfg = config

    def _price_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.endswith("_HUB") or c == "PJM_RTLMP" or c.endswith("_GEN")]

    def _rolling_zscore(self, s: pd.Series) -> pd.Series:
        return s.rolling(self.cfg.zscore_window, min_periods=2).apply(
            _zscore, raw=True
        ).shift(1)

    def compute(
        self,
        raw: pd.DataFrame,
        hub_cols: list[str] | None = None,
        zone_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        f = {}
        df = raw
        base = self._price_cols(df)

        f["hour"] = df.index.hour
        f["dow"] = df.index.dayofweek
        f["month"] = df.index.month
        f["is_peak"] = (df.index.hour.isin(range(7, 23))).astype(int)
        f["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        for col in base:
            for lag in self.cfg.lags:
                f[f"{col}_lag_{lag}"] = df[col].shift(lag)

        hub_cols = hub_cols or []
        zone_cols = zone_cols or []
        pairs = self._spread_pairs(hub_cols, zone_cols, base)
        for a, b in pairs:
            if a in df.columns and b in df.columns:
                spread = df[a] - df[b]
                col = f"spread_{a}_{b}"
                f[col] = spread
                for lag in self.cfg.lags:
                    f[f"{col}_lag_{lag}"] = spread.shift(lag)

        for col in base:
            for w in self.cfg.roll_windows:
                f[f"{col}_ma_{w}"] = df[col].rolling(w, min_periods=1).mean().shift(1)
                std = df[col].rolling(w, min_periods=1).std().fillna(0).shift(1)
                f[f"{col}_std_{w}"] = std

        for col in base:
            if col not in df.columns:
                continue
            f[f"{col}_zscore"] = self._rolling_zscore(df[col])

        if "load" in df.columns:
            load_24 = df["load"].rolling(24, min_periods=1).mean().shift(1)
            surprise = df["load"] - load_24
            f["load_surprise"] = surprise
            f["load_surprise_pct"] = surprise / load_24.replace(0, np.nan)
            f["load_surprise_zscore"] = self._rolling_zscore(surprise)

        for src in ["wind_mw", "solar_mw"]:
            if src not in df.columns:
                continue
            seasonal = df[src].groupby([df.index.month, df.index.hour]).transform("mean")
            shortfall = df[src] - seasonal
            f[f"{src}_shortfall"] = shortfall
            f[f"{src}_shortfall_pct"] = shortfall / seasonal.replace(0, np.nan)

        if "wind_mw" in df.columns and "solar_mw" in df.columns:
            total_rn = df["wind_mw"] + df["solar_mw"]
            f["renewable_total_mw"] = total_rn
            seasonal = total_rn.groupby([total_rn.index.month, total_rn.index.hour]).transform("mean")
            f["renewable_shortfall_total"] = total_rn - seasonal

        if "forced_outage_mw" in df.columns:
            f["outage_intensity"] = df["forced_outage_mw"].diff().shift(1)
            f["outage_ma_24h"] = df["forced_outage_mw"].rolling(24, min_periods=1).mean().shift(1)
            normal = df["forced_outage_mw"].rolling(168, min_periods=1).mean().shift(1)
            f["outage_abnormal"] = (df["forced_outage_mw"] - normal) / normal.replace(0, np.nan)
            f["outage_zscore"] = self._rolling_zscore(df["forced_outage_mw"])

        if "gas_price" in df.columns:
            f["gas_ma_24h"] = df["gas_price"].rolling(24, min_periods=1).mean().shift(1)
            f["gas_return_1h"] = df["gas_price"].pct_change().shift(1)
            avg_power = df[[c for c in base if c in df.columns]].mean(axis=1)
            ratio = df["gas_price"] / avg_power.replace(0, np.nan)
            f["gas_power_ratio"] = ratio
            f["gas_power_ratio_zscore"] = self._rolling_zscore(ratio)

        if "load" in df.columns:
            cap = 120000.0
            margin = (cap - df["load"]) / cap
            f["reserve_margin"] = margin
            f["reserve_margin_change"] = margin.diff().shift(1)
            f["reserve_margin_zscore"] = self._rolling_zscore(margin)

        spread_cols = [c for c in f if c.startswith("spread_") and not any(c.endswith(f"_{l}") for l in self.cfg.lags)]
        for col in spread_cols:
            if col in f:
                val = f[col] if isinstance(f[col], pd.Series) else df[col]
                f[f"{col}_momentum"] = val.diff(6).shift(1)
                f[f"{col}_zscore"] = self._rolling_zscore(val)

        result = pd.concat([df] + [pd.Series(v, name=k) for k, v in f.items()], axis=1)
        dupes = result.columns.duplicated()
        if dupes.any():
            result = result.loc[:, ~dupes]
        return result

    def _spread_pairs(self, hub_cols, zone_cols, base_cols):
        pairs = []
        hubs = [h for h in hub_cols if h in base_cols or h in zone_cols]
        for i, a in enumerate(hubs):
            for b in hubs[i + 1:]:
                pairs.append((a, b))
        for h in hubs:
            if "PJM_RTLMP" in base_cols and h != "PJM_RTLMP":
                pairs.append((h, "PJM_RTLMP"))
        for h in hubs[:3]:
            for z in zone_cols[:5]:
                pairs.append((h, z))
        seen = set()
        deduped = []
        for a, b in pairs:
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                deduped.append((a, b))
        return deduped

    @staticmethod
    def label_feature_family(col: str) -> str:
        cl = col.lower()
        if any(x in cl for x in ["load", "surprise"]):
            return "load_surprise"
        if any(x in cl for x in ["temp"]):
            return "temperature"
        if any(x in cl for x in ["wind", "solar", "renewable"]):
            return "renewable"
        if any(x in cl for x in ["outage"]):
            return "outage"
        if any(x in cl for x in ["spread", "basis", "momentum", "congestion"]):
            return "congestion"
        if any(x in cl for x in ["reserve", "margin"]):
            return "reserve_margin"
        if any(x in cl for x in ["gas"]):
            return "gas"
        if any(x in cl for x in ["lag", "ma", "std", "zscore"]):
            return "price_momentum"
        if cl in ["is_peak", "is_weekend", "hour", "dow", "month"]:
            return "time"
        return "other"

    @staticmethod
    def feature_families() -> list[str]:
        return [
            "price_momentum",
            "load_surprise",
            "temperature",
            "renewable",
            "outage",
            "congestion",
            "reserve_margin",
            "gas",
        ]
