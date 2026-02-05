import numpy as np
import pandas as pd

from power_market_research.features.engineering import FeatureEngineer


class PnLAttributor:
    """Attribute P&L to feature families for explainability.

    Uses a linear decomposition: for each period, the P&L is split
    proportionally to the absolute feature exposure of each family.
    """

    FAMILIES = FeatureEngineer.feature_families()

    def __init__(self):
        self.family_columns: dict[str, list[str]] = {}

    def attribute(
        self,
        pnl: pd.Series,
        features: pd.DataFrame,
        position_strength: pd.DataFrame,
    ) -> pd.DataFrame:
        family_cols = self._map_families(features)

        position_strength = position_strength.reindex(features.index, fill_value=0)

        exposures = {}
        for family in self.FAMILIES:
            cols = [c for c in family_cols.get(family, []) if c in features.columns]
            if not cols:
                exposures[family] = pd.Series(0.0, index=features.index)
                continue
            vals = features[cols].fillna(0)
            weights = np.ones(len(cols))
            for i, c in enumerate(cols):
                base = c.split("_lag_")[0] if "_lag_" in c else c
                for sc in position_strength.columns:
                    if base in sc or sc in base:
                        w = position_strength[sc].abs()
                        weights[i] = w.iloc[0] if not w.empty else 1.0
                        break
            exposures[family] = vals.abs().dot(weights) / len(cols)

        exposure_df = pd.DataFrame(exposures, index=features.index)
        total_exposure = exposure_df.sum(axis=1).replace(0, np.nan)
        contribution = exposure_df.div(total_exposure, axis=0).mul(pnl, axis=0)

        contribution["unexplained"] = pnl - contribution[self.FAMILIES].sum(axis=1)
        return contribution

    def _map_families(self, features: pd.DataFrame) -> dict[str, list[str]]:
        if self.family_columns:
            return self.family_columns
        for col in features.columns:
            family = FeatureEngineer.label_feature_family(col)
            self.family_columns.setdefault(family, []).append(col)
        return self.family_columns

    def summary(self, contributions: pd.DataFrame) -> pd.DataFrame:
        total_pnl = contributions.sum()
        total = total_pnl.sum()
        pct = total_pnl / total * 100 if total != 0 else total_pnl * 0
        s = pd.DataFrame({"total_pnl": total_pnl, "pct_contribution": pct})
        return s.sort_values("total_pnl", ascending=False)

    def print_summary(self, contributions: pd.DataFrame) -> None:
        s = self.summary(contributions)
        print(f"\n{'=' * 60}")
        print("P&L ATTRIBUTION BY FEATURE FAMILY")
        print(f"{'=' * 60}")
        print(f"{'Family':<25s} {'Total P&L':>12s} {'% Contribution':>15s}")
        print(f"{'-' * 54}")
        for family, row in s.iterrows():
            print(f"{family:<25s} {row['total_pnl']:>12,.2f} {row['pct_contribution']:>14.2f}%")
        print(f"{'-' * 54}")
        print(f"{'Total':<25s} {s['total_pnl'].sum():>12,.2f} {s['pct_contribution'].sum():>14.2f}%")
        print(f"{'=' * 60}")
