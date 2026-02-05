from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from power_market_research.config import BacktestConfig
from power_market_research.signals.base import Signal, SignalResult


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    exposures: pd.DataFrame
    returns: pd.Series
    pnl: pd.Series
    signal_result: SignalResult
    config: BacktestConfig


class BacktestEngine:
    """Systematic backtester for power market strategies.

    Handles:
      - Position limit enforcement
      - Transaction costs
      - Volatility scaling
      - Portfolio leverage capping
      - Trade recording
    """

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self.capital = config.initial_capital

    def run(
        self,
        signal: Signal,
        features: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> BacktestResult:
        signal_result = signal.generate(features)
        positions = signal_result.positions.copy()
        timestamps = positions.index

        if "PJM_RTLMP" in prices.columns:
            ref_prices = prices["PJM_RTLMP"]
        else:
            ref_prices = prices.iloc[:, 0]

        positions = self._enforce_limits(positions)
        positions = self._volatility_scale(
            positions, prices, signal_result.signal_strength
        )
        positions = self._cap_leverage(positions, ref_prices)

        pnl = self._compute_pnl(positions, prices)
        tx_costs = self._compute_tx_costs(positions, prices)
        pnl_net = pnl - tx_costs

        equity = self._build_equity_curve(pnl_net)
        trades = self._record_trades(positions, pnl_net)

        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            exposures=positions.abs(),
            returns=pnl_net / self.capital,
            pnl=pnl_net,
            signal_result=signal_result,
            config=self.cfg,
        )

    def _enforce_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        return positions.clip(-self.cfg.position_limit_mw, self.cfg.position_limit_mw)

    def _volatility_scale(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        strength: pd.DataFrame,
    ) -> pd.DataFrame:
        returns = prices.pct_change()
        vol = returns.rolling(
            self.cfg.volatility_lookback, min_periods=1
        ).std().fillna(0)
        target_vol = self.cfg.volatility_target
        annual_factor = np.sqrt(8760)
        scaling = target_vol / (vol * annual_factor)
        scaling = scaling.clip(upper=5.0)

        scaled = positions.copy()
        for col in scaled.columns:
            inst = col.replace("zone_", "").replace("_", "_")
            for price_col in prices.columns:
                if price_col in col or col.endswith(price_col):
                    if price_col in scaling.columns:
                        scaled[col] = positions[col] * scaling[price_col]
                    break
            else:
                avg_scale = scaling.mean(axis=1)
                scaled[col] = positions[col] * avg_scale

        scaled = scaled * strength.where(strength > 0, 1.0)
        return scaled

    def _cap_leverage(
        self, positions: pd.DataFrame, ref_prices: pd.Series
    ) -> pd.DataFrame:
        notional = positions.abs().mul(ref_prices, axis=0)
        total_notional = notional.sum(axis=1)
        leverage = total_notional / self.capital
        cap_factor = (self.cfg.max_portfolio_leverage / leverage).clip(upper=1.0)
        return positions.mul(cap_factor, axis=0)

    def _compute_pnl(
        self, positions: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.Series:
        if prices.columns.duplicated().any():
            prices = prices.loc[:, ~prices.columns.duplicated()]
        price_changes = prices.diff().shift(-1)
        pnl = pd.Series(0.0, index=positions.index)
        for col in positions.columns:
            pos = positions[col]
            if col.startswith("zone_"):
                zone = col.replace("zone_", "")
                if zone in price_changes.columns:
                    pnl += pos.values * price_changes[zone].values.ravel()
                else:
                    pnl += pos.values * price_changes.mean(axis=1).values.ravel()
            else:
                parts = col.split("_")
                leg1 = "_".join(parts[:2])
                leg2 = "_".join(parts[2:])
                if leg1 in price_changes.columns and leg2 in price_changes.columns:
                    s1 = price_changes[leg1]
                    s2 = price_changes[leg2]
                    if isinstance(s1, pd.DataFrame):
                        s1 = s1.iloc[:, 0]
                    if isinstance(s2, pd.DataFrame):
                        s2 = s2.iloc[:, 0]
                    pnl += pos.values * (s1.values - s2.values)
                elif leg1 in price_changes.columns:
                    s1 = price_changes[leg1]
                    if isinstance(s1, pd.DataFrame):
                        s1 = s1.iloc[:, 0]
                    pnl += pos.values * s1.values
                else:
                    pnl += pos.values * price_changes.mean(axis=1).values
        return pnl

    def _compute_tx_costs(
        self, positions: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.Series:
        turnover = positions.diff().abs()
        ref_prices = (
            prices["PJM_RTLMP"]
            if "PJM_RTLMP" in prices.columns
            else prices.iloc[:, 0]
        )
        cost_per_mwh = self.cfg.transaction_cost_per_mwh
        tx = turnover.mul(cost_per_mwh, axis=0).sum(axis=1)
        tx = tx * (ref_prices / ref_prices.mean())
        return tx.fillna(0.0)

    def _build_equity_curve(self, pnl_net: pd.Series) -> pd.Series:
        pnl_net = pnl_net.fillna(0)
        returns = pnl_net / self.capital
        equity = (1 + returns).cumprod() * self.capital
        return equity

    def _record_trades(
        self, positions: pd.DataFrame, pnl: pd.Series
    ) -> pd.DataFrame:
        position_changes = positions.diff().fillna(0)
        trade_mask = position_changes != 0
        trade_records = []
        for col in positions.columns:
            trade_times = trade_mask.index[trade_mask[col]]
            for t in trade_times:
                trade_records.append(
                    {
                        "timestamp": t,
                        "instrument": col,
                        "size_change": float(position_changes.loc[t, col]),
                        "new_position": float(positions.loc[t, col]),
                    }
                )
        if not trade_records:
            return pd.DataFrame(columns=["timestamp", "instrument", "size_change", "new_position"])
        trades = pd.DataFrame(trade_records)
        trades["pnl"] = trades["timestamp"].map(pnl)
        return trades.sort_index().reset_index(drop=True)
