from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataLoader(ABC):
    def __init__(self, start_date: str, end_date: str):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

    @abstractmethod
    def load_prices(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def load_fundamentals(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def load_weather(self) -> pd.DataFrame:
        ...

    def get_date_range(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start_date, self.end_date, freq="h")

    def combine(
        self,
        prices: Optional[pd.DataFrame] = None,
        fundamentals: Optional[pd.DataFrame] = None,
        weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        dfs = []
        if prices is not None:
            dfs.append(prices)
        if fundamentals is not None:
            dfs.append(fundamentals)
        if weather is not None:
            dfs.append(weather)
        if not dfs:
            raise ValueError("At least one data source required")
        df = dfs[0]
        for other in dfs[1:]:
            df = df.join(other, how="outer")
        return df.sort_index()
