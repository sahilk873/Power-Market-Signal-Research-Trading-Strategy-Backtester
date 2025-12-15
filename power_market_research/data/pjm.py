import numpy as np
import pandas as pd

from .loader import DataLoader


class PJMDataGenerator:
    """Generates realistic synthetic PJM market data for research & development.

    Price patterns mimic real PJM hourly LMP behaviour:
      - Daily seasonality (higher during day, lower at night)
      - Weekly seasonality (higher weekday peaks)
      - Annual seasonality (higher summer/winter)
      - Temperature sensitivity
      - Random congestion events
      - Zone-specific basis spreads
    """

    SEASON_PEAK_MONTHS = [1, 2, 6, 7, 8, 12]
    PEAK_HOURS = list(range(7, 23))

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._hub_base_prices: dict[str, float] = {
            "WEST_HUB": 32.0,
            "EAST_HUB": 38.0,
            "PJM_RTLMP": 35.0,
        }
        self._zone_bias: dict[str, float] = {
            "AEP": -2.5, "APS": 1.0, "ATSI": 0.5, "BGE": 2.0,
            "COMED": 1.5, "DAY": -1.0, "DEOK": -3.0, "DOM": 3.0,
            "DPL": 1.0, "DUQ": 0.0, "EKPC": -2.0, "JCPL": 2.5,
            "METED": 1.5, "PECO": 2.0, "PENELEC": -1.5, "PEPCO": 3.5,
            "PPL": 1.0, "PSEG": 2.0, "RECO": -0.5, "WEST": -4.0,
            "EAST": 3.0,
        }
        self._gen_names: list[str] = [
            f"GEN_{i:03d}" for i in range(1, 51)
        ]

    def _price_seasonality(self, dt: pd.Timestamp) -> float:
        hour = dt.hour
        dow = dt.dayofweek
        doy = dt.dayofyear
        peak_hour = 1.0 if hour in self.PEAK_HOURS else 0.4
        weekday = 1.0 if dow < 5 else 0.7
        summer_ramp = max(0.0, 1.0 - abs(doy - 205) / 90)
        winter_ramp = max(0.0, 1.0 - abs(doy - 15) / 60)
        annual = 0.3 + 0.7 * max(summer_ramp, winter_ramp)
        return peak_hour * weekday * annual

    def generate_hub_prices(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        records: list[dict] = []
        n = len(dates)
        for hub, base in self._hub_base_prices.items():
            noise = self.rng.normal(0, 5, n)
            congestion = self.rng.exponential(3, n) * self.rng.binomial(1, 0.03, n)
            hub_prices = np.zeros(n)
            for i, dt in enumerate(dates):
                season = self._price_seasonality(dt)
                temp_shock = 2.0 * np.sin(2 * np.pi * dt.dayofyear / 365)
                hub_prices[i] = (base + noise[i]) * season + congestion[i] * 8.0
            hub_prices = np.maximum(hub_prices, -5.0)
            records.append(
                pd.DataFrame({hub: hub_prices}, index=dates)
            )
        hub_prices_df = pd.concat(records, axis=1)
        hub_prices_df["PJM_RTLMP"] = hub_prices_df[
            [h for h in self._hub_base_prices]
        ].mean(axis=1)
        return hub_prices_df

    def generate_zone_prices(
        self, dates: pd.DatetimeIndex, hub_df: pd.DataFrame
    ) -> pd.DataFrame:
        n = len(dates)
        records: list[dict] = []
        rtlmp = hub_df["PJM_RTLMP"].values
        for zone, bias in self._zone_bias.items():
            zone_noise = self.rng.normal(0, 3, n)
            congestion = self.rng.exponential(5, n) * self.rng.binomial(1, 0.04, n)
            zone_prices = rtlmp + bias + zone_noise + congestion * 6.0
            zone_prices = np.maximum(zone_prices, -10.0)
            records.append(
                pd.DataFrame({zone: zone_prices}, index=dates)
            )
        return pd.concat(records, axis=1)

    def generate_load(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates)
        base_load = 80000.0
        load = np.zeros(n)
        for i, dt in enumerate(dates):
            season = self._price_seasonality(dt)
            doy_effect = 15000 * np.sin(2 * np.pi * (dt.dayofyear - 200) / 365)
            noise = self.rng.normal(0, 3000)
            load[i] = base_load + doy_effect + 20000 * (season - 0.5) + noise
        load = np.maximum(load, 30000)
        return pd.DataFrame({"load": load}, index=dates)

    def generate_temperature(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates)
        temp = np.zeros(n)
        for i, dt in enumerate(dates):
            doy = dt.dayofyear
            base = 55 + 25 * np.sin(2 * np.pi * (doy - 100) / 365)
            noise = self.rng.normal(0, 5)
            temp[i] = base + noise
        clim = pd.Series(temp, index=dates).rolling(8760, min_periods=1).mean()
        anomaly = pd.Series(temp, index=dates) - clim
        return pd.DataFrame(
            {"temperature": temp, "temp_anomaly": anomaly.values},
            index=dates,
        )

    def generate_renewables(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates)
        wind = np.zeros(n)
        solar = np.zeros(n)
        for i, dt in enumerate(dates):
            wind_speed = max(0, self.rng.gamma(4, 2))
            wind[i] = wind_speed * 500 + self.rng.normal(0, 200)
            hour = dt.hour
            doy = dt.dayofyear
            sun_angle = max(
                0,
                np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0,
            )
            seasonal = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365)
            solar[i] = sun_angle * seasonal * 3000 + self.rng.normal(0, 100)
        wind = np.maximum(wind, 0)
        solar = np.maximum(solar, 0)
        return pd.DataFrame(
            {"wind_mw": wind, "solar_mw": solar},
            index=dates,
        )

    def generate_forced_outages(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates)
        base_outage = 3000.0
        outage = np.zeros(n)
        for i in range(n):
            spike = self.rng.exponential(1000) if self.rng.random() < 0.05 else 0
            outage[i] = base_outage + self.rng.normal(0, 200) + spike
        outage = np.maximum(outage, 500)
        return pd.DataFrame({"forced_outage_mw": outage}, index=dates)

    def generate_gas_prices(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates)
        base_gas = 3.00
        gas = np.zeros(n)
        for i, dt in enumerate(dates):
            doy = dt.dayofyear
            seasonal = 1 + 0.4 * np.sin(2 * np.pi * (doy - 15) / 365)
            noise = self.rng.normal(0, 0.20)
            gas[i] = base_gas * seasonal + noise
        gas = np.maximum(gas, 0.50)
        return pd.DataFrame({"gas_price": gas}, index=dates)

    def generate_all(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        hubs = self.generate_hub_prices(dates)
        zones = self.generate_zone_prices(dates, hubs)
        prices = pd.concat([hubs, zones], axis=1)
        load = self.generate_load(dates)
        weather = self.generate_temperature(dates)
        renewables = self.generate_renewables(dates)
        outages = self.generate_forced_outages(dates)
        gas = self.generate_gas_prices(dates)
        return prices.join([load, weather, renewables, outages, gas])


class PJMRealDataLoader(DataLoader):
    """Loads real PJM data from CSV/Parquet files.

    Expected columns for prices: timestamp, node_id, lmp, congestion, loss, marginal_loss
    Expected columns for fundamentals: timestamp, load, wind_mw, solar_mw, forced_outage_mw
    Weather: timestamp, temperature, temp_anomaly (or station-based and aggregated)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        price_path: str = "",
        fundamental_path: str = "",
        weather_path: str = "",
    ):
        super().__init__(start_date, end_date)
        self.price_path = price_path
        self.fundamental_path = fundamental_path
        self.weather_path = weather_path

    def load_prices(self) -> pd.DataFrame:
        if not self.price_path:
            return pd.DataFrame()
        df = pd.read_parquet(self.price_path) if self.price_path.endswith(".parquet") else pd.read_csv(self.price_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df = df.loc[self.start_date : self.end_date]
        if "node_id" in df.columns:
            df = df.pivot(columns="node_id", values="lmp")
        return df

    def load_fundamentals(self) -> pd.DataFrame:
        if not self.fundamental_path:
            return pd.DataFrame()
        df = pd.read_parquet(self.fundamental_path) if self.fundamental_path.endswith(".parquet") else pd.read_csv(self.fundamental_path, parse_dates=["timestamp"])
        return df.set_index("timestamp").loc[self.start_date : self.end_date]

    def load_weather(self) -> pd.DataFrame:
        if not self.weather_path:
            return pd.DataFrame()
        df = pd.read_parquet(self.weather_path) if self.weather_path.endswith(".parquet") else pd.read_csv(self.weather_path, parse_dates=["timestamp"])
        return df.set_index("timestamp").loc[self.start_date : self.end_date]
