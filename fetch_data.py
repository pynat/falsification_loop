# fetch_data.py
# downloads usdt hourly data from binance and builds dollar bars.
# run this ONCE before any other script in the pipeline.
# outputs: dollar_bars.csv

import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from config import SYMBOL, INTERVAL, YEARS, DOLLAR_BAR_THRESHOLD


def fetch_ohlcv(symbol: str, interval: str, years: int) -> pd.DataFrame:
    url        = "https://api.binance.com/api/v1/klines"
    start_time = datetime.now() - timedelta(days=365 * years)
    end_time   = datetime.now()
    all_data   = []
    chunk_map  = {"1m": 1, "1h": 40, "1d": 90}
    chunk_days = chunk_map.get(interval, 40)

    while start_time < end_time:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime":   int((start_time + timedelta(days=chunk_days)).timestamp() * 1000),
            "limit":     1000,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            logging.error(f"request failed: {r.status_code}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        start_time = pd.to_datetime(data[-1][0], unit="ms") + timedelta(milliseconds=1)
        logging.info(f"fetched up to {start_time.date()}")
        time.sleep(0.25)

    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ]
    df = pd.DataFrame(all_data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_base", "taker_quote"]:
        df[col] = df[col].astype(float)
    df["dollar_volume"] = df["quote_volume"]
    return df[["open", "high", "low", "close", "volume", "dollar_volume",
               "trades", "taker_base", "taker_quote"]]


def build_dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    bars      = []
    cum_dv    = 0.0
    o         = None
    h         = -np.inf
    l         = np.inf
    vol       = 0.0
    trades    = 0.0
    t_base    = 0.0
    t_quote   = 0.0
    start_idx = None

    for ts, row in df.iterrows():
        if o is None:
            o         = row["open"]
            start_idx = ts
        h       = max(h, row["high"])
        l       = min(l, row["low"])
        vol    += row["volume"]
        cum_dv += row["dollar_volume"]
        trades += row["trades"]
        t_base += row["taker_base"]
        t_quote += row["taker_quote"]
        if cum_dv >= threshold:
            bars.append({
                "timestamp":    start_idx,
                "open":         o,
                "high":         h,
                "low":          l,
                "close":        row["close"],
                "volume":       vol,
                "dollar_volume": cum_dv,
                "trades":       trades,
                "taker_base":   t_base,
                "taker_quote":  t_quote,
            })
            o, h, l, vol, cum_dv      = None, -np.inf, np.inf, 0.0, 0.0
            trades, t_base, t_quote   = 0.0, 0.0, 0.0

    bar_df = pd.DataFrame(bars).set_index("timestamp")
    logging.info(f"dollar bars: {len(bar_df)} (threshold=${threshold:,.0f})")
    return bar_df


if __name__ == "__main__":
    print(f"fetching {YEARS} years of {SYMBOL} {INTERVAL} data from binance...")
    raw = fetch_ohlcv(SYMBOL, INTERVAL, YEARS)
    raw.to_csv("results/hourly.csv")
    logging.info(f"hourly rows: {len(raw)}  ({raw.index[0].date()} to {raw.index[-1].date()})")

    print(f"building dollar bars (threshold=${DOLLAR_BAR_THRESHOLD:,.0f})...")
    dollar_bars = build_dollar_bars(raw, DOLLAR_BAR_THRESHOLD)
    dollar_bars.to_csv("results/dollar_bars.csv")

    print("─" * 70)
    print(f"hourly rows:  {len(raw)}")
    print(f"dollar bars:  {len(dollar_bars)}")
    print(f"date range:   {dollar_bars.index[0].date()} to {dollar_bars.index[-1].date()}")
    print(f"bars/day avg: {len(dollar_bars) / (YEARS * 365):.1f}")
    print("─" * 70)
    print("saved -> dollar_bars.csv")
