import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

FEATURE_COLUMNS = ["close", "open", "high", "low", "volume", "adjclose"]

@dataclass
class MarketData:
    frames: dict
    ticker_meta: pd.DataFrame
    symbols: list
    sector_to_id: dict
    symbol_to_sector_id: dict

def load_market_data(npy_path, ticker_csv_path):
    frames = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(frames, dict):
        raise TypeError("Expected ticker -> DataFrame dict in NPY.")

    meta = pd.read_csv(ticker_csv_path)
    for col in ("Symbol", "Sector"):
        if col not in meta.columns:
            raise ValueError(f"Ticker CSV missing {col}")

    meta["Symbol"] = meta["Symbol"].astype(str)
    meta["Sector"] = meta["Sector"].astype(str)
    symbols = sorted(set(frames) & set(meta["Symbol"]))

    sector_names = sorted(meta.loc[meta["Symbol"].isin(symbols), "Sector"].unique())
    sector_to_id = {s: i for i, s in enumerate(sector_names)}
    sector_by_symbol = meta.drop_duplicates("Symbol").set_index("Symbol")["Sector"].to_dict()
    symbol_to_sector_id = {s: sector_to_id[sector_by_symbol[s]] for s in symbols}

    clean = {}
    for symbol in symbols:
        df = frames[symbol].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{symbol} missing {missing}")
        clean[symbol] = df[FEATURE_COLUMNS].astype(float)

    return MarketData(clean, meta, symbols, sector_to_id, symbol_to_sector_id)

def fit_feature_scaler(market, symbols, start, end):
    chunks = []
    for s in symbols:
        x = market.frames[s].loc[start:end, FEATURE_COLUMNS].to_numpy(float)
        if len(x):
            chunks.append(x)
    mat = np.concatenate(chunks, axis=0)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    std[std == 0] = 1.0
    return mean, std

class StockWindowDataset(Dataset):
    def __init__(self, market, symbols, start, end, lookback, lookahead, mean, std):
        self.market = market
        self.lookback = lookback
        self.lookahead = lookahead
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.samples = []

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        for symbol in symbols:
            df = market.frames[symbol]
            vals = df[FEATURE_COLUMNS].to_numpy(np.float32)
            closes = df["close"].to_numpy(float)
            dates = df.index

            for i in range(lookback - 1, len(df) - lookahead):
                d = dates[i]
                if d < start_ts or d > end_ts:
                    continue
                window = vals[i-lookback+1:i+1]
                c0 = closes[i]
                c1 = closes[i+lookahead]
                if c0 == 0:
                    continue
                target = (c1 - c0) / c0
                if np.isfinite(window).all() and np.isfinite(target):
                    self.samples.append((symbol, i, float(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        symbol, i, target = self.samples[idx]
        df = self.market.frames[symbol]
        window = df[FEATURE_COLUMNS].iloc[i-self.lookback+1:i+1].to_numpy(np.float32)
        x = (window - self.mean) / self.std
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(target, dtype=torch.float32),
            "sector_id": torch.tensor(self.market.symbol_to_sector_id[symbol], dtype=torch.long),
            "symbol": symbol,
            "date": str(df.index[i].date()),
        }
