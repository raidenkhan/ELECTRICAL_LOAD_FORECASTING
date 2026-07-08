"""GPU-optimized data loading for time series forecasting."""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


FEATURE_COLS = [
    'demand_mw', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'temperature_c', 'is_holiday',
]


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load CSV and engineer features. Returns sorted, gap-filled DataFrame."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # Ensure no missing hours: reindex to full hourly grid
    full_idx = pd.date_range(df['date'].min(), df['date'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
    df = df.set_index(pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'] - 1, unit='h'))
    df = df.reindex(full_idx)
    df.index.name = 'ts'
    df = df.reset_index()
    df['date'] = df['ts'].dt.normalize()  # midnight Timestamp for consistent filtering
    df['hour'] = df['ts'].dt.hour + 1

    # Forward-fill demand (gap hours get previous hour's demand)
    df['demand_mw'] = df['demand_mw'].ffill()
    df['temperature_c'] = df['temperature_c'].fillna(28.0)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

    # Calendar features
    ts = df['ts']
    df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)

    return df


def normalize_features(df: pd.DataFrame, feature_cols: list, means: dict = None, stds: dict = None):
    """Z-score normalize training features. Returns df, means, stds."""
    result = df.copy()
    if means is None:
        means = {}
        stds = {}
        for c in feature_cols:
            v = df[c].values.astype(np.float32)
            m = float(np.nanmean(v))
            s = float(np.nanstd(v)) + 1e-8
            means[c] = m
            stds[c] = s
            result[c] = (v - m) / s
    else:
        for c in feature_cols:
            result[c] = (df[c].values.astype(np.float32) - means[c]) / stds[c]
    return result, means, stds


class SequenceDataset(Dataset):
    """Sliding-window dataset: (input_window, n_features) -> (forecast_horizon,)."""

    def __init__(self, features: np.ndarray, target: np.ndarray, input_window: int, forecast_horizon: int):
        assert len(features) == len(target), f"features {len(features)} != target {len(target)}"
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.target = np.ascontiguousarray(target, dtype=np.float32)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.n_samples = len(features) - input_window - forecast_horizon + 1
        if self.n_samples < 1:
            raise ValueError(f"Not enough data: {len(features)} rows, need > {input_window + forecast_horizon}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.input_window]  # (input_window, n_features)
        y = self.target[idx + self.input_window: idx + self.input_window + self.forecast_horizon]  # (forecast_horizon,)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_dataloader(
    df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):
    dataset = SequenceDataset(
        features=df[FEATURE_COLS].values,
        target=df['demand_mw'].values,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )
