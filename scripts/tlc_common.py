"""Shared helpers for Week 4 NYC TLC debugging experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 6320

CATEGORICAL_COLUMNS = [
    "pickup_location_id",
    "dropoff_location_id",
    "pickup_hour",
    "pickup_day_of_week",
]
NUMERIC_COLUMNS = ["pickup_day_of_month", "trip_distance", "passenger_count"]
TARGET_LOG_COLUMN = "target_log_duration"
TARGET_SECONDS_COLUMN = "target_duration_seconds"
SPLIT_COLUMN = "split"

# Embedding vocabulary sizes = max observed ID + 1
CATEGORY_VOCAB_SIZES = {
    "pickup_location_id": 266,
    "dropoff_location_id": 266,
    "pickup_hour": 24,
    "pickup_day_of_week": 7,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return repo_root() / "data" / "week04_tlc_trip_duration_smoke_10k.parquet"


def default_output_dir() -> Path:
    return repo_root() / "outputs"


def load_prepared_frame(data_path: Path) -> pd.DataFrame:
    columns = (
        CATEGORICAL_COLUMNS
        + NUMERIC_COLUMNS
        + [TARGET_LOG_COLUMN, TARGET_SECONDS_COLUMN, SPLIT_COLUMN]
    )
    frame = pd.read_parquet(data_path, columns=columns)
    expected = {"train", "validation", "test"}
    actual = set(frame[SPLIT_COLUMN].unique())
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing split values: {sorted(missing)}")
    return frame


@dataclass
class SplitTensors:
    categorical: torch.Tensor
    numeric: torch.Tensor
    target_log: torch.Tensor
    target_seconds: torch.Tensor


@dataclass
class PreparedSplits:
    train: SplitTensors
    validation: SplitTensors
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    target_log_mean: float
    target_log_std: float


def _frame_to_arrays(
    frame: pd.DataFrame,
    numeric_mean: np.ndarray | None,
    numeric_std: np.ndarray | None,
    scale_numeric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    categorical = frame[CATEGORICAL_COLUMNS].to_numpy(dtype=np.int64)
    numeric_values = frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    if scale_numeric:
        if numeric_mean is None or numeric_std is None:
            numeric_mean = numeric_values.mean(axis=0)
            numeric_std = numeric_values.std(axis=0)
            numeric_std = np.where(numeric_std == 0, 1.0, numeric_std)
        numeric = (numeric_values - numeric_mean) / numeric_std
    else:
        numeric_mean = np.zeros(numeric_values.shape[1], dtype=np.float32)
        numeric_std = np.ones(numeric_values.shape[1], dtype=np.float32)
        numeric = numeric_values
    target_log = frame[TARGET_LOG_COLUMN].to_numpy(dtype=np.float32)
    target_seconds = frame[TARGET_SECONDS_COLUMN].to_numpy(dtype=np.float32)
    return categorical, numeric, target_log, target_seconds, numeric_mean, numeric_std


def prepare_splits(
    frame: pd.DataFrame,
    *,
    scale_numeric: bool = True,
) -> PreparedSplits:
    train_frame = frame[frame[SPLIT_COLUMN] == "train"].reset_index(drop=True)
    val_frame = frame[frame[SPLIT_COLUMN] == "validation"].reset_index(drop=True)

    train_cat, train_num, train_log, train_sec, num_mean, num_std = _frame_to_arrays(
        train_frame, None, None, scale_numeric
    )
    val_cat, val_num, val_log, val_sec, _, _ = _frame_to_arrays(
        val_frame, num_mean, num_std, scale_numeric
    )

    return PreparedSplits(
        train=SplitTensors(
            categorical=torch.tensor(train_cat, dtype=torch.long),
            numeric=torch.tensor(train_num, dtype=torch.float32),
            target_log=torch.tensor(train_log, dtype=torch.float32),
            target_seconds=torch.tensor(train_sec, dtype=torch.float32),
        ),
        validation=SplitTensors(
            categorical=torch.tensor(val_cat, dtype=torch.long),
            numeric=torch.tensor(val_num, dtype=torch.float32),
            target_log=torch.tensor(val_log, dtype=torch.float32),
            target_seconds=torch.tensor(val_sec, dtype=torch.float32),
        ),
        numeric_mean=num_mean,
        numeric_std=num_std,
        target_log_mean=float(train_log.mean()),
        target_log_std=float(train_log.std() if train_log.std() > 0 else 1.0),
    )


def scale_targets(target_log: torch.Tensor, mean: float, std: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return target_log
    return (target_log - mean) / std


def unscale_predictions(pred_scaled: torch.Tensor, mean: float, std: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return pred_scaled
    return pred_scaled * std + mean


def log_to_seconds(pred_log: torch.Tensor) -> torch.Tensor:
    return torch.expm1(pred_log).clamp_min(0.0)


def make_loader(
    tensors: SplitTensors,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        tensors.categorical,
        tensors.numeric,
        tensors.target_log,
        tensors.target_seconds,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def mae_seconds(y_true: torch.Tensor, y_pred_seconds: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_true - y_pred_seconds)).item())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
