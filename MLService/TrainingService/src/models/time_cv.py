from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class TimeFoldResult:
    fold_index: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    mae: float
    rmse: float
    r2: float


def _build_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_expanding_time_folds(
    df: pd.DataFrame,
    *,
    time_column: str,
    n_folds: int = 3,
    min_train_ratio: float = 0.50,
    val_ratio: float = 0.15,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if time_column not in df.columns:
        raise KeyError(f"Time column '{time_column}' not found.")

    ordered_df = df.sort_values(by=time_column).reset_index(drop=True)
    n_rows = len(ordered_df)

    min_train_size = int(n_rows * min_train_ratio)
    val_size = int(n_rows * val_ratio)

    if min_train_size <= 0 or val_size <= 0:
        raise ValueError("Fold sizes are too small.")

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    remaining = n_rows - min_train_size - val_size
    if remaining < 0:
        raise ValueError("Not enough rows for time CV.")

    step = max(1, remaining // max(1, n_folds - 1))

    for fold_idx in range(n_folds):
        train_end = min_train_size + fold_idx * step
        val_start = train_end
        val_end = val_start + val_size

        if val_end > n_rows:
            break

        train_df = ordered_df.iloc[:train_end].copy()
        val_df = ordered_df.iloc[val_start:val_end].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        folds.append((train_df.reset_index(drop=True), val_df.reset_index(drop=True)))

    if not folds:
        raise ValueError("No valid time folds could be created.")

    return folds


def run_time_series_cv(
    *,
    df: pd.DataFrame,
    time_column: str,
    train_and_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.Series, pd.Series]],
    n_folds: int = 3,
    min_train_ratio: float = 0.50,
    val_ratio: float = 0.15,
) -> tuple[list[TimeFoldResult], dict[str, float]]:
    folds = build_expanding_time_folds(
        df=df,
        time_column=time_column,
        n_folds=n_folds,
        min_train_ratio=min_train_ratio,
        val_ratio=val_ratio,
    )

    fold_results: list[TimeFoldResult] = []

    for i, (train_df, val_df) in enumerate(folds, start=1):
        y_true, y_pred = train_and_predict_fn(train_df, val_df)
        metrics = _build_metrics(y_true, y_pred)

        fold_results.append(
            TimeFoldResult(
                fold_index=i,
                train_start=str(train_df[time_column].min()),
                train_end=str(train_df[time_column].max()),
                val_start=str(val_df[time_column].min()),
                val_end=str(val_df[time_column].max()),
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
            )
        )

    summary = {
        "mean_mae": float(np.mean([f.mae for f in fold_results])),
        "mean_rmse": float(np.mean([f.rmse for f in fold_results])),
        "mean_r2": float(np.mean([f.r2 for f in fold_results])),
        "std_r2": float(np.std([f.r2 for f in fold_results])),
        "fold_count": len(fold_results),
    }

    return fold_results, summary