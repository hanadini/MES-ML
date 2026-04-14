from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    shuffle: bool = True,
) -> SplitResult:
    """
    Split a dataframe into train/validation/test.

    val_size and test_size are proportions of the full dataset.
    Example:
        test_size=0.15, val_size=0.15
        -> 70% train, 15% val, 15% test
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if test_size <= 0 or val_size <= 0:
        raise ValueError("test_size and val_size must be > 0.")

    if (test_size + val_size) >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return SplitResult(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def split_time_based(
    df: pd.DataFrame,
    time_column: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SplitResult:
    """
    Time-aware split:
    - oldest rows -> train
    - middle rows -> validation
    - newest rows -> test

    Useful later for more production-realistic MDF1 evaluation.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if time_column not in df.columns:
        raise KeyError(f"Time column '{time_column}' not found in dataframe.")

    if (test_size + val_size) >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")

    ordered_df = df.sort_values(by=time_column).reset_index(drop=True)
    n_rows = len(ordered_df)

    test_count = int(n_rows * test_size)
    val_count = int(n_rows * val_size)
    train_count = n_rows - val_count - test_count

    if train_count <= 0:
        raise ValueError("Not enough rows for time-based split.")

    train_df = ordered_df.iloc[:train_count].copy()
    val_df = ordered_df.iloc[train_count:train_count + val_count].copy()
    test_df = ordered_df.iloc[train_count + val_count:].copy()

    return SplitResult(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )