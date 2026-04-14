from __future__ import annotations

import numpy as np
import pandas as pd

from config.columns import MODEL_FEATURES, ALL_TARGETS, PRIMARY_TARGETs, ID_COLUMNS, TIME_COLUMN
from config.settings import CLEANED_FILE_PATH, MAX_MISSING_RATIO_PER_FEATURE


TEXT_NA_VALUES = {"", " ", "nan", "NaN", "None", "-", "--", "null", "NULL"}


def normalize_text_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(list(TEXT_NA_VALUES), np.nan)
    return df


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TIME_COLUMN in df.columns:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce", dayfirst=True)
    return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [col for col in MODEL_FEATURES + ALL_TARGETS if col in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def drop_rows_without_primary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_targets = [col for col in PRIMARY_TARGETs if col in df.columns]

    if existing_targets:
        df = df[df[existing_targets].notna().all(axis=1)].copy()

    return df


def drop_high_missing_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    removable_features = []
    for col in MODEL_FEATURES:
        if col in df.columns and df[col].isna().mean() > MAX_MISSING_RATIO_PER_FEATURE:
            removable_features.append(col)

    df = df.drop(columns=removable_features, errors="ignore")
    return df, removable_features


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TIME_COLUMN in df.columns:
        df = df.sort_values(by=TIME_COLUMN, ascending=True)
    return df


def basic_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    subset_cols = []
    for col in ID_COLUMNS + [TIME_COLUMN]:
        if col in df.columns:
            subset_cols.append(col)

    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    return df


def save_cleaned_dataframe(df: pd.DataFrame) -> None:
    CLEANED_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEANED_FILE_PATH, index=False)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = normalize_text_na(df)
    df = parse_time_column(df)
    df = coerce_numeric_columns(df)
    df = drop_rows_without_primary_target(df)
    df = basic_deduplicate(df)
    df = sort_by_time(df)
    df, dropped_features = drop_high_missing_feature_columns(df)
    save_cleaned_dataframe(df)
    return df, dropped_features