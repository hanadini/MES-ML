from __future__ import annotations

import pandas as pd

from config.columns import PRIMARY_TARGETs, MODEL_FEATURES, ALL_TARGETS, ID_COLUMNS, TIME_COLUMN


def _is_usable_feature(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if df[col].notna().sum() == 0:
        return False

    name = str(col)
    if name.startswith("lab"):
        return False
    if name.startswith("vkstockProxy"):
        return False
    if name.startswith("abs24h"):
        return False

    return True


def get_available_features(df: pd.DataFrame) -> list[str]:
    excluded = set(ALL_TARGETS) | set(ID_COLUMNS) | {TIME_COLUMN}

    base_features = [
        col for col in MODEL_FEATURES
        if col not in excluded
        and _is_usable_feature(df, col)
    ]

    engineered_features = [
        col for col in df.columns
        if col not in excluded
        and col not in base_features
        and _is_usable_feature(df, col)
    ]

    return base_features + engineered_features


def select_top_correlated_features(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    top_n: int = 30
) -> list[str]:
    if targets is None:
        targets = PRIMARY_TARGETs

    available_features = get_available_features(df)
    selected_features: list[str] = []

    for target in targets:
        if target not in df.columns:
            continue

        numeric_df = df[available_features + [target]].copy()
        numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")

        corr = numeric_df.corr(numeric_only=True)[target]
        corr = corr.drop(labels=[target], errors="ignore").dropna()

        top_features = corr.abs().sort_values(ascending=False).head(top_n).index.tolist()

        for feature in top_features:
            if feature not in selected_features:
                selected_features.append(feature)

    return selected_features