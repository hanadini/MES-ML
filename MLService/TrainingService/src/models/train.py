from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features.selectors import get_available_features
from config.columns import PRIMARY_TARGETs, TIME_COLUMN


def prepare_modeling_dataset(
    df: pd.DataFrame,
    selected_features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare X, y for modeling.
    """

    available_features = get_available_features(df)

    if selected_features is not None:
        available_features = [col for col in selected_features if col in available_features]

    if not available_features:
        raise ValueError("No usable features were selected.")

    modeling_df = df.copy()
    modeling_df = modeling_df.sort_values(by=TIME_COLUMN).reset_index(drop=True)

    X = modeling_df[available_features].copy()
    y = modeling_df[PRIMARY_TARGETs].copy()

    for col in PRIMARY_TARGETs:
        y[col] = pd.to_numeric(y[col], errors="coerce")

    valid_idx = y.notna().all(axis=1)
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X, y, available_features


def evaluate_regression_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """
    Evaluate multi-target regression model.
    """

    predictions = model.predict(X_test)

    results: dict[str, dict[str, float]] = {}

    for i, target in enumerate(y_test.columns):
        y_true = y_test.iloc[:, i]
        y_pred = predictions[:, i]

        results[target] = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "R2": float(r2_score(y_true, y_pred)),
        }

    return results


def extract_feature_importance(
    model_object,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Supports:
    - Pipeline(imputer + model)
    - raw model

    Works for:
    - RandomForestRegressor
    - XGBRegressor
    - LGBMRegressor
    """

    if hasattr(model_object, "named_steps"):
        final_model = model_object.named_steps["model"]
    else:
        final_model = model_object

    if not hasattr(final_model, "feature_importances_"):
        raise ValueError(
            f"Model '{type(final_model).__name__}' does not expose feature_importances_."
        )

    importances = final_model.feature_importances_

    if len(importances) != len(feature_names):
        min_len = min(len(importances), len(feature_names))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)