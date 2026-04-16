from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_regression_model(model_name: str, model_params: dict | None = None) -> Any:
    params = model_params or {}
    model_name = model_name.lower().strip()

    if model_name in {"linear_regression", "lr"}:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression(**params)),
        ])

    if model_name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(**params)),
        ])

    if model_name == "lasso":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Lasso(**params)),
        ])

    if model_name in {"random_forest", "rf"}:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**params)),
        ])

    if model_name in {"xgb", "xgboost"}:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(**params)),
        ])

    if model_name in {"lgbm", "lightgbm"}:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(**params)),
        ])

    supported = [
        "linear_regression", "lr",
        "ridge",
        "lasso",
        "random_forest", "rf",
        "xgb", "xgboost",
        "lgbm", "lightgbm",
    ]
    raise ValueError(f"Unsupported model_name='{model_name}'. Supported: {supported}")