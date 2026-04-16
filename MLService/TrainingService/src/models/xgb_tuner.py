from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor


def _build_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tune_xgb_regressor(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 25,
    random_state: int = 42,
    early_stopping_rounds: int = 30,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Tune XGB on train -> validation only.
    Test set must remain untouched.
    """

    param_dist = {
        "n_estimators": [400, 600, 800, 1000, 1200],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 9],
        "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
        "reg_alpha": [0.0, 0.1, 0.3, 1.0, 3.0],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0, 15.0],
    }

    sampled_params = list(
        ParameterSampler(
            param_distributions=param_dist,
            n_iter=n_iter,
            random_state=random_state,
        )
    )

    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -np.inf

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    for i, params in enumerate(sampled_params, start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )

        model.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val_imp)
        metrics = _build_metrics(y_val, y_val_pred)

        best_iteration = getattr(model, "best_iteration", None)
        print(
            f"[XGB tuning {i}/{len(sampled_params)}] "
            f"R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f} | "
            f"best_iteration={best_iteration} | params={params}"
        )

        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_params = dict(params)
            best_metrics = metrics

            if best_iteration is not None:
                best_params["n_estimators"] = int(best_iteration) + 1

    if best_params is None or best_metrics is None:
        raise RuntimeError("XGB tuning failed to produce best parameters.")

    return best_params, best_metrics