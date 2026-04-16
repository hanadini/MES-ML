from __future__ import annotations

from typing import Callable

import pandas as pd

from models.ensemble import evaluate_ensemble
from models.registry import get_regression_model
from models.time_cv import run_time_series_cv
from models.xgb_tuner import tune_xgb_regressor
from features.selectors import select_top_correlated_features


def _select_features_for_fold(
    train_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    target_column: str,
) -> list[str]:
    selected_feature_columns = select_top_correlated_features(
        train_df,
        targets=[target_column],
        top_n=30,
    )

    if "operPressSpeed" in selected_feature_columns:
        selected_feature_columns.remove("operPressSpeed")

    selected_feature_columns = [
        col for col in selected_feature_columns
        if col in candidate_feature_columns
    ]

    if not selected_feature_columns:
        raise ValueError("No features selected after fold feature selection.")

    return selected_feature_columns


def _fit_rf_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    candidate_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    selected_features = _select_features_for_fold(
        train_df=train_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    X_train = train_df[selected_features]
    y_train = train_df[target_column]

    X_val = val_df[selected_features]
    y_val = val_df[target_column]

    model = get_regression_model(
        model_name="rf",
        model_params={
            "n_estimators": 300,
            "max_depth": 12,
            "random_state": 42,
            "n_jobs": -1,
        },
    )

    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_val), index=y_val.index)
    return y_val, y_pred


def _fit_xgb_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    candidate_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    selected_features = _select_features_for_fold(
        train_df=train_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    X_train = train_df[selected_features]
    y_train = train_df[target_column]

    X_val = val_df[selected_features]
    y_val = val_df[target_column]

    best_params, _ = tune_xgb_regressor(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_iter=15,
        random_state=42,
        early_stopping_rounds=30,
    )

    model = get_regression_model(
        model_name="xgb",
        model_params=best_params,
    )

    imputer = model.named_steps["imputer"]
    xgb_model = model.named_steps["model"]

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    xgb_model.fit(
        X_train_imp,
        y_train,
        verbose=False,
    )

    y_pred = pd.Series(xgb_model.predict(X_val_imp), index=y_val.index)
    return y_val, y_pred


def _fit_ensemble_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    candidate_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    y_true_rf, y_pred_rf = _fit_rf_fold(
        train_df=train_df,
        val_df=val_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    y_true_xgb, y_pred_xgb = _fit_xgb_fold(
        train_df=train_df,
        val_df=val_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    if not y_true_rf.index.equals(y_true_xgb.index):
        raise ValueError("RF and XGB fold validation indices do not match.")

    _, best_weight = evaluate_ensemble(
        y_true=y_true_rf,
        pred_a=y_pred_xgb,
        pred_b=y_pred_rf,
        weights=[0.55, 0.60, 0.65, 0.70, 0.75],
        overfit_penalty=0.20,
    )

    y_pred_ensemble = pd.Series(
        best_weight * y_pred_xgb.values + (1.0 - best_weight) * y_pred_rf.values,
        index=y_true_rf.index,
    )

    return y_true_rf, y_pred_ensemble


def run_model_cv_summary(
    *,
    df: pd.DataFrame,
    time_column: str,
    candidate_feature_columns: list[str],
    target_column: str,
    model_kind: str,
    n_folds: int = 3,
) -> tuple[list, dict]:
    model_kind = model_kind.lower().strip()

    if model_kind == "rf":
        fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.Series, pd.Series]] = (
            lambda train_df, val_df: _fit_rf_fold(
                train_df=train_df,
                val_df=val_df,
                candidate_feature_columns=candidate_feature_columns,
                target_column=target_column,
            )
        )
    elif model_kind == "xgb":
        fn = lambda train_df, val_df: _fit_xgb_fold(
            train_df=train_df,
            val_df=val_df,
            candidate_feature_columns=candidate_feature_columns,
            target_column=target_column,
        )
    elif model_kind == "ensemble_xgb_rf":
        fn = lambda train_df, val_df: _fit_ensemble_fold(
            train_df=train_df,
            val_df=val_df,
            candidate_feature_columns=candidate_feature_columns,
            target_column=target_column,
        )
    else:
        raise ValueError(
            f"Unsupported model_kind='{model_kind}'. "
            f"Supported: ['rf', 'xgb', 'ensemble_xgb_rf']"
        )

    fold_results, summary = run_time_series_cv(
        df=df,
        time_column=time_column,
        train_and_predict_fn=fn,
        n_folds=n_folds,
        min_train_ratio=0.50,
        val_ratio=0.15,
    )

    return fold_results, summary