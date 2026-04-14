from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.splitter import split_train_val_test, split_time_based
from models.registry import get_regression_model
from models.artifact_saver import save_full_artifact_bundle
from models.train import extract_feature_importance
from utils.logging_utils import get_logger
from features.selectors import select_top_correlated_features

logger = get_logger(__name__)


@dataclass
class TrainingPipelineResult:
    algorithm_name: str
    artifact_model_name: str
    target_name: str
    metrics: dict
    artifact_dir: str


def _build_regression_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_single_target_regression_pipeline(
    *,
    df: pd.DataFrame,
    candidate_feature_columns: list[str],
    target_column: str,
    algorithm_name: str,
    artifact_model_name: str,
    model_params: dict | None,
    artifacts_root: str,
    preprocessor: Any | None = None,
    notes: str | None = None,
    random_state: int = 42,
    use_time_based_split: bool = True,
    time_column: str | None = None,
) -> TrainingPipelineResult:
    logger.info(
        "Starting regression pipeline | target='%s' | algorithm='%s' | artifacts='%s'",
        target_column,
        algorithm_name,
        artifact_model_name,
    )

    required_columns = set(candidate_feature_columns + [target_column])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {sorted(missing)}")

    selected_columns = candidate_feature_columns + [target_column]
    if time_column and time_column in df.columns:
        selected_columns.append(time_column)

    work_df = df[selected_columns].dropna(subset=[target_column]).copy()

    if use_time_based_split:
        if not time_column:
            raise ValueError("time_column must be provided when use_time_based_split=True")
        split_result = split_time_based(
            df=work_df,
            time_column=time_column,
            test_size=0.15,
            val_size=0.15,
        )
    else:
        split_result = split_train_val_test(
            df=work_df,
            test_size=0.15,
            val_size=0.15,
            random_state=random_state,
            shuffle=True,
        )

    train_df = split_result.train_df.copy()
    val_df = split_result.val_df.copy()
    test_df = split_result.test_df.copy()

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
        raise ValueError("No features selected after train-only feature selection.")

    X_train = train_df[selected_feature_columns]
    y_train = train_df[target_column]

    X_val = val_df[selected_feature_columns]
    y_val = val_df[target_column]

    X_test = test_df[selected_feature_columns]
    y_test = test_df[target_column]

    model = get_regression_model(model_name=algorithm_name, model_params=model_params)

    algo = algorithm_name.lower()

    if algo in {"xgb", "xgboost"}:
        imputer = model.named_steps["imputer"]
        xgb_model = model.named_steps["model"]

        imputer.fit(X_train)
        # Fit imputer on training data
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        xgb_model.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
            verbose=False,
        )
        # Re-wrap trained model back into pipeline
        model.named_steps["model"] = xgb_model

    elif algo in {"lgbm", "lightgbm"}:
        imputer = model.named_steps["imputer"]
        lgbm_model = model.named_steps["model"]

        imputer.fit(X_train)
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        lgbm_model.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
            eval_metric="l2",
        )

        model.named_steps["model"] = lgbm_model
    else:
        model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model.named_steps["feature_names_"] = selected_feature_columns


    metrics = {
        "validation": _build_regression_metrics(y_val, y_val_pred),
        "test": _build_regression_metrics(y_test, y_test_pred),
        "row_counts": {
            "train": len(split_result.train_df),
            "val": len(split_result.val_df),
            "test": len(split_result.test_df),
        },
        "split_type": "time_based" if use_time_based_split else "random",
        "selected_feature_count": len(selected_feature_columns),
        "selected_features": selected_feature_columns,
    }

    feature_importance_df = None
    if algorithm_name.lower() in {
        "rf", "random_forest",
        "xgb", "xgboost",
        "lgbm", "lightgbm",
    }:
        try:
            feature_importance_df = extract_feature_importance(model, selected_feature_columns)
        except Exception as exc:
            logger.warning("Feature importance extraction skipped: %s", exc)


    artifact_dir = save_full_artifact_bundle(
        artifacts_root=artifacts_root,
        model_name=artifact_model_name,
        model=model,
        feature_names=selected_feature_columns,
        metrics=metrics,
        target_name=target_column,
        algorithm=algorithm_name,
        preprocessor=preprocessor,
        feature_importance_df=feature_importance_df,
        notes=notes,
        extra_metadata={
            "pipeline_type": "single_target_regression",
            "split_type": "time_based" if use_time_based_split else "random",
        },
    )

    logger.info(
        "Training completed | target='%s' | algorithm='%s' | artifacts='%s'",
        target_column,
        algorithm_name,
        artifact_dir,
    )

    return TrainingPipelineResult(
        algorithm_name=algorithm_name,
        artifact_model_name=artifact_model_name,
        target_name=target_column,
        metrics=metrics,
        artifact_dir=str(artifact_dir),
    )