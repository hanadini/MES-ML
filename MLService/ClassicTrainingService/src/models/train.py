from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from features.selectors import get_available_features
from config.columns import PRIMARY_TARGETs, TIME_COLUMN


def prepare_modeling_dataset(
    df: pd.DataFrame,
    selected_features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    available_features = get_available_features(df)

    if selected_features is not None:
        available_features = [col for col in selected_features if col in available_features]

    if not available_features:
        raise ValueError("No usable features were selected. Check feature selection and leakage filters.")

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


def time_based_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def evaluate_regression_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> dict[str, dict[str, float]]:
    predictions = model.predict(X_test)

    results: dict[str, dict[str, float]] = {}
    for i, target in enumerate(y_test.columns):
        y_true = y_test.iloc[:, i]
        y_pred = predictions[:, i]

        results[target] = {
            "MAE": float(round(mean_absolute_error(y_true, y_pred), 4)),
            "RMSE": float(round(mean_squared_error(y_true, y_pred, squared=False), 4)),
            "R2": float(round(r2_score(y_true, y_pred), 4)),
        }

    return results


def build_models() -> dict[str, Pipeline]:
    models = {
        "linear_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(LinearRegression())),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            )),
        ]),
    }
    return models


def save_metrics(
    metrics: dict[str, dict[str, dict[str, float]]],
    metrics_file_name: str = "model_metrics.json"
) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / metrics_file_name

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_best_model(
    best_model_name: str,
    best_model,
    feature_names: list[str],
    model_file_name: str = "model.joblib",
    feature_file_name: str = "model_features.json"
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / model_file_name)

    feature_path = MODELS_DIR / feature_file_name
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": best_model_name,
                "targets": PRIMARY_TARGETs,
                "features": feature_names,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def extract_feature_importance(
    model_pipeline,
    feature_names: list[str]
) -> pd.DataFrame:
    rf_model = model_pipeline.named_steps["model"]

    all_importances = np.array([est.feature_importances_ for est in rf_model.estimators_])
    importances = all_importances.mean(axis=0)

    used_feature_names = feature_names[:len(importances)]

    importance_df = pd.DataFrame({
        "feature": used_feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return importance_df


def save_feature_importance(
    importance_df: pd.DataFrame,
    csv_file_name: str = "feature_importance.csv",
    excel_file_name: str = "feature_importance.xlsx"
) -> None:
    FEATURE_LISTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = FEATURE_LISTS_DIR / csv_file_name
    xlsx_path = FEATURE_LISTS_DIR / excel_file_name

    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    importance_df.to_excel(xlsx_path, index=False)


def train_target_models(
    df: pd.DataFrame,
    selected_features: list[str] | None = None
) -> dict[str, dict[str, dict[str, float]]]:
    X, y, available_features = prepare_modeling_dataset(df, selected_features=selected_features)
    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)

    # Remove features that are entirely missing in training data
    train_non_empty_cols = [col for col in X_train.columns if X_train[col].notna().sum() > 0]
    X_train = X_train[train_non_empty_cols].copy()
    X_test = X_test[train_non_empty_cols].copy()
    available_features = train_non_empty_cols

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Available features used: {len(available_features)}")

    models = build_models()
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    best_model_name = None
    best_model = None
    best_r2 = -np.inf

    target_tag = "_".join(PRIMARY_TARGETs)
    baseline_tag = f"{target_tag}_baseline_v1"

    metrics_file_name = f"{baseline_tag}_metrics.json"
    model_file_name = f"{baseline_tag}.joblib"
    feature_file_name = f"{baseline_tag}_features.json"
    importance_csv_name = f"{baseline_tag}_feature_importance.csv"
    importance_excel_name = f"{baseline_tag}_feature_importance.xlsx"

    for model_name, model in models.items():
        print(f"Training model: {model_name}")
        model.fit(X_train, y_train)

        metrics = evaluate_regression_model(model, X_test, y_test)
        all_metrics[model_name] = metrics

        print(f"{model_name} metrics:")
        for target_name, target_metrics in metrics.items():
            print(f"  {target_name}: {target_metrics}")

        if model_name == "random_forest":
            importance_df = extract_feature_importance(model, available_features)

            print("\nTop 20 important features (mean across targets):")
            for _, row in importance_df.head(20).iterrows():
                print(f"  {row['feature']}: {round(row['importance'], 4)}")

            save_feature_importance(
                importance_df,
                csv_file_name=importance_csv_name,
                excel_file_name=importance_excel_name
            )

        avg_r2 = np.mean([target_metrics["R2"] for target_metrics in metrics.values()])

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model_name = model_name
            best_model = model

    save_metrics(
        all_metrics,
        metrics_file_name=metrics_file_name
    )

    if best_model is not None and best_model_name is not None:
        save_best_model(
            best_model_name,
            best_model,
            available_features,
            model_file_name=model_file_name,
            feature_file_name=feature_file_name
        )
        print(f"Best model saved: {best_model_name}")

    return all_metrics