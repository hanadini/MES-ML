from __future__ import annotations

from copy import deepcopy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.columns import (
    engineered_only_features,
    PRIMARY_TARGETs,
    TIME_COLUMN,
)
from data.cleaner import clean_dataframe
from data.ingest_raw_ksoft import ingest_raw_ksoft_file
from data.validator import validate_dataframe
from eda.audit import run_initial_audit
from features.engineering import add_engineered_features
from models.ensemble import evaluate_ensemble
from models.cv_runner import run_model_cv_summary
from models.model_registery import save_best_models_registry, print_best_models_summary
from pipeline.training_pipeline import run_single_target_regression_pipeline
from utils.artifact_cleaner import clean_artifacts_dir
from utils.comparison_utils import save_comparison_summary
from utils.seed import set_seed


def _print_cv_summary(target: str, model_kind: str, fold_results: list, summary: dict) -> None:
    print(f"\n[CV] Target={target} | Model={model_kind}")
    for fold in fold_results:
        print(
            f"  Fold {fold.fold_index}: "
            f"train=({fold.train_start} -> {fold.train_end}) | "
            f"val=({fold.val_start} -> {fold.val_end}) | "
            f"R2={fold.r2:.4f} | RMSE={fold.rmse:.4f} | MAE={fold.mae:.4f}"
        )

    print(
        f"  Mean R2={summary['mean_r2']:.4f} | "
        f"Std R2={summary['std_r2']:.4f} | "
        f"Mean RMSE={summary['mean_rmse']:.4f} | "
        f"Mean MAE={summary['mean_mae']:.4f} | "
        f"Folds={summary['fold_count']}"
    )


def main() -> None:
    # ======================================
    # STEP 0 — Setup
    # ======================================
    set_seed(42)

    confirm = input("Clean artifacts directory? (y/n): ")
    if confirm.lower() == "y":
        clean_artifacts_dir(
            artifacts_root="artifacts",
            keep_registry=False,
        )

    print("Step 1: Ingesting raw Ksoft file...")
    raw_df = ingest_raw_ksoft_file()
    print(f"Raw data shape: {raw_df.shape}")

    # ======================================
    # STEP 2 — Validation
    # ======================================
    print("Step 2: Running validation...")
    validation_report = validate_dataframe(raw_df)

    missing_required = validation_report["schema_checks"]["missing_required_columns"]
    if missing_required:
        print(f"Missing required columns: {missing_required}")
        print("Continuing with available columns only...")

    # ======================================
    # STEP 3 — Cleaning
    # ======================================
    print("Step 3: Cleaning dataset...")
    clean_df, dropped_features = clean_dataframe(raw_df)
    print(f"Cleaned data shape: {clean_df.shape}")

    # ======================================
    # STEP 4 — Feature Engineering
    # ======================================
    print("Step 4: Engineering features...")
    engineered_df = add_engineered_features(clean_df)

    # ======================================
    # STEP 5 — Audit
    # ======================================
    print("Step 5: Running audit...")
    audit_text = run_initial_audit(engineered_df)
    print(audit_text)

    # ======================================
    # STEP 6 — Feature Candidate Set
    # ======================================
    print("Step 6: Preparing candidate feature set...")
    candidate_features = engineered_only_features.copy()
    print(f"Candidate features available: {len(candidate_features)}")

    # ======================================
    # STEP 6.1 — Time-aware CV summaries
    # ======================================
    print("Step 6.1: Running time-aware CV summaries...")

    for target in PRIMARY_TARGETs:
        rf_folds, rf_summary = run_model_cv_summary(
            df=engineered_df,
            time_column=TIME_COLUMN,
            candidate_feature_columns=candidate_features,
            target_column=target,
            model_kind="rf",
            n_folds=3,
        )
        _print_cv_summary(target, "rf", rf_folds, rf_summary)

        xgb_folds, xgb_summary = run_model_cv_summary(
            df=engineered_df,
            time_column=TIME_COLUMN,
            candidate_feature_columns=candidate_features,
            target_column=target,
            model_kind="xgb",
            n_folds=3,
        )
        _print_cv_summary(target, "xgb", xgb_folds, xgb_summary)

        ensemble_folds, ensemble_summary = run_model_cv_summary(
            df=engineered_df,
            time_column=TIME_COLUMN,
            candidate_feature_columns=candidate_features,
            target_column=target,
            model_kind="ensemble_xgb_rf",
            n_folds=3,
        )
        _print_cv_summary(target, "ensemble_xgb_rf", ensemble_folds, ensemble_summary)

    # ======================================
    # STEP 7 — Training per target and algorithm
    # ======================================
    print("Step 7: Training models...")

    all_results = []

    algorithm_configs = [
        {
            "algorithm_name": "rf",
            "artifact_suffix": "rf_v2",
            "model_params": {
                "n_estimators": 300,
                "max_depth": 12,
                "random_state": 42,
                "n_jobs": -1,
            },
            "notes": "MDF1 baseline Random Forest model with engineered features",
        },
        {
            "algorithm_name": "xgb",
            "artifact_suffix": "xgb_tuned_v2",
            "model_params": {},
            "notes": "MDF1 tuned XGBoost model with early stopping and engineered features",
        },
        {
            "algorithm_name": "lgbm",
            "artifact_suffix": "lgbm_v1",
            "model_params": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
            "notes": "MDF1 LightGBM model with engineered features",
        },
    ]

    results_by_target: dict[str, dict[str, object]] = {}

    for target in PRIMARY_TARGETs:
        print(f"\n==============================")
        print(f"Training target: {target}")
        print(f"==============================")

        results_by_target[target] = {}

        for algo_cfg in algorithm_configs:
            algorithm_name = algo_cfg["algorithm_name"]
            artifact_model_name = f"{target}_{algo_cfg['artifact_suffix']}"

            print(f"\n--- Training for target: {target} | algorithm: {algorithm_name} ---")

            result = run_single_target_regression_pipeline(
                df=engineered_df,
                candidate_feature_columns=candidate_features,
                target_column=target,
                algorithm_name=algorithm_name,
                artifact_model_name=artifact_model_name,
                model_params=algo_cfg["model_params"],
                artifacts_root="artifacts",
                time_column=TIME_COLUMN,
                use_time_based_split=True,
                notes=algo_cfg["notes"],
            )

            print(f"{algorithm_name.upper()} Metrics:")
            print(result.metrics)

            all_results.append(result)
            results_by_target[target][algorithm_name] = result

        # ======================================
        # STEP 7.1 — RF + XGB Ensemble
        # ======================================
        if "rf" in results_by_target[target] and "xgb" in results_by_target[target]:
            print(f"\n--- Building Ensemble (RF + XGB) for {target} ---")

            rf_result = results_by_target[target]["rf"]
            xgb_result = results_by_target[target]["xgb"]

            val_metrics, best_weight = evaluate_ensemble(
                y_true=rf_result.y_val_true,
                pred_a=xgb_result.y_val_pred,
                pred_b=rf_result.y_val_pred,
                weights=[0.55, 0.60, 0.65, 0.70, 0.75],
                overfit_penalty=0.20,
            )

            test_pred = (
                best_weight * xgb_result.y_test_pred
                + (1 - best_weight) * rf_result.y_test_pred
            )

            test_metrics = {
                "mae": float(mean_absolute_error(rf_result.y_test_true, test_pred)),
                "rmse": float(mean_squared_error(rf_result.y_test_true, test_pred) ** 0.5),
                "r2": float(r2_score(rf_result.y_test_true, test_pred)),
            }

            print(f"Ensemble weight for XGB: {best_weight:.2f}")
            print(f"ENSEMBLE Validation Metrics: {val_metrics}")
            print(f"ENSEMBLE Test Metrics: {test_metrics}")

            best_single_test_r2 = max(
                rf_result.metrics["test"]["r2"],
                xgb_result.metrics["test"]["r2"],
            )

            min_gain = 0.001

            if test_metrics["r2"] > best_single_test_r2 + min_gain:
                print("Ensemble accepted: beats best single model on test.")

                ensemble_result = deepcopy(xgb_result)
                ensemble_result.algorithm_name = "ensemble_xgb_rf"
                ensemble_result.artifact_model_name = f"{target}_ensemble_xgb_rf_v1"
                ensemble_result.metrics = {
                    "validation": val_metrics,
                    "test": test_metrics,
                    "row_counts": xgb_result.metrics["row_counts"],
                    "split_type": xgb_result.metrics["split_type"],
                    "selected_feature_count": xgb_result.metrics["selected_feature_count"],
                    "selected_features": xgb_result.metrics["selected_features"],
                    "ensemble": {
                        "xgb_weight": best_weight,
                        "rf_weight": 1 - best_weight,
                        "members": [
                            xgb_result.artifact_model_name,
                            rf_result.artifact_model_name,
                        ],
                    },
                }

                all_results.append(ensemble_result)
                results_by_target[target]["ensemble_xgb_rf"] = ensemble_result
            else:
                print("Ensemble rejected: does not improve test performance enough.")

    # ======================================
    # STEP 8 — Save comparison summary
    # ======================================
    save_comparison_summary(all_results)
    print("Comparison summary saved under reports/comparison/")

    # ======================================
    # STEP 9 — Select and save best model per target
    # ======================================
    best_models = save_best_models_registry(
        all_results=all_results,
        output_path="artifacts/best_models_registry.json",
    )

    print_best_models_summary(best_models)

    print("\nTraining completed.")
    print("Best model registry saved under artifacts/best_models_registry.json")


if __name__ == "__main__":
    main()