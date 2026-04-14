from config.columns import (
    MODEL_FEATURES,
    engineered_only_features,
    PRIMARY_TARGETs,
    TIME_COLUMN,
)
from data.cleaner import clean_dataframe
from data.ingest_raw_ksoft import ingest_raw_ksoft_file
from data.validator import validate_dataframe
from eda.audit import run_initial_audit
from features.engineering import add_engineered_features
from models.model_registery import save_best_models_registry, print_best_models_summary
from pipeline.training_pipeline import run_single_target_regression_pipeline
from utils.comparison_utils import save_comparison_summary
from utils.seed import set_seed


def main() -> None:
    # ======================================
    # STEP 0 — Setup
    # ======================================
    set_seed(42)

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
            "artifact_suffix": "xgb_v1",
            "model_params": {
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            },
            "notes": "MDF1 XGBoost model with engineered features",
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

    for target in PRIMARY_TARGETs:
        print(f"\n==============================")
        print(f"Training target: {target}")
        print(f"==============================")

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