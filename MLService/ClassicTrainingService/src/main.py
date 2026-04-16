from config.columns import MODEL_FEATURES, engineered_only_features, PRIMARY_TARGETs
from data.cleaner import clean_dataframe
from data.ingest_raw_ksoft import ingest_raw_ksoft_file
from data.validator import validate_dataframe
from eda.audit import run_initial_audit
from features.engineering import add_engineered_features, get_engineered_feature_names
from features.selectors import select_top_correlated_features
from models.train import train_target_models


def main() -> None:
    print('Step 1: Ingesting raw Ksoft file...')
    raw_df = ingest_raw_ksoft_file()
    print(f'Raw data shape: {raw_df.shape}')
    # print("First 30 columns:")
    # print(raw_df.columns[:30].tolist())

    matching_time_cols = [c for c in raw_df.columns if "Production" in str(c) or "production" in str(c)]
    print("Possible production date columns:", matching_time_cols)

    print(' Step 2: Running validation...')
    validation_report = validate_dataframe(raw_df)
    print('Validation completed')

    missing_required = validation_report['schema_checks']['missing_required_columns']
    if missing_required:
        print(f"Missing required columns: {missing_required}")
        for col in missing_required:
            print(f" - {col}")
        print("Please fix schema better for training the model.")
        available_model_features = [col for col in MODEL_FEATURES if col in raw_df.columns]
        print(f"Available model features: {len(available_model_features)}")
        print("\nContinuing with available columns only...")

    print('Step 3: Cleaning dataset...')
    clean_df, dropped_features = clean_dataframe(raw_df)
    print(f'Cleaned data shape: {clean_df.shape}')

    if dropped_features:
        print('Dropped high-missing feature columns:')
        for col in dropped_features:
            print(f' - {col}')

    print("Step 4a: Building engineered features...")
    engineered_df = add_engineered_features(clean_df)
    engineered_feature_names = get_engineered_feature_names(clean_df)
    print(f"Engineered features added: {len(engineered_feature_names)}")
    for col in engineered_feature_names:
        print(f" - {col}")

    print('step 4: Running initial audit...')
    audit_text = run_initial_audit(engineered_df)
    print(audit_text)

    print("Step 5a: Selecting top correlated features...")
    top_features = select_top_correlated_features(engineered_df,targets=PRIMARY_TARGETs,top_n=30)
    print("Top 30 selected features:")
    for col in top_features:
        print(f" - {col}")

    top_features = [f for f in top_features if f != "operPressSpeed"]
    print("Removed from experiment:", "operPressSpeed")

    print("Step 5: Training baseline models engineered-only features...")
    # metrics = train_density_models(engineered_df, selected_features=top_features)
    metrics = train_target_models(engineered_df, selected_features=engineered_only_features)
    print("Model training completed.")
    print(metrics)


if __name__ == '__main__':
    main()