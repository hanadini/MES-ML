from __future__ import annotations

import pandas as pd

from config.settings import (
    INTERIM_DIR,
    METADATA_DIR,
    RAW_FILE_PATH,
    SHEET_NAME,
    COLUMN_MAPPING_FILE_PATH,
    FLATTENED_FILE_PATH,
)


def ensure_directories() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_sheet() -> pd.DataFrame:
    """
    Load the raw Excel sheet exactly as-is, without assigning any header.
    This is the safest approach for Ksoft exports with multi-row headers.
    """
    return pd.read_excel(
        RAW_FILE_PATH,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
    )


def extract_human_readable_names(raw_df: pd.DataFrame) -> list[str]:
    """
    Excel row 1 -> human-readable labels
    pandas index 0
    """
    return [str(col).strip() for col in raw_df.iloc[0].tolist()]


def extract_technical_names(raw_df: pd.DataFrame) -> list[str]:
    """
    Excel row 2 -> technical FQTTS names
    pandas index 1
    """
    technical_names = [str(col).strip() for col in raw_df.iloc[1].tolist()]

    # Replace empty / nan-like names with fallback names
    cleaned_names = []
    for idx, name in enumerate(technical_names):
        if name.lower() in {"nan", "none", ""}:
            cleaned_names.append(f"unnamed_col_{idx}")
        else:
            cleaned_names.append(name)

    return cleaned_names


def make_unique(names: list[str]) -> list[str]:
    """
    Ensure duplicate column names become unique.
    Example:
        labBendingAvg -> labBendingAvg
        labBendingAvg -> labBendingAvg__1
    """
    seen: dict[str, int] = {}
    unique_names: list[str] = []

    for name in names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}__{seen[name]}")

    return unique_names


def build_model_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the actual modeling dataframe:
    - columns come from Excel row 2 (technical names)
    - data starts from Excel row 3
    """
    technical_names = extract_technical_names(raw_df)
    technical_names = make_unique(technical_names)

    df = raw_df.iloc[2:].copy()
    df.columns = technical_names
    df.reset_index(drop=True, inplace=True)

    return df


def build_column_mapping(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mapping between row 1 human-readable names and row 2 technical names.
    """
    human_names = extract_human_readable_names(raw_df)
    technical_names = extract_technical_names(raw_df)
    technical_names = make_unique(technical_names)

    max_len = max(len(human_names), len(technical_names))

    # pad in unlikely mismatch case
    if len(human_names) < max_len:
        human_names += [""] * (max_len - len(human_names))
    if len(technical_names) < max_len:
        technical_names += [""] * (max_len - len(technical_names))

    mapping_df = pd.DataFrame(
        {
            "human_readable_name": human_names,
            "technical_name": technical_names,
        }
    )

    return mapping_df


def save_column_mapping(mapping_df: pd.DataFrame) -> None:
    mapping_df.to_csv(COLUMN_MAPPING_FILE_PATH, index=False, encoding="utf-8-sig")


def save_flattened_dataframe(df: pd.DataFrame) -> None:
    """
    Save a robust parquet + CSV backup.
    Mixed object columns are converted to strings for safe parquet writing.
    """
    df_to_save = df.copy()

    object_cols = df_to_save.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df_to_save[col] = df_to_save[col].apply(
            lambda x: str(x).strip() if pd.notna(x) else x
        )

    df_to_save.to_parquet(FLATTENED_FILE_PATH, index=False)

    csv_backup_path = FLATTENED_FILE_PATH.with_suffix(".csv")
    df_to_save.to_csv(csv_backup_path, index=False, encoding="utf-8-sig")


def ingest_raw_ksoft_file() -> pd.DataFrame:
    ensure_directories()

    raw_df = load_raw_sheet()

    model_df = build_model_dataframe(raw_df)
    column_mapping_df = build_column_mapping(raw_df)

    save_column_mapping(column_mapping_df)
    save_flattened_dataframe(model_df)

    return model_df


if __name__ == "__main__":
    df = ingest_raw_ksoft_file()
    print("Ingestion completed.")
    print("Shape:", df.shape)
    print("First 20 columns:")
    print(df.columns[:20].tolist())