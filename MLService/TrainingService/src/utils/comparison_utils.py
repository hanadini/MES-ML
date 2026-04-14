from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def build_comparison_rows(results: list) -> list[dict]:
    rows = []

    for result in results:
        row = {
            "target_name": result.target_name,
            "algorithm_name": result.algorithm_name,
            "artifact_model_name": result.artifact_model_name,
            "artifact_dir": result.artifact_dir,

            "validation_mae": result.metrics["validation"]["mae"],
            "validation_rmse": result.metrics["validation"]["rmse"],
            "validation_r2": result.metrics["validation"]["r2"],

            "test_mae": result.metrics["test"]["mae"],
            "test_rmse": result.metrics["test"]["rmse"],
            "test_r2": result.metrics["test"]["r2"],

            "train_rows": result.metrics["row_counts"]["train"],
            "val_rows": result.metrics["row_counts"]["val"],
            "test_rows": result.metrics["row_counts"]["test"],

            "split_type": result.metrics.get("split_type", ""),
        }
        rows.append(row)

    return rows


def save_comparison_summary(results: list, output_dir: str = "reports/comparisons") -> None:
    rows = build_comparison_rows(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "model_comparison.csv", index=False, encoding="utf-8-sig")
    df.to_excel(output_path / "model_comparison.xlsx", index=False)

    with open(output_path / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)