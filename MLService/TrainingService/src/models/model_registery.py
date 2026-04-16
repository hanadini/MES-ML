from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def save_best_models_registry(
    all_results,
    output_path: str = "artifacts/best_models_registry.json",
):
    grouped_results = defaultdict(list)

    for result in all_results:
        grouped_results[result.target_name].append(result)

    best_models: dict[str, dict[str, Any]] = {}

    for target_name, results in grouped_results.items():
        best_result = max(
            results,
            key=lambda r: r.metrics["test"]["r2"]
        )

        registry_entry: dict[str, Any] = {
            "target_name": best_result.target_name,
            "algorithm_name": best_result.algorithm_name,
            "artifact_model_name": best_result.artifact_model_name,
            "artifact_dir": best_result.artifact_dir,
            "test_r2": best_result.metrics["test"]["r2"],
            "test_rmse": best_result.metrics["test"]["rmse"],
            "test_mae": best_result.metrics["test"]["mae"],
            "validation_r2": best_result.metrics["validation"]["r2"],
        }

        # Add serving metadata for PredictionService
        if best_result.algorithm_name == "ensemble_xgb_rf":
            ensemble_info = best_result.metrics.get("ensemble", {})

            registry_entry["type"] = "weighted_ensemble"
            registry_entry["members"] = {
                "xgb": ensemble_info["members"][0],
                "rf": ensemble_info["members"][1],
            }
            registry_entry["weights"] = {
                "xgb": ensemble_info["xgb_weight"],
                "rf": ensemble_info["rf_weight"],
            }

        else:
            registry_entry["type"] = "single_model"

        best_models[target_name] = registry_entry

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(best_models, indent=2),
        encoding="utf-8",
    )

    return best_models


def print_best_models_summary(best_models):
    print("\n" + "=" * 50)
    print("BEST MODEL PER TARGET")
    print("=" * 50)

    for target, info in best_models.items():
        print(f"\nTarget: {target}")
        print(f"  Model : {info['algorithm_name']}")
        print(f"  Type  : {info['type']}")
        print(f"  R2    : {info['test_r2']:.4f}")

        if info["type"] == "weighted_ensemble":
            print(f"  Members : {info['members']}")
            print(f"  Weights : {info['weights']}")