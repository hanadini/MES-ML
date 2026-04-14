from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Dict, Optional

import joblib

from config.settings import ARTIFACTS_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, "LoadedModelBundle"] = {}
_REGISTRY_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


class LoadedModelBundle:
    def __init__(
        self,
        model_name: str,
        model: Any,
        feature_names: List[str],
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        target_name: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.feature_names = feature_names
        self.metrics = metrics or {}
        self.metadata = metadata or {}
        self.target_name = target_name
        self.algorithm_name = algorithm_name
        self.artifact_dir = artifact_dir


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_models_registry() -> Dict[str, Dict[str, Any]]:
    global _REGISTRY_CACHE

    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE

    registry_path = ARTIFACTS_DIR / "best_models_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Best model registry not found: {registry_path}"
        )

    registry = _read_json(registry_path)

    if not isinstance(registry, dict):
        raise ValueError(
            f"best_models_registry.json must contain a target->model mapping dictionary: {registry_path}"
        )

    _REGISTRY_CACHE = registry
    logger.info("Loaded best model registry: %s", registry_path)
    return registry


def get_registered_target_map() -> Dict[str, str]:
    registry = load_best_models_registry()
    return {
        target_name: model_info["artifact_model_name"]
        for target_name, model_info in registry.items()
    }


def resolve_model_name_for_target(target_name: str) -> str:
    registry = load_best_models_registry()

    if target_name not in registry:
        raise ValueError(
            f"Unsupported target '{target_name}'. Supported targets: {sorted(registry.keys())}"
        )

    return registry[target_name]["artifact_model_name"]


def load_model_bundle(model_name: str) -> LoadedModelBundle:
    if model_name in _MODEL_CACHE:
        logger.info("Using cached model bundle: %s", model_name)
        return _MODEL_CACHE[model_name]

    model_dir = ARTIFACTS_DIR / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifact directory not found: {model_dir}")

    model_path = model_dir / "model.joblib"
    features_path = model_dir / "features.json"
    metrics_path = model_dir / "metrics.json"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    model = joblib.load(model_path)
    feature_names = _read_json(features_path)

    if not isinstance(feature_names, list):
        raise ValueError(f"features.json must contain a list of feature names: {features_path}")

    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}

    registry = load_best_models_registry()
    matched_target_name = None
    matched_algorithm_name = None

    for target_name, info in registry.items():
        if info.get("artifact_model_name") == model_name:
            matched_target_name = target_name
            matched_algorithm_name = info.get("algorithm_name")
            break

    logger.info("Loaded model bundle from disk: %s", model_name)

    bundle = LoadedModelBundle(
        model_name=model_name,
        model=model,
        feature_names=feature_names,
        metrics=metrics,
        metadata=metadata,
        target_name=matched_target_name,
        algorithm_name=matched_algorithm_name,
        artifact_dir=str(model_dir),
    )

    _MODEL_CACHE[model_name] = bundle
    return bundle


def load_model_bundle_for_target(target_name: str) -> LoadedModelBundle:
    normalized_target = target_name.strip()
    model_name = resolve_model_name_for_target(normalized_target)
    return load_model_bundle(model_name)