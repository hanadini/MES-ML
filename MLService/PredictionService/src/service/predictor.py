from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from service.model_loader import LoadedModelBundle, load_model_bundle


def build_input_dataframe(
    incoming_features: Dict[str, Any],
    expected_feature_names: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    row: Dict[str, Any] = {}
    missing_features: List[str] = []

    for feature_name in expected_feature_names:
        if feature_name in incoming_features:
            row[feature_name] = incoming_features[feature_name]
        else:
            row[feature_name] = None
            missing_features.append(feature_name)

    df = pd.DataFrame([row], columns=expected_feature_names)
    return df, missing_features


def _predict_from_bundle(
    bundle: LoadedModelBundle,
    incoming_features: Dict[str, Any],
) -> Tuple[float, List[str]]:
    X, missing_features = build_input_dataframe(
        incoming_features=incoming_features,
        expected_feature_names=bundle.feature_names,
    )

    prediction = bundle.model.predict(X)

    if hasattr(prediction, "__len__"):
        prediction_value = float(prediction[0])
    else:
        prediction_value = float(prediction)

    return prediction_value, missing_features


def predict_single_model(
    bundle: LoadedModelBundle,
    incoming_features: Dict[str, Any],
    target: str,
) -> Dict[str, Any]:
    prediction_value, missing_features = _predict_from_bundle(bundle, incoming_features)

    return {
        "target": target,
        "model_name": bundle.model_name,
        "algorithm_name": bundle.algorithm_name or bundle.metadata.get("algorithm", "unknown"),
        "prediction": prediction_value,
        "used_feature_count": len(bundle.feature_names),
        "missing_features": missing_features,
    }


def predict_weighted_ensemble(
    *,
    target: str,
    incoming_features: Dict[str, Any],
    members: Dict[str, str],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    required_member_keys = {"xgb", "rf"}

    if not required_member_keys.issubset(set(members.keys())):
        raise ValueError(
            f"Weighted ensemble for target '{target}' must contain members {sorted(required_member_keys)}"
        )

    if not required_member_keys.issubset(set(weights.keys())):
        raise ValueError(
            f"Weighted ensemble for target '{target}' must contain weights {sorted(required_member_keys)}"
        )

    xgb_bundle = load_model_bundle(members["xgb"])
    rf_bundle = load_model_bundle(members["rf"])

    xgb_pred, xgb_missing = _predict_from_bundle(xgb_bundle, incoming_features)
    rf_pred, rf_missing = _predict_from_bundle(rf_bundle, incoming_features)

    prediction_value = (
        weights["xgb"] * xgb_pred +
        weights["rf"] * rf_pred
    )

    # Feature sets are expected to be aligned, but use union to be safe
    missing_features = sorted(set(xgb_missing) | set(rf_missing))
    used_feature_count = max(len(xgb_bundle.feature_names), len(rf_bundle.feature_names))

    return {
        "target": target,
        "model_name": f"{target}_ensemble_xgb_rf_v1",
        "algorithm_name": "ensemble_xgb_rf",
        "prediction": float(prediction_value),
        "used_feature_count": used_feature_count,
        "missing_features": missing_features,
        "ensemble_members": {
            "xgb": xgb_bundle.model_name,
            "rf": rf_bundle.model_name,
        },
        "ensemble_weights": weights,
    }