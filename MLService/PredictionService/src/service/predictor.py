from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from service.model_loader import LoadedModelBundle


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


def predict_single(
    bundle: LoadedModelBundle,
    incoming_features: Dict[str, Any],
    target: str,
) -> Dict[str, Any]:
    X, missing_features = build_input_dataframe(
        incoming_features=incoming_features,
        expected_feature_names=bundle.feature_names,
    )

    prediction = bundle.model.predict(X)

    if hasattr(prediction, "__len__"):
        prediction_value = float(prediction[0])
    else:
        prediction_value = float(prediction)

    return {
        "target": target,
        "model_name": bundle.model_name,
        "algorithm_name": bundle.algorithm_name or bundle.metadata.get("algorithm", "unknown"),
        "prediction": prediction_value,
        "used_feature_count": len(bundle.feature_names),
        "missing_features": missing_features,
    }