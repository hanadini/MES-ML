from __future__ import annotations

from typing import List, Dict
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    target: str = Field(
        ...,
        description="Requested business target (e.g. labDensityAverage or labBendingAvg)"
    )

    model_name: str = Field(
        ...,
        description="Resolved model artifact name used for prediction"
    )

    algorithm_name: str = Field(
        ...,
        description="Resolved algorithm or serving strategy used for prediction (e.g. xgb, rf, ensemble_xgb_rf)"
    )

    prediction: float = Field(
        ...,
        description="Predicted target value"
    )

    used_feature_count: int = Field(
        ...,
        description="Number of features used by the model"
    )

    missing_features: List[str] = Field(
        ...,
        description="List of features not provided in input and automatically filled with null"
    )


class PredictAllItemResponse(BaseModel):
    model_name: str = Field(
        ...,
        description="Resolved model artifact name used for prediction"
    )

    algorithm_name: str = Field(
        ...,
        description="Resolved algorithm or serving strategy used for prediction"
    )

    prediction: float = Field(
        ...,
        description="Predicted target value"
    )


class PredictAllResponse(BaseModel):
    predictions: Dict[str, PredictAllItemResponse] = Field(
        ...,
        description="Predictions for all configured targets"
    )

    used_feature_count_by_target: Dict[str, int] = Field(
        ...,
        description="Number of features used by each target model"
    )

    missing_features_by_target: Dict[str, List[str]] = Field(
        ...,
        description="Missing input features per target"
    )