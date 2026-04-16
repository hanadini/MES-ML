from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    target: str = Field(
        ...,
        description="Target to predict, e.g. labDensityAverage or labBendingAvg",
        example="labDensityAverage"
    )

    features: Dict[str, Any] = Field(
        ...,
        description="Incoming feature dictionary",
        example={
            "rawThickness": 18.2,
            "pressPressureMid_mean": 142.5,
            "beltSpeed1": 31.2
        }
    )


class PredictAllRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Incoming feature dictionary used for all configured targets",
        example={
            "rawThickness": 18.2,
            "pressPressureMid_mean": 142.5,
            "beltSpeed1": 31.2
        }
    )