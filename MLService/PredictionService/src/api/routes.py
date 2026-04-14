from __future__ import annotations

from fastapi import APIRouter, HTTPException

from config.settings import ARTIFACTS_DIR
from schemas.request_schema import PredictionRequest, PredictAllRequest
from schemas.response_schema import (
    PredictionResponse,
    PredictAllResponse,
    PredictAllItemResponse,
)
from service.model_loader import (
    load_best_models_registry,
    get_registered_target_map,
    load_model_bundle_for_target,
)
from service.predictor import predict_single

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/models")
def list_models() -> dict:
    try:
        model_dirs = sorted(
            [
                p.name for p in ARTIFACTS_DIR.iterdir()
                if p.is_dir()
            ]
        )

        registry = load_best_models_registry()

        return {
            "available_models": model_dirs,
            "count": len(model_dirs),
            "best_model_registry": registry,
            "target_model_map": get_registered_target_map(),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        target_name = request.target.strip()
        bundle = load_model_bundle_for_target(target_name)

        result = predict_single(
            bundle=bundle,
            incoming_features=request.features,
            target=target_name,
        )

        return PredictionResponse(**result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-all", response_model=PredictAllResponse)
def predict_all(request: PredictAllRequest) -> PredictAllResponse:
    try:
        registry = load_best_models_registry()

        predictions = {}
        used_feature_count_by_target = {}
        missing_features_by_target = {}

        for target_name in registry.keys():
            bundle = load_model_bundle_for_target(target_name)

            result = predict_single(
                bundle=bundle,
                incoming_features=request.features,
                target=target_name,
            )

            predictions[target_name] = PredictAllItemResponse(
                model_name=result["model_name"],
                algorithm_name=result["algorithm_name"],
                prediction=result["prediction"],
            )

            used_feature_count_by_target[target_name] = result["used_feature_count"]
            missing_features_by_target[target_name] = result["missing_features"]

        return PredictAllResponse(
            predictions=predictions,
            used_feature_count_by_target=used_feature_count_by_target,
            missing_features_by_target=missing_features_by_target,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))