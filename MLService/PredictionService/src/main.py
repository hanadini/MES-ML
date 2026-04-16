from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from utils.artifact_sync import sync_best_artifacts
from utils.logging_utils import get_logger

logger = get_logger(__name__)

TRAINING_ARTIFACTS_DIR = (
    r"C:\Users\edvazat\PycharmProjects\MDF1-MLtraining\TrainingService\artifacts"
)

PREDICTION_ARTIFACTS_DIR = (
    r"C:\Users\edvazat\PycharmProjects\MDF1-MLtraining\PredictionService\artifacts"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PredictionService startup initiated")
    logger.info("Syncing artifacts from TrainingService...")

    sync_best_artifacts(
        training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
        prediction_artifacts_dir=PREDICTION_ARTIFACTS_DIR,
    )

    logger.info("Artifact sync completed successfully")
    logger.info("PredictionService started successfully")

    yield

    logger.info("PredictionService shutdown completed")


app = FastAPI(
    title="MDF1 Prediction Service",
    description="ML prediction service for MDF1 production data",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "MDF1 Prediction Service",
        "status": "running"
    }


app.include_router(router)