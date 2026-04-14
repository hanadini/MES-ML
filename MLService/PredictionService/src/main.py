from __future__ import annotations

from fastapi import FastAPI

from api.routes import router
from utils.logging_utils import get_logger

logger = get_logger(__name__)


app = FastAPI(
    title="MDF1 Prediction Service",
    description="ML prediction service for MDF1 production data",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    logger.info("PredictionService started successfully")


@app.get("/")
def root():
    return {
        "service": "MDF1 Prediction Service",
        "status": "running"
    }


app.include_router(router)