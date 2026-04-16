from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "labDensityAverage_rf_v1")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

TARGET_MODEL_MAP = {
    "density": "labDensityAverage_rf_v1",
    "bending": "labBendingAvg_rf_v1"
}