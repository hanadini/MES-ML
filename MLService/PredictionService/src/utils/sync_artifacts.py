from __future__ import annotations

from utils.artifact_sync import sync_best_artifacts


def main() -> None:
    sync_best_artifacts(
        training_artifacts_dir=r"C:\Users\edvazat\PycharmProjects\MDF1-MLtraining\TrainingService\artifacts",
        prediction_artifacts_dir=r"C:\Users\edvazat\PycharmProjects\MDF1-MLtraining\PredictionService\artifacts",
    )


if __name__ == "__main__":
    main()