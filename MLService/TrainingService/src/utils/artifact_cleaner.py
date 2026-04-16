from __future__ import annotations

import shutil
from pathlib import Path


def clean_artifacts_dir(
    artifacts_root: str = "artifacts",
    keep_registry: bool = True,
) -> None:
    """
    Clean artifacts directory before training.

    Args:
        artifacts_root: path to artifacts folder
        keep_registry: keep best_models_registry.json if True
    """
    artifacts_path = Path(artifacts_root)

    if not artifacts_path.exists():
        print(f"[Cleaner] No artifacts directory found: {artifacts_path}")
        return

    print(f"[Cleaner] Cleaning artifacts directory: {artifacts_path}")

    for item in artifacts_path.iterdir():
        # Keep registry if needed
        if keep_registry and item.name == "best_models_registry.json":
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"[Cleaner] Failed to remove {item}: {e}")

    print("[Cleaner] Artifacts cleaned.")