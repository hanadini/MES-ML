from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)


def write_excel(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_excel(path, index=False, **kwargs)


def safe_file_name(name: str) -> str:
    """
    Convert arbitrary text to a safer filename fragment.
    """
    cleaned = name.strip().replace(" ", "_")
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        cleaned = cleaned.replace(ch, "_")
    return cleaned