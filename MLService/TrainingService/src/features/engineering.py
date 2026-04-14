from __future__ import annotations

import pandas as pd

from config.columns import pressure_cols, temp_cols, thickness_cols


def _existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _filter_by_suffix(cols: list[str], suffix: str) -> list[str]:
    return [c for c in cols if c.endswith(suffix)]


def _extract_zone_number(col_name: str) -> int | None:
    digits = "".join(ch for ch in col_name if ch.isdigit())
    return int(digits) if digits else None


def _filter_zone_range(cols: list[str], start_zone: int, end_zone: int) -> list[str]:
    selected = []
    for c in cols:
        zone = _extract_zone_number(c)
        if zone is not None and start_zone <= zone <= end_zone:
            selected.append(c)
    return selected


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    existing_pressure = _existing(out, pressure_cols)
    existing_temp = _existing(out, temp_cols)
    existing_thickness = _existing(out, thickness_cols)

    out = _safe_numeric(out, existing_pressure + existing_temp + existing_thickness)

    new_cols: dict[str, pd.Series] = {}

    pressure_l = _filter_by_suffix(existing_pressure, "L")
    pressure_c = _filter_by_suffix(existing_pressure, "C")
    pressure_r = _filter_by_suffix(existing_pressure, "R")

    if pressure_l:
        new_cols["pressPressureL_mean"] = out[pressure_l].mean(axis=1)
        new_cols["pressPressureL_std"] = out[pressure_l].std(axis=1)

    if pressure_c:
        new_cols["pressPressureC_mean"] = out[pressure_c].mean(axis=1)
        new_cols["pressPressureC_std"] = out[pressure_c].std(axis=1)

    if pressure_r:
        new_cols["pressPressureR_mean"] = out[pressure_r].mean(axis=1)
        new_cols["pressPressureR_std"] = out[pressure_r].std(axis=1)

    if pressure_l and pressure_r:
        new_cols["pressPressureLR_mean_diff"] = out[pressure_l].mean(axis=1) - out[pressure_r].mean(axis=1)

    if existing_pressure:
        pressure_mean = out[existing_pressure].mean(axis=1)
        pressure_std = out[existing_pressure].std(axis=1)
        pressure_min = out[existing_pressure].min(axis=1)
        pressure_max = out[existing_pressure].max(axis=1)

        new_cols["pressPressureGlobal_mean"] = pressure_mean
        new_cols["pressPressureGlobal_std"] = pressure_std
        new_cols["pressPressureGlobal_min"] = pressure_min
        new_cols["pressPressureGlobal_max"] = pressure_max
        new_cols["pressPressureGlobal_range"] = pressure_max - pressure_min

    pressure_front = _filter_zone_range(existing_pressure, 1, 7)
    pressure_mid = _filter_zone_range(existing_pressure, 8, 14)
    pressure_end = _filter_zone_range(existing_pressure, 15, 21)

    if pressure_front:
        new_cols["pressPressureFront_mean"] = out[pressure_front].mean(axis=1)
    if pressure_mid:
        new_cols["pressPressureMid_mean"] = out[pressure_mid].mean(axis=1)
    if pressure_end:
        new_cols["pressPressureEnd_mean"] = out[pressure_end].mean(axis=1)

    if pressure_front and pressure_end:
        new_cols["pressPressureFrontEnd_diff"] = out[pressure_front].mean(axis=1) - out[pressure_end].mean(axis=1)

    if existing_temp:
        temp_mean = out[existing_temp].mean(axis=1)
        temp_std = out[existing_temp].std(axis=1)
        temp_min = out[existing_temp].min(axis=1)
        temp_max = out[existing_temp].max(axis=1)

        new_cols["tempGlobal_mean"] = temp_mean
        new_cols["tempGlobal_std"] = temp_std
        new_cols["tempGlobal_min"] = temp_min
        new_cols["tempGlobal_max"] = temp_max
        new_cols["tempGlobal_range"] = temp_max - temp_min

    if existing_thickness:
        thick_mean = out[existing_thickness].mean(axis=1)
        thick_std = out[existing_thickness].std(axis=1)
        thick_min = out[existing_thickness].min(axis=1)
        thick_max = out[existing_thickness].max(axis=1)

        new_cols["thicknessClosed_mean"] = thick_mean
        new_cols["thicknessClosed_std"] = thick_std
        new_cols["thicknessClosed_min"] = thick_min
        new_cols["thicknessClosed_max"] = thick_max
        new_cols["thicknessClosed_range"] = thick_max - thick_min

    engineered_df = pd.DataFrame(new_cols, index=out.index)
    out = pd.concat([out, engineered_df], axis=1)

    return out


def get_engineered_feature_names(df: pd.DataFrame) -> list[str]:
    base_cols = set(df.columns)
    enriched = add_engineered_features(df)
    return [c for c in enriched.columns if c not in base_cols]