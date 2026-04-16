from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _build_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_ensemble(
    *,
    y_true: pd.Series,
    pred_a,
    pred_b,
    weights: list[float],
    overfit_penalty: float = 0.25,
) -> tuple[dict, float]:
    """
    Evaluate weighted ensemble on validation.

    score = validation_r2 - overfit_penalty * abs(validation_r2 - blended_stability)
    Here blended_stability is approximated by preferring smoother mixtures and
    discouraging extreme validation-only gains.
    """
    best_metrics = None
    best_weight = None
    best_score = -np.inf

    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    for w in weights:
        pred = w * pred_a + (1.0 - w) * pred_b
        metrics = _build_metrics(y_true, pred)

        # small regularization against overly aggressive weights
        weight_penalty = overfit_penalty * abs(w - 0.5)
        score = metrics["r2"] - weight_penalty

        if score > best_score:
            best_score = score
            best_weight = w
            best_metrics = metrics

    return best_metrics, best_weight