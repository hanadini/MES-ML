from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_xgb_model(**overrides):
    params = {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.3,
        "reg_lambda": 3.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBRegressor(**params)


# XGB_PARAM_GRID = {
#     "n_estimators": [300, 500, 700, 900],
#     "max_depth": [3, 4, 5, 6],
#     "learning_rate": [0.01, 0.03, 0.05, 0.08],
#     "subsample": [0.7, 0.8, 0.9, 1.0],
#     "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
#     "min_child_weight": [1, 3, 5, 7],
#     "gamma": [0.0, 0.1, 0.3, 0.5],
#     "reg_alpha": [0.0, 0.1, 0.3, 1.0],
#     "reg_lambda": [1.0, 3.0, 5.0, 10.0],
# }


def get_lgbm_model():
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )