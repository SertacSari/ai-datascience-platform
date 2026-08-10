from typing import Any

import pandas as pd


TEST_SIZE = 0.2
RANDOM_STATE = 42


def ensure_sklearn_dependencies_available() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for model training") from exc


def split_feature_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        str(column)
        for column in x.columns
        if pd.api.types.is_numeric_dtype(x[column])
    ]
    categorical_features = [
        str(column)
        for column in x.columns
        if str(column) not in numeric_features
    ]

    return numeric_features, categorical_features


def get_numeric_metric(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)

    if isinstance(value, (int, float)):
        return float(value)

    return None


def build_recommended_actions(
    warnings: list[dict[str, str]],
    action_map: dict[str, str],
) -> list[str]:
    actions = []
    for warning in warnings:
        action = action_map.get(warning["code"])
        if action and action not in actions:
            actions.append(action)

    return actions
