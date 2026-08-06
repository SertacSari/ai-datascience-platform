from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import HTTPException, status

from app.services.training_utils import (
    RANDOM_STATE,
    TEST_SIZE,
    ensure_sklearn_dependencies_available,
    get_numeric_metric,
    split_feature_columns,
)


REGRESSION_MODEL_NAME = "RandomForestRegressor"
SMALL_DATASET_ROWS = 100
MIN_REGRESSION_ROWS = 20
WEAK_R2_THRESHOLD = 0.50
HIGH_ERROR_RATIO_THRESHOLD = 0.30
PREDICTION_SAMPLE_SIZE = 10

REGRESSION_METRIC_EXPLANATIONS = {
    "mae": "Average absolute prediction error in target units.",
    "rmse": "Typical prediction error, with larger mistakes weighted more heavily.",
    "r2_score": "How much target variation the model explains on the test split.",
}


def reject_bad_regression_data(message: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )


def validate_regression_target_column(
    df: pd.DataFrame,
    target_column: str,
) -> pd.Series:
    if target_column not in df.columns:
        reject_bad_regression_data("Target column does not exist in dataset")

    target = df[target_column]

    if not pd.api.types.is_numeric_dtype(target):
        reject_bad_regression_data("Regression target column must be numerical")

    if target.isna().any():
        reject_bad_regression_data("Target column must not contain missing values")

    if target.isin([float("inf"), float("-inf")]).any():
        reject_bad_regression_data("Target column must contain only finite values")

    return target


def validate_regression_features(
    df: pd.DataFrame,
    target_column: str,
) -> list[str]:
    feature_columns = [column for column in df.columns if column != target_column]

    if not feature_columns:
        reject_bad_regression_data(
            "Dataset must contain at least one feature column"
        )

    empty_feature_columns = [
        str(column)
        for column in feature_columns
        if df[column].isna().all()
    ]
    if empty_feature_columns:
        reject_bad_regression_data(
            "Feature columns must not be completely empty: "
            + ", ".join(empty_feature_columns)
        )

    infinite_feature_columns = []
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].isin(
            [float("inf"), float("-inf")]
        ).any():
            infinite_feature_columns.append(str(column))

    if infinite_feature_columns:
        reject_bad_regression_data(
            "Feature columns must contain only finite values: "
            + ", ".join(infinite_feature_columns)
        )

    return feature_columns


def validate_regression_ml_readiness(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(df) < MIN_REGRESSION_ROWS:
        reject_bad_regression_data(
            f"Regression requires at least {MIN_REGRESSION_ROWS} rows"
        )

    target = validate_regression_target_column(df, target_column)
    feature_columns = validate_regression_features(df, target_column)

    return df[feature_columns], target


def build_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
):
    ensure_sklearn_dependencies_available()

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_regression_interpretation(
    metrics: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    warnings = []
    mae = get_numeric_metric(metrics, "mae")
    rmse = get_numeric_metric(metrics, "rmse")
    r2 = get_numeric_metric(metrics, "r2_score")
    target_mean = get_numeric_metric(metrics, "target_mean")
    target_min = get_numeric_metric(metrics, "target_min")
    target_max = get_numeric_metric(metrics, "target_max")
    metric_values = [mae, rmse, r2, target_mean, target_min, target_max]

    if any(value is None for value in metric_values):
        warnings.append(
            {
                "code": "missing_metrics",
                "message": "Some regression metrics are missing, so performance should be reviewed carefully.",
            }
        )

    if r2 is not None and r2 < WEAK_R2_THRESHOLD:
        warnings.append(
            {
                "code": "weak_r2",
                "message": "The model explains limited target variation on the test split.",
            }
        )

    if (
        rmse is not None
        and target_mean is not None
        and target_min is not None
        and target_max is not None
    ):
        target_range = target_max - target_min
        high_vs_range = (
            target_range > 0
            and rmse / target_range > HIGH_ERROR_RATIO_THRESHOLD
        )
        high_vs_mean = (
            target_mean != 0
            and rmse / abs(target_mean) > HIGH_ERROR_RATIO_THRESHOLD
        )

        if high_vs_range or high_vs_mean:
            warnings.append(
                {
                    "code": "high_error",
                    "message": "Prediction error is large compared with the target scale.",
                }
            )

    if row_count < SMALL_DATASET_ROWS:
        warnings.append(
            {
                "code": "small_dataset",
                "message": "The dataset is small, so regression results may change with more data.",
            }
        )

    if any(value is None for value in metric_values):
        quality_level = "weak"
        summary = "The regression result is missing some metrics, so performance should be reviewed carefully."
    elif r2 >= 0.70 and not any(
        warning["code"] == "high_error" for warning in warnings
    ):
        quality_level = "good"
        summary = "The model shows good regression performance on the test split."
    elif r2 >= WEAK_R2_THRESHOLD:
        quality_level = "fair"
        summary = "The model shows fair regression performance, but errors should be reviewed."
    else:
        quality_level = "weak"
        summary = "The model shows weak regression performance and should not be relied on without review."

    return {
        "summary": summary,
        "quality_level": quality_level,
        "warnings": warnings,
        "metric_explanations": REGRESSION_METRIC_EXPLANATIONS,
    }


def train_regression_model(
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    ensure_sklearn_dependencies_available()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    numeric_features, categorical_features = split_feature_columns(x)
    pipeline = build_regression_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(math.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    prediction_sample = [
        {
            "actual": float(actual),
            "predicted": float(predicted),
        }
        for actual, predicted in zip(
            y_test.head(PREDICTION_SAMPLE_SIZE),
            predictions[:PREDICTION_SAMPLE_SIZE],
        )
    ]

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "target_mean": float(y.mean()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "test_size": TEST_SIZE,
        "model_name": REGRESSION_MODEL_NAME,
    }
    report_json = {
        "prediction_sample": prediction_sample,
        "test_size": TEST_SIZE,
        "model_name": REGRESSION_MODEL_NAME,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "interpretation": build_regression_interpretation(
            metrics=metrics,
            row_count=len(y),
        ),
    }

    return {
        "metrics": metrics,
        "report_json": report_json,
    }


def build_regression_model_result_payload(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    x, y = validate_regression_ml_readiness(
        df=df,
        target_column=target_column,
    )
    training_output = train_regression_model(x=x, y=y)

    return (
        REGRESSION_MODEL_NAME,
        training_output["metrics"],
        training_output["report_json"],
    )
