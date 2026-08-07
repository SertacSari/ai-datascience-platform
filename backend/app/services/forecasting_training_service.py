from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import HTTPException, status

from app.services.training_utils import (
    TEST_SIZE,
    RANDOM_STATE,
    ensure_sklearn_dependencies_available,
    get_numeric_metric,
    split_feature_columns,
)


FORECASTING_MODEL_NAME = "RandomForestRegressorForecasting"
MIN_FORECASTING_ROWS = 30
PREDICTION_SAMPLE_SIZE = 10
WEAK_R2_THRESHOLD = 0.50
HIGH_ERROR_RATIO_THRESHOLD = 0.30
HIGH_MAPE_THRESHOLD = 20.0
SMALL_DATASET_ROWS = 100
SHORT_DATE_RANGE_DAYS = 30

FORECASTING_METRIC_EXPLANATIONS = {
    "mae": "Average absolute forecast error in target units.",
    "rmse": "Typical forecast error, with larger mistakes weighted more heavily.",
    "mape": "Average percentage forecast error when actual values are non-zero.",
    "r2_score": "How much target variation the model explains on the held-out time period.",
}


def reject_bad_forecasting_data(message: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )


def get_forecasting_date_column(config_json: dict[str, Any]) -> str:
    date_column = (config_json or {}).get("date_column")

    if not isinstance(date_column, str) or not date_column.strip():
        reject_bad_forecasting_data("Forecasting requires config_json.date_column")

    return date_column.strip()


def validate_forecasting_ml_readiness(
    df: pd.DataFrame,
    target_column: str,
    config_json: dict[str, Any],
) -> tuple[pd.DataFrame, str, pd.Series]:
    date_column = get_forecasting_date_column(config_json)

    if date_column == target_column:
        reject_bad_forecasting_data(
            "Forecasting date column must be different from target column"
        )

    if date_column not in df.columns:
        reject_bad_forecasting_data(
            "Forecasting date column does not exist in dataset"
        )

    if target_column not in df.columns:
        reject_bad_forecasting_data("Target column does not exist in dataset")

    parsed_dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
        format="mixed",
    )

    if parsed_dates.isna().any():
        reject_bad_forecasting_data(
            "Forecasting date column contains invalid or missing dates"
        )

    if parsed_dates.duplicated().any():
        reject_bad_forecasting_data(
            "Forecasting date column must contain unique values"
        )

    target = df[target_column]

    if not pd.api.types.is_numeric_dtype(target):
        reject_bad_forecasting_data("Forecasting target column must be numerical")

    if target.isna().any():
        reject_bad_forecasting_data("Target column must not contain missing values")

    if target.isin([float("inf"), float("-inf")]).any():
        reject_bad_forecasting_data("Target column must contain only finite values")

    if len(df) < MIN_FORECASTING_ROWS:
        reject_bad_forecasting_data(
            f"Forecasting requires at least {MIN_FORECASTING_ROWS} rows"
        )

    sorted_df = df.copy()
    sorted_df[date_column] = parsed_dates
    sorted_df = sorted_df.sort_values(date_column).reset_index(drop=True)

    return sorted_df, date_column, sorted_df[target_column]


def validate_forecasting_extra_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    empty_feature_columns = [
        str(column)
        for column in feature_columns
        if df[column].isna().all()
    ]
    if empty_feature_columns:
        reject_bad_forecasting_data(
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
        reject_bad_forecasting_data(
            "Feature columns must contain only finite values: "
            + ", ".join(infinite_feature_columns)
        )


def build_forecasting_features(
    sorted_df: pd.DataFrame,
    date_column: str,
    target_column: str,
) -> pd.DataFrame:
    dates = sorted_df[date_column]
    features = pd.DataFrame(
        {
            "time_index": range(len(sorted_df)),
            "year": dates.dt.year,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "day_of_week": dates.dt.dayofweek,
        }
    )
    extra_feature_columns = [
        column
        for column in sorted_df.columns
        if column not in {date_column, target_column}
    ]
    validate_forecasting_extra_features(sorted_df, extra_feature_columns)

    for column in extra_feature_columns:
        features[str(column)] = sorted_df[column]

    return features


def build_forecasting_pipeline(
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


def calculate_mape(y_test: pd.Series, predictions) -> float | None:
    non_zero_mask = y_test != 0

    if not non_zero_mask.any():
        return None

    absolute_percentage_errors = (
        (y_test[non_zero_mask] - predictions[non_zero_mask]).abs()
        / y_test[non_zero_mask].abs()
    )
    return float(absolute_percentage_errors.mean() * 100)


def build_forecasting_interpretation(
    metrics: dict[str, Any],
    row_count: int,
    date_range_days: int,
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
                "message": "Some forecasting metrics are missing, so performance should be reviewed carefully.",
            }
        )

    if r2 is not None and r2 < WEAK_R2_THRESHOLD:
        warnings.append(
            {
                "code": "weak_r2",
                "message": "The model explains limited target variation in the held-out time period.",
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
                    "message": "Forecast error is large compared with the target scale.",
                }
            )

    mape = get_numeric_metric(metrics, "mape")
    if mape is not None and mape > HIGH_MAPE_THRESHOLD:
        warnings.append(
            {
                "code": "high_mape",
                "message": "Average percentage forecast error is high.",
            }
        )

    if row_count < SMALL_DATASET_ROWS:
        warnings.append(
            {
                "code": "small_dataset",
                "message": "The dataset is small, so forecasting results may change with more history.",
            }
        )

    if date_range_days < SHORT_DATE_RANGE_DAYS:
        warnings.append(
            {
                "code": "short_date_range",
                "message": "The date range is short, so long-term forecasting confidence is limited.",
            }
        )

    if any(value is None for value in metric_values):
        quality_level = "weak"
        summary = "The forecasting result is missing some metrics, so performance should be reviewed carefully."
    elif r2 >= 0.70 and not any(
        warning["code"] in {"high_error", "high_mape"} for warning in warnings
    ):
        quality_level = "good"
        summary = "The model shows good forecasting performance on the held-out time period."
    elif r2 >= WEAK_R2_THRESHOLD:
        quality_level = "fair"
        summary = "The model shows fair forecasting performance, but errors should be reviewed."
    else:
        quality_level = "weak"
        summary = "The model shows weak forecasting performance and should not be relied on without review."

    return {
        "summary": summary,
        "quality_level": quality_level,
        "warnings": warnings,
        "metric_explanations": FORECASTING_METRIC_EXPLANATIONS,
    }


def train_forecasting_model(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
) -> dict[str, Any]:
    ensure_sklearn_dependencies_available()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    split_index = int(len(features) * (1 - TEST_SIZE))
    x_train = features.iloc[:split_index]
    x_test = features.iloc[split_index:]
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]
    test_dates = dates.iloc[split_index:]

    numeric_features, categorical_features = split_feature_columns(features)
    pipeline = build_forecasting_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    predictions_series = pd.Series(predictions, index=y_test.index)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(math.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    mape = calculate_mape(y_test, predictions_series)
    prediction_sample = [
        {
            "date": date.isoformat(),
            "actual": float(actual),
            "predicted": float(predicted),
        }
        for date, actual, predicted in zip(
            test_dates.dt.date.head(PREDICTION_SAMPLE_SIZE),
            y_test.head(PREDICTION_SAMPLE_SIZE),
            predictions[:PREDICTION_SAMPLE_SIZE],
        )
    ]

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2_score": r2,
        "target_mean": float(target.mean()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "test_size": TEST_SIZE,
        "model_name": FORECASTING_MODEL_NAME,
    }

    return {
        "metrics": metrics,
        "prediction_sample": prediction_sample,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_start_date": test_dates.iloc[0].date().isoformat(),
        "test_end_date": test_dates.iloc[-1].date().isoformat(),
    }


def build_forecasting_model_result_payload(
    df: pd.DataFrame,
    target_column: str,
    config_json: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    sorted_df, date_column, target = validate_forecasting_ml_readiness(
        df=df,
        target_column=target_column,
        config_json=config_json,
    )
    features = build_forecasting_features(
        sorted_df=sorted_df,
        date_column=date_column,
        target_column=target_column,
    )
    training_output = train_forecasting_model(
        features=features,
        target=target,
        dates=sorted_df[date_column],
    )
    date_range_days = int(
        (sorted_df[date_column].max() - sorted_df[date_column].min()).days
    )
    report_json = {
        "prediction_sample": training_output["prediction_sample"],
        "date_column": date_column,
        "target_column": target_column,
        "test_start_date": training_output["test_start_date"],
        "test_end_date": training_output["test_end_date"],
        "numeric_features": training_output["numeric_features"],
        "categorical_features": training_output["categorical_features"],
        "interpretation": build_forecasting_interpretation(
            metrics=training_output["metrics"],
            row_count=len(target),
            date_range_days=date_range_days,
        ),
    }

    return (
        FORECASTING_MODEL_NAME,
        training_output["metrics"],
        report_json,
    )
