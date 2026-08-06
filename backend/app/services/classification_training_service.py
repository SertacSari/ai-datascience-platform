from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, status

from app.models.dataset import Dataset
from app.services.dataset_service import read_stored_dataset_file
from app.services.training_utils import (
    RANDOM_STATE,
    TEST_SIZE,
    ensure_sklearn_dependencies_available,
    get_numeric_metric,
    split_feature_columns,
)


MIN_CLASSIFICATION_ROWS = 20
MAX_CLASSIFICATION_CLASSES = 20
MIN_CLASS_EXAMPLES = 5
ID_LIKE_UNIQUE_RATIO = 0.9
CONTINUOUS_NUMERIC_UNIQUE_COUNT = 10
CONTINUOUS_NUMERIC_UNIQUE_RATIO = 0.2
MODEL_NAME = "RandomForestClassifier"
LOW_METRIC_THRESHOLD = 0.70
GOOD_METRIC_THRESHOLD = 0.85
CLASS_IMBALANCE_THRESHOLD = 0.70
SMALL_DATASET_ROWS = 100


METRIC_EXPLANATIONS = {
    "accuracy": "Overall share of correct predictions.",
    "precision": "When the model predicts a class, how often it is correct.",
    "recall": "How many real cases of a class the model catches.",
    "f1_score": "Balance between precision and recall.",
}

def reject_bad_classification_data(message: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )


def validate_classification_target_column(
    df: pd.DataFrame,
    target_column: str,
) -> pd.Series:
    if target_column not in df.columns:
        reject_bad_classification_data("Target column does not exist in dataset")

    target = df[target_column]

    if target.isna().any():
        reject_bad_classification_data("Target column must not contain missing values")

    if pd.api.types.is_numeric_dtype(target) and target.isin(
        [float("inf"), float("-inf")]
    ).any():
        reject_bad_classification_data("Target column must contain only finite values")

    return target


def validate_original_target_was_not_fabricated(
    dataset: Dataset,
    target_column: str,
) -> None:
    if not dataset.cleaned_file_path:
        return

    if not Path(dataset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original dataset file not found on server",
        )

    original_df = read_stored_dataset_file(dataset.file_path)

    if target_column not in original_df.columns:
        reject_bad_classification_data("Target column does not exist in dataset")

    original_target = original_df[target_column]

    if original_target.isna().any():
        reject_bad_classification_data(
            "Target column must not contain missing values before cleaning"
        )

    if pd.api.types.is_numeric_dtype(original_target) and original_target.isin(
        [float("inf"), float("-inf")]
    ).any():
        reject_bad_classification_data(
            "Target column must contain only finite values before cleaning"
        )


def validate_classification_features(
    df: pd.DataFrame,
    target_column: str,
) -> list[str]:
    feature_columns = [column for column in df.columns if column != target_column]

    if not feature_columns:
        reject_bad_classification_data(
            "Dataset must contain at least one feature column"
        )

    empty_feature_columns = [
        str(column)
        for column in feature_columns
        if df[column].isna().all()
    ]
    if empty_feature_columns:
        reject_bad_classification_data(
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
        reject_bad_classification_data(
            "Feature columns must contain only finite values: "
            + ", ".join(infinite_feature_columns)
        )

    return feature_columns


def validate_classification_class_distribution(
    target: pd.Series,
) -> dict[str, int]:
    row_count = len(target)
    class_distribution = target.value_counts(dropna=False)
    unique_class_count = int(class_distribution.shape[0])
    unique_ratio = unique_class_count / row_count if row_count else 0

    if row_count < MIN_CLASSIFICATION_ROWS:
        reject_bad_classification_data(
            f"Classification requires at least {MIN_CLASSIFICATION_ROWS} rows"
        )

    if unique_class_count < 2:
        reject_bad_classification_data(
            "Classification target must have at least 2 classes"
        )

    if unique_ratio >= ID_LIKE_UNIQUE_RATIO:
        reject_bad_classification_data(
            "Classification target looks ID-like because most values are unique"
        )

    if (
        pd.api.types.is_numeric_dtype(target)
        and unique_class_count > CONTINUOUS_NUMERIC_UNIQUE_COUNT
        and unique_ratio > CONTINUOUS_NUMERIC_UNIQUE_RATIO
    ):
        reject_bad_classification_data(
            "Numeric classification target looks continuous; use regression instead"
        )

    if unique_class_count > MAX_CLASSIFICATION_CLASSES:
        reject_bad_classification_data(
            f"Classification target must have at most {MAX_CLASSIFICATION_CLASSES} classes"
        )

    too_small_classes = class_distribution[
        class_distribution < MIN_CLASS_EXAMPLES
    ]
    if not too_small_classes.empty:
        reject_bad_classification_data(
            f"Each class needs at least {MIN_CLASS_EXAMPLES} examples"
        )

    return {str(label): int(count) for label, count in class_distribution.items()}


def validate_classification_ml_readiness(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    target = validate_classification_target_column(df, target_column)
    feature_columns = validate_classification_features(df, target_column)
    class_distribution = validate_classification_class_distribution(target)

    return df[feature_columns], target, class_distribution


def build_classification_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
):
    ensure_sklearn_dependencies_available()

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
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
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_classification_model(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[Any, pd.Series, pd.Series, dict[str, Any]]:
    ensure_sklearn_dependencies_available()

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features, categorical_features = split_feature_columns(x)
    pipeline = build_classification_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    labels = sorted(y.astype(str).unique())
    y_test_as_text = y_test.astype(str)
    predictions_as_text = pd.Series(predictions).astype(str)

    metrics = {
        "accuracy": float(accuracy_score(y_test_as_text, predictions_as_text)),
        "precision": float(
            precision_score(
                y_test_as_text,
                predictions_as_text,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test_as_text,
                predictions_as_text,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test_as_text,
                predictions_as_text,
                average="weighted",
                zero_division=0,
            )
        ),
    }
    report_json = {
        "confusion_matrix": confusion_matrix(
            y_test_as_text,
            predictions_as_text,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            y_test_as_text,
            predictions_as_text,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "test_size": TEST_SIZE,
        "model_name": MODEL_NAME,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    return pipeline, y_test, pd.Series(predictions), {
        "metrics": metrics,
        "report_json": report_json,
    }


def build_classification_interpretation(
    metrics: dict[str, Any],
    class_distribution: dict[str, int],
) -> dict[str, Any]:
    warnings = []
    accuracy = get_numeric_metric(metrics, "accuracy")
    precision = get_numeric_metric(metrics, "precision")
    recall = get_numeric_metric(metrics, "recall")
    f1_score = get_numeric_metric(metrics, "f1_score")
    metric_values = [accuracy, precision, recall, f1_score]

    if any(value is None for value in metric_values):
        warnings.append(
            {
                "code": "missing_metrics",
                "message": "Some model metrics are missing, so performance should be reviewed carefully.",
            }
        )

    if accuracy is not None and accuracy < LOW_METRIC_THRESHOLD:
        warnings.append(
            {
                "code": "low_accuracy",
                "message": "Overall correctness is low, so predictions may be unreliable.",
            }
        )

    if f1_score is not None and f1_score < LOW_METRIC_THRESHOLD:
        warnings.append(
            {
                "code": "weak_f1",
                "message": "The balance between precision and recall is weak.",
            }
        )

    if precision is not None and precision < LOW_METRIC_THRESHOLD:
        warnings.append(
            {
                "code": "low_precision",
                "message": "When the model predicts a class, it may be wrong too often because precision is low.",
            }
        )

    if recall is not None and recall < LOW_METRIC_THRESHOLD:
        warnings.append(
            {
                "code": "low_recall",
                "message": "The model may miss some real cases because recall is low.",
            }
        )

    row_count = sum(class_distribution.values())
    largest_class_count = max(class_distribution.values(), default=0)
    largest_class_ratio = largest_class_count / row_count if row_count else 0

    if largest_class_ratio > CLASS_IMBALANCE_THRESHOLD:
        warnings.append(
            {
                "code": "class_imbalance",
                "message": "One class dominates the dataset, so the model may favor that class.",
            }
        )

    if row_count and row_count < SMALL_DATASET_ROWS:
        warnings.append(
            {
                "code": "small_dataset",
                "message": "The dataset is small, so the result may change with more data.",
            }
        )

    if any(value is None for value in metric_values):
        quality_level = "weak"
        summary = "The model result is missing some metrics, so performance should be reviewed carefully."
    elif accuracy >= GOOD_METRIC_THRESHOLD and f1_score >= GOOD_METRIC_THRESHOLD:
        quality_level = "good"
        summary = "The model shows good classification performance on the test split."
    elif accuracy >= LOW_METRIC_THRESHOLD and f1_score >= LOW_METRIC_THRESHOLD:
        quality_level = "fair"
        summary = "The model shows fair classification performance, but results should be reviewed."
    else:
        quality_level = "weak"
        summary = "The model shows weak classification performance and should not be relied on without review."

    return {
        "summary": summary,
        "quality_level": quality_level,
        "warnings": warnings,
        "metric_explanations": METRIC_EXPLANATIONS,
    }


def build_model_result_payload(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    x, y, class_distribution = validate_classification_ml_readiness(
        df=df,
        target_column=target_column,
    )
    _, _, _, training_output = train_classification_model(x=x, y=y)

    metrics = {
        **training_output["metrics"],
        "class_distribution": class_distribution,
        "test_size": TEST_SIZE,
        "model_name": MODEL_NAME,
    }
    report_json = {
        **training_output["report_json"],
        "interpretation": build_classification_interpretation(
            metrics=metrics,
            class_distribution=class_distribution,
        ),
    }

    return MODEL_NAME, metrics, report_json
