import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from app.models.analysis_job import AnalysisJob
from app.models.dataset import Dataset
from app.models.enums import JobStatus, TaskType
from app.models.model_result import ModelResult
from app.models.user import User
from app.schemas.analysis import AnalysisJobRunResponse
from app.services.analysis_service import create_analysis_job, get_analysis_job_result
from app.services.classification_training_service import (
    build_classification_interpretation,
    validate_classification_ml_readiness,
)
from app.services.forecasting_training_service import build_forecasting_interpretation
from app.services.model_training_service import run_analysis_job
from app.services.regression_training_service import build_regression_interpretation


def add_user(db_session, username: str) -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        password_hash="not-used-in-tests",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def add_dataset_with_frame(
    db_session,
    tmp_path,
    user: User,
    dataframe: pd.DataFrame,
) -> Dataset:
    dataset_path = tmp_path / f"{user.username}_dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    dataset = Dataset(
        user_id=user.id,
        file_name=dataset_path.name,
        file_path=str(dataset_path),
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


def add_dataset_with_paths(
    db_session,
    user: User,
    file_path: str,
    cleaned_file_path: str | None = None,
) -> Dataset:
    dataset = Dataset(
        user_id=user.id,
        file_name="dataset.csv",
        file_path=file_path,
        cleaned_file_path=cleaned_file_path,
        row_count=20,
        column_count=3,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


def add_analysis_job(
    db_session,
    user: User,
    dataset: Dataset,
    task_type: TaskType = TaskType.CLASSIFICATION,
    target_column: str = "target",
    config_json: dict | None = None,
) -> AnalysisJob:
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type=task_type,
        target_column=target_column,
        config_json=config_json or {},
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job


def valid_classification_frame(row_count: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        target = "yes" if index % 2 else "no"
        rows.append(
            {
                "age": 20 + (index % 15),
                "segment": "A" if target == "yes" else "B",
                "score": index % 7,
                "target": target,
            }
        )
    return pd.DataFrame(rows)


def valid_regression_frame(row_count: int = 80) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        size = 700 + (index * 13)
        rooms = 2 + (index % 4)
        neighborhood = "central" if index % 3 == 0 else "suburban"
        neighborhood_bonus = 25_000 if neighborhood == "central" else 8_000
        price = 50_000 + (size * 180) + (rooms * 12_000) + neighborhood_bonus
        rows.append(
            {
                "size_sqft": size,
                "rooms": rooms,
                "neighborhood": neighborhood,
                "price": price,
            }
        )
    return pd.DataFrame(rows)


def valid_forecasting_frame(row_count: int = 60) -> pd.DataFrame:
    rows = []
    start_date = pd.Timestamp("2026-01-01")
    for index in range(row_count):
        date = start_date + pd.Timedelta(days=index)
        promotion = "yes" if index % 10 in {0, 1} else "no"
        promotion_bonus = 45 if promotion == "yes" else 0
        sales = 200 + (index * 3.5) + ((index % 7) * 8) + promotion_bonus
        rows.append(
            {
                "date": date.date().isoformat(),
                "promotion": promotion,
                "store_visits": 100 + (index * 2),
                "sales": sales,
            }
        )
    return pd.DataFrame(rows)


def assert_regression_create_rejected_without_job(
    db_session,
    tmp_path,
    dataframe: pd.DataFrame,
    expected_detail: str,
    username: str,
) -> None:
    user = add_user(db_session, username)
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)

    with pytest.raises(HTTPException) as error:
        create_analysis_job(
            db=db_session,
            dataset_id=dataset.id,
            task_type=TaskType.REGRESSION,
            target_column="price",
            config_json={},
            current_user=user,
        )

    assert error.value.status_code == 400
    assert expected_detail in error.value.detail
    assert db_session.query(AnalysisJob).count() == 0


def test_valid_classification_job_can_run_and_creates_model_result(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "owner")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_classification_frame(),
    )
    job = add_analysis_job(db_session, user, dataset)

    updated_job, model_result = run_analysis_job(db_session, job.id, user)

    assert updated_job.status == JobStatus.COMPLETED
    assert updated_job.finished_at is not None
    assert model_result.analysis_id == job.id
    assert model_result.model_name == "RandomForestClassifier"
    assert set(model_result.metrics) >= {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "class_distribution",
        "test_size",
        "model_name",
    }
    assert "confusion_matrix" not in model_result.metrics
    assert "classification_report" not in model_result.metrics
    assert "confusion_matrix" in model_result.report_json
    assert "classification_report" in model_result.report_json
    assert "interpretation" in model_result.report_json
    assert set(model_result.report_json["interpretation"]) == {
        "summary",
        "quality_level",
        "warnings",
        "metric_explanations",
    }
    assert "class_distribution" not in model_result.report_json
    assert (
        db_session.query(ModelResult)
        .filter(ModelResult.analysis_id == job.id)
        .count()
        == 1
    )

    response = AnalysisJobRunResponse(job=updated_job, model_result=model_result)
    assert response.job.id == job.id
    assert response.model_result.id == model_result.id


def test_non_owned_job_cannot_run(db_session, tmp_path) -> None:
    owner = add_user(db_session, "owner")
    other_user = add_user(db_session, "other")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        owner,
        valid_classification_frame(),
    )
    job = add_analysis_job(db_session, owner, dataset)

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, other_user)

    assert error.value.status_code == 404


def test_valid_forecasting_job_can_run_and_creates_model_result(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "forecasting_user")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_forecasting_frame(),
    )
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    updated_job, model_result = run_analysis_job(db_session, job.id, user)
    persisted_result = get_analysis_job_result(db_session, job.id, user)

    assert updated_job.status == JobStatus.COMPLETED
    assert model_result.id == persisted_result.id
    assert model_result.model_name == "RandomForestRegressorForecasting"
    assert set(model_result.metrics) >= {
        "mae",
        "rmse",
        "mape",
        "r2_score",
        "target_mean",
        "target_min",
        "target_max",
        "test_size",
        "model_name",
    }
    assert model_result.metrics["model_name"] == "RandomForestRegressorForecasting"
    assert "prediction_sample" in model_result.report_json
    assert len(model_result.report_json["prediction_sample"]) <= 10
    assert set(model_result.report_json["prediction_sample"][0]) == {
        "date",
        "actual",
        "predicted",
    }
    assert model_result.report_json["date_column"] == "date"
    assert model_result.report_json["target_column"] == "sales"
    assert model_result.report_json["test_start_date"] < model_result.report_json[
        "test_end_date"
    ]
    assert "numeric_features" in model_result.report_json
    assert "categorical_features" in model_result.report_json
    assert "interpretation" in model_result.report_json
    assert set(model_result.report_json["interpretation"]) == {
        "summary",
        "quality_level",
        "warnings",
        "metric_explanations",
    }


def test_non_owned_forecasting_job_cannot_run(db_session, tmp_path) -> None:
    owner = add_user(db_session, "forecasting_owner")
    other_user = add_user(db_session, "forecasting_other")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        owner,
        valid_forecasting_frame(),
    )
    job = add_analysis_job(
        db_session,
        owner,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, other_user)

    assert error.value.status_code == 404


def test_forecasting_rejects_missing_date_column_config(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "forecasting_missing_config")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_forecasting_frame(),
    )
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "config_json.date_column" in error.value.detail


@pytest.mark.parametrize(
    "date_value,expected_detail",
    [
        ("not-a-date", "invalid or missing dates"),
        (None, "invalid or missing dates"),
    ],
)
def test_forecasting_rejects_invalid_dates(
    db_session,
    tmp_path,
    date_value,
    expected_detail: str,
) -> None:
    user = add_user(db_session, "forecasting_invalid_dates")
    dataframe = valid_forecasting_frame()
    dataframe.loc[0, "date"] = date_value
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert expected_detail in error.value.detail


def test_forecasting_rejects_duplicate_dates(db_session, tmp_path) -> None:
    user = add_user(db_session, "forecasting_duplicate_dates")
    dataframe = valid_forecasting_frame()
    dataframe.loc[1, "date"] = dataframe.loc[0, "date"]
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "unique values" in error.value.detail


def test_forecasting_rejects_date_column_same_as_target(db_session, tmp_path) -> None:
    user = add_user(db_session, "forecasting_same_date_target")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_forecasting_frame(),
    )
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "sales"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "different from target" in error.value.detail


def test_forecasting_rejects_non_numeric_target(db_session, tmp_path) -> None:
    user = add_user(db_session, "forecasting_non_numeric_target")
    dataframe = valid_forecasting_frame()
    dataframe["sales"] = [
        "high" if index % 2 else "low" for index in range(len(dataframe))
    ]
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "Forecasting target column must be numerical" in error.value.detail


@pytest.mark.parametrize(
    "target_value,expected_detail",
    [
        (None, "missing values"),
        (float("inf"), "finite values"),
    ],
)
def test_forecasting_rejects_missing_or_infinite_target(
    db_session,
    tmp_path,
    target_value,
    expected_detail: str,
) -> None:
    user = add_user(db_session, f"forecasting_{expected_detail.replace(' ', '_')}")
    dataframe = valid_forecasting_frame()
    dataframe["sales"] = dataframe["sales"].astype(float)
    dataframe.loc[0, "sales"] = target_value
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert expected_detail in error.value.detail


def test_forecasting_rejects_too_few_rows(db_session, tmp_path) -> None:
    user = add_user(db_session, "forecasting_too_few_rows")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_forecasting_frame(row_count=29),
    )
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        config_json={"date_column": "date"},
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "at least 30 rows" in error.value.detail


def test_forecasting_interpretation_warns_for_weak_metrics() -> None:
    interpretation = build_forecasting_interpretation(
        metrics={
            "mae": 35.0,
            "rmse": 45.0,
            "mape": 25.0,
            "r2_score": 0.25,
            "target_mean": 100.0,
            "target_min": 50.0,
            "target_max": 150.0,
        },
        row_count=150,
        date_range_days=120,
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "weak"
    assert {"weak_r2", "high_error", "high_mape"} <= warning_codes
    assert interpretation["metric_explanations"]["mape"] == (
        "Average percentage forecast error when actual values are non-zero."
    )


def test_forecasting_interpretation_warns_for_small_dataset_and_short_range() -> None:
    interpretation = build_forecasting_interpretation(
        metrics={
            "mae": 5.0,
            "rmse": 8.0,
            "mape": 6.0,
            "r2_score": 0.76,
            "target_mean": 100.0,
            "target_min": 50.0,
            "target_max": 150.0,
        },
        row_count=60,
        date_range_days=20,
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "good"
    assert warning_codes == {"small_dataset", "short_date_range"}


def test_forecasting_interpretation_warns_for_missing_metrics() -> None:
    interpretation = build_forecasting_interpretation(
        metrics={
            "mae": 5.0,
            "rmse": 8.0,
            "r2_score": 0.76,
        },
        row_count=150,
        date_range_days=120,
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "weak"
    assert warning_codes == {"missing_metrics"}
    assert "missing some metrics" in interpretation["summary"]


def test_valid_regression_job_can_run_and_creates_model_result(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "regression_owner")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_regression_frame(),
    )
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.REGRESSION,
        target_column="price",
    )

    updated_job, model_result = run_analysis_job(db_session, job.id, user)
    persisted_result = get_analysis_job_result(db_session, job.id, user)

    assert updated_job.status == JobStatus.COMPLETED
    assert model_result.id == persisted_result.id
    assert model_result.model_name == "RandomForestRegressor"
    assert set(model_result.metrics) >= {
        "mae",
        "rmse",
        "r2_score",
        "target_mean",
        "target_min",
        "target_max",
        "test_size",
        "model_name",
    }
    assert model_result.metrics["model_name"] == "RandomForestRegressor"
    assert "prediction_sample" in model_result.report_json
    assert len(model_result.report_json["prediction_sample"]) <= 10
    assert set(model_result.report_json["prediction_sample"][0]) == {
        "actual",
        "predicted",
    }
    assert "interpretation" in model_result.report_json
    assert set(model_result.report_json["interpretation"]) == {
        "summary",
        "quality_level",
        "warnings",
        "metric_explanations",
    }


def test_valid_regression_request_creates_job_with_created_status(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "regression_create_valid")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_regression_frame(),
    )

    job = create_analysis_job(
        db=db_session,
        dataset_id=dataset.id,
        task_type=TaskType.REGRESSION,
        target_column="price",
        config_json={},
        current_user=user,
    )

    assert job.status == JobStatus.CREATED
    assert job.task_type == TaskType.REGRESSION
    assert job.target_column == "price"


def test_non_owned_regression_job_cannot_run(db_session, tmp_path) -> None:
    owner = add_user(db_session, "regression_owner")
    other_user = add_user(db_session, "regression_other")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        owner,
        valid_regression_frame(),
    )
    job = add_analysis_job(
        db_session,
        owner,
        dataset,
        task_type=TaskType.REGRESSION,
        target_column="price",
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, other_user)

    assert error.value.status_code == 404


def test_regression_rejects_non_numeric_target(db_session, tmp_path) -> None:
    user = add_user(db_session, "regression_non_numeric")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_classification_frame(),
    )
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.REGRESSION,
        target_column="target",
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "Regression target column must be numerical" in error.value.detail


@pytest.mark.parametrize(
    "target_value,expected_detail",
    [
        (None, "missing values"),
        (float("inf"), "finite values"),
    ],
)
def test_regression_rejects_missing_or_infinite_target(
    db_session,
    tmp_path,
    target_value,
    expected_detail: str,
) -> None:
    user = add_user(db_session, f"regression_{expected_detail.replace(' ', '_')}")
    dataframe = valid_regression_frame()
    dataframe["price"] = dataframe["price"].astype(float)
    dataframe.loc[0, "price"] = target_value
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.REGRESSION,
        target_column="price",
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert expected_detail in error.value.detail


def test_regression_rejects_no_usable_feature_columns(db_session, tmp_path) -> None:
    user = add_user(db_session, "regression_no_features")
    dataframe = pd.DataFrame({"price": [100_000 + index for index in range(20)]})
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(
        db_session,
        user,
        dataset,
        task_type=TaskType.REGRESSION,
        target_column="price",
    )

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "feature column" in error.value.detail


def test_create_regression_job_rejects_non_numeric_target(
    db_session,
    tmp_path,
) -> None:
    dataframe = valid_classification_frame()
    dataframe = dataframe.rename(columns={"target": "price"})

    assert_regression_create_rejected_without_job(
        db_session=db_session,
        tmp_path=tmp_path,
        dataframe=dataframe,
        expected_detail="Regression target column must be numerical",
        username="create_regression_non_numeric",
    )


@pytest.mark.parametrize(
    "target_value,expected_detail,username",
    [
        (None, "missing values", "create_regression_missing"),
        (float("inf"), "finite values", "create_regression_infinite"),
    ],
)
def test_create_regression_job_rejects_missing_or_infinite_target(
    db_session,
    tmp_path,
    target_value,
    expected_detail: str,
    username: str,
) -> None:
    dataframe = valid_regression_frame()
    dataframe["price"] = dataframe["price"].astype(float)
    dataframe.loc[0, "price"] = target_value

    assert_regression_create_rejected_without_job(
        db_session=db_session,
        tmp_path=tmp_path,
        dataframe=dataframe,
        expected_detail=expected_detail,
        username=username,
    )


def test_create_regression_job_rejects_no_usable_feature_columns(
    db_session,
    tmp_path,
) -> None:
    dataframe = pd.DataFrame({"price": [100_000 + index for index in range(20)]})

    assert_regression_create_rejected_without_job(
        db_session=db_session,
        tmp_path=tmp_path,
        dataframe=dataframe,
        expected_detail="feature column",
        username="create_regression_no_features",
    )


def test_create_regression_job_rejects_too_few_rows(
    db_session,
    tmp_path,
) -> None:
    dataframe = valid_regression_frame(row_count=19)

    assert_regression_create_rejected_without_job(
        db_session=db_session,
        tmp_path=tmp_path,
        dataframe=dataframe,
        expected_detail="at least 20 rows",
        username="create_regression_too_few_rows",
    )


@pytest.mark.parametrize(
    "job_status",
    [JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED],
)
def test_only_created_jobs_can_run(
    db_session,
    tmp_path,
    job_status: JobStatus,
) -> None:
    user = add_user(db_session, f"status_{job_status.value}")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_classification_frame(),
    )
    job = add_analysis_job(db_session, user, dataset)
    job.status = job_status
    db_session.commit()

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 409
    assert error.value.detail == "Only created jobs can be run"


@pytest.mark.parametrize(
    "dataframe,expected_detail",
    [
        (
            pd.DataFrame({"target": ["yes", "no"] * 10}),
            "feature column",
        ),
        (
            pd.DataFrame(
                {
                    "feature": range(20),
                    "target": [f"id_{index}" for index in range(20)],
                }
            ),
            "ID-like",
        ),
        (
            pd.DataFrame(
                {
                    "feature": range(10),
                    "target": ["yes", "no"] * 5,
                }
            ),
            "at least 20 rows",
        ),
        (
            pd.DataFrame(
                {
                    "feature": range(20),
                    "target": ["yes"] * 20,
                }
            ),
            "at least 2 classes",
        ),
        (
            pd.DataFrame(
                {
                    "feature": range(24),
                    "target": ["majority"] * 20 + ["minority"] * 4,
                }
            ),
            "at least 5 examples",
        ),
        (
            pd.DataFrame(
                {
                    "feature": range(40),
                    "target": [float(index % 12) + 0.25 for index in range(40)],
                }
            ),
            "looks continuous",
        ),
        (
            pd.DataFrame(
                {
                    "feature": [float("inf")] + list(range(1, 20)),
                    "target": ["yes", "no"] * 10,
                }
            ),
            "finite values",
        ),
        (
            pd.DataFrame(
                {
                    "feature": [None] * 20,
                    "target": ["yes", "no"] * 10,
                }
            ),
            "completely empty",
        ),
    ],
)
def test_invalid_classification_training_data_is_rejected(
    db_session,
    tmp_path,
    dataframe: pd.DataFrame,
    expected_detail: str,
) -> None:
    user = add_user(db_session, expected_detail.replace(" ", "_").lower())
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)
    job = add_analysis_job(db_session, user, dataset)

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert expected_detail in error.value.detail

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED


def test_classification_job_creation_rejects_bad_target_before_run(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "creation_rejects_bad_target")
    dataframe = pd.DataFrame(
        {
            "feature": range(20),
            "target": [f"id_{index}" for index in range(20)],
        }
    )
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)

    with pytest.raises(HTTPException) as error:
        create_analysis_job(
            db=db_session,
            dataset_id=dataset.id,
            task_type=TaskType.CLASSIFICATION,
            target_column="target",
            config_json={},
            current_user=user,
        )

    assert error.value.status_code == 400
    assert "ID-like" in error.value.detail
    assert db_session.query(AnalysisJob).count() == 0


def test_missing_target_values_are_rejected_instead_of_filled(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "missing_target")
    original_path = tmp_path / "original.csv"
    cleaned_path = tmp_path / "cleaned.csv"
    valid_classification_frame().to_csv(original_path, index=False)
    dataframe_with_missing_target = valid_classification_frame()
    dataframe_with_missing_target.loc[0, "target"] = None
    dataframe_with_missing_target.to_csv(cleaned_path, index=False)
    dataset = add_dataset_with_paths(
        db_session,
        user,
        file_path=str(original_path),
        cleaned_file_path=str(cleaned_path),
    )

    with pytest.raises(HTTPException) as creation_error:
        create_analysis_job(
            db=db_session,
            dataset_id=dataset.id,
            task_type=TaskType.CLASSIFICATION,
            target_column="target",
            config_json={},
            current_user=user,
        )

    assert creation_error.value.status_code == 400
    assert "missing values" in creation_error.value.detail

    job = add_analysis_job(db_session, user, dataset)

    with pytest.raises(HTTPException) as run_error:
        run_analysis_job(db_session, job.id, user)

    assert run_error.value.status_code == 400
    assert "missing values" in run_error.value.detail
    db_session.refresh(job)
    assert job.status == JobStatus.FAILED


def test_original_missing_target_is_rejected_even_if_cleaned_file_filled_it(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "fabricated_target")
    original_path = tmp_path / "original_missing_target.csv"
    cleaned_path = tmp_path / "cleaned_filled_target.csv"
    original_dataframe = valid_classification_frame()
    original_dataframe.loc[0, "target"] = None
    cleaned_dataframe = original_dataframe.copy()
    cleaned_dataframe.loc[0, "target"] = "no"
    original_dataframe.to_csv(original_path, index=False)
    cleaned_dataframe.to_csv(cleaned_path, index=False)
    dataset = add_dataset_with_paths(
        db_session,
        user,
        file_path=str(original_path),
        cleaned_file_path=str(cleaned_path),
    )

    with pytest.raises(HTTPException) as creation_error:
        create_analysis_job(
            db=db_session,
            dataset_id=dataset.id,
            task_type=TaskType.CLASSIFICATION,
            target_column="target",
            config_json={},
            current_user=user,
        )

    assert creation_error.value.status_code == 400
    assert "before cleaning" in creation_error.value.detail

    job = add_analysis_job(db_session, user, dataset)

    with pytest.raises(HTTPException) as run_error:
        run_analysis_job(db_session, job.id, user)

    assert run_error.value.status_code == 400
    assert "before cleaning" in run_error.value.detail


def test_strict_validation_can_be_reused_without_training() -> None:
    x, y, class_distribution = validate_classification_ml_readiness(
        df=valid_classification_frame(),
        target_column="target",
    )

    assert list(x.columns) == ["age", "segment", "score"]
    assert len(y) == 40
    assert class_distribution == {"no": 20, "yes": 20}


def test_duplicate_model_result_for_same_job_is_blocked_by_database(
    db_session,
    tmp_path,
) -> None:
    user = add_user(db_session, "duplicate_result")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_classification_frame(),
    )
    job = add_analysis_job(db_session, user, dataset)
    db_session.add_all(
        [
            ModelResult(
                analysis_id=job.id,
                model_name="first",
                metrics={},
                report_json={},
            ),
            ModelResult(
                analysis_id=job.id,
                model_name="second",
                metrics={},
                report_json={},
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_classification_interpretation_for_good_result_has_no_metric_warnings() -> None:
    interpretation = build_classification_interpretation(
        metrics={
            "accuracy": 0.91,
            "precision": 0.9,
            "recall": 0.89,
            "f1_score": 0.9,
        },
        class_distribution={"no": 80, "yes": 70},
    )

    assert interpretation["quality_level"] == "good"
    assert "good classification performance" in interpretation["summary"]
    assert interpretation["warnings"] == []
    assert interpretation["metric_explanations"]["accuracy"] == (
        "Overall share of correct predictions."
    )


def test_classification_interpretation_warns_for_weak_metrics() -> None:
    interpretation = build_classification_interpretation(
        metrics={
            "accuracy": 0.62,
            "precision": 0.61,
            "recall": 0.55,
            "f1_score": 0.58,
        },
        class_distribution={"no": 80, "yes": 70},
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "weak"
    assert warning_codes == {
        "low_accuracy",
        "low_precision",
        "weak_f1",
        "low_recall",
    }


def test_classification_interpretation_warns_for_low_precision_only() -> None:
    interpretation = build_classification_interpretation(
        metrics={
            "accuracy": 0.82,
            "precision": 0.65,
            "recall": 0.81,
            "f1_score": 0.78,
        },
        class_distribution={"no": 80, "yes": 70},
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "fair"
    assert warning_codes == {"low_precision"}


def test_classification_interpretation_warns_for_class_imbalance() -> None:
    interpretation = build_classification_interpretation(
        metrics={
            "accuracy": 0.88,
            "precision": 0.86,
            "recall": 0.87,
            "f1_score": 0.86,
        },
        class_distribution={"no": 90, "yes": 10},
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "good"
    assert "class_imbalance" in warning_codes


def test_classification_interpretation_warns_for_missing_metrics() -> None:
    interpretation = build_classification_interpretation(
        metrics={
            "accuracy": 0.91,
            "recall": 0.89,
            "f1_score": 0.9,
        },
        class_distribution={"no": 80, "yes": 70},
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "weak"
    assert warning_codes == {"missing_metrics"}
    assert "missing some metrics" in interpretation["summary"]


def test_regression_interpretation_warns_for_weak_r2_and_high_error() -> None:
    interpretation = build_regression_interpretation(
        metrics={
            "mae": 25.0,
            "rmse": 40.0,
            "r2_score": 0.35,
            "target_mean": 100.0,
            "target_min": 50.0,
            "target_max": 150.0,
        },
        row_count=150,
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "weak"
    assert {"weak_r2", "high_error"} <= warning_codes
    assert interpretation["metric_explanations"]["rmse"] == (
        "Typical prediction error, with larger mistakes weighted more heavily."
    )


def test_regression_interpretation_warns_for_small_dataset() -> None:
    interpretation = build_regression_interpretation(
        metrics={
            "mae": 5.0,
            "rmse": 8.0,
            "r2_score": 0.75,
            "target_mean": 100.0,
            "target_min": 50.0,
            "target_max": 150.0,
        },
        row_count=80,
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "good"
    assert warning_codes == {"small_dataset"}


def test_regression_interpretation_warns_for_missing_metrics() -> None:
    interpretation = build_regression_interpretation(
        metrics={
            "mae": 5.0,
            "rmse": 8.0,
            "r2_score": 0.75,
        },
        row_count=150,
    )

    warning_codes = {warning["code"] for warning in interpretation["warnings"]}

    assert interpretation["quality_level"] == "weak"
    assert warning_codes == {"missing_metrics"}
    assert "missing some metrics" in interpretation["summary"]
