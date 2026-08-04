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
from app.services.analysis_service import create_analysis_job
from app.services.classification_training_service import (
    run_analysis_job,
    validate_classification_ml_readiness,
)


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
) -> AnalysisJob:
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type=task_type,
        target_column=target_column,
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


@pytest.mark.parametrize(
    "task_type",
    [TaskType.REGRESSION, TaskType.FORECASTING],
)
def test_regression_and_forecasting_jobs_cannot_run_yet(
    db_session,
    tmp_path,
    task_type: TaskType,
) -> None:
    user = add_user(db_session, f"user_{task_type.value}")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        user,
        valid_classification_frame(),
    )
    job = add_analysis_job(db_session, user, dataset, task_type=task_type)

    with pytest.raises(HTTPException) as error:
        run_analysis_job(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "Only classification jobs" in error.value.detail


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
