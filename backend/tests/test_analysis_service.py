from datetime import datetime

import pandas as pd
import pytest
from fastapi import HTTPException

from app.models.analysis_job import AnalysisJob
from app.models.dataset import Dataset
from app.models.enums import JobStatus
from app.models.model_result import ModelResult
from app.models.user import User
from app.services.dataset_service import get_owned_dataset
from app.services.analysis_service import (
    get_analysis_job,
    get_analysis_job_result,
    list_analysis_jobs,
    validate_forecasting_target,
    validate_target_column,
)


def add_user(db_session, username: str) -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        password_hash="not-used-in-tests",
    )
    db_session.add(user)
    db_session.commit()
    return user


def add_dataset(db_session, user: User, file_path: str = "/unused.csv") -> Dataset:
    dataset = Dataset(
        user_id=user.id,
        file_name="dataset.csv",
        file_path=file_path,
        row_count=2,
        column_count=1,
    )
    db_session.add(dataset)
    db_session.commit()
    return dataset


def test_normalized_numeric_header_can_be_selected() -> None:
    dataframe = pd.DataFrame({"2024": [1, 2]})

    validate_target_column(dataframe, "2024")


def test_target_column_rejects_infinite_values() -> None:
    dataframe = pd.DataFrame({"target": [1.0, float("inf")]})

    with pytest.raises(HTTPException) as error:
        validate_target_column(dataframe, "target")

    assert error.value.status_code == 400


def test_forecasting_requires_string_date_column() -> None:
    dataframe = pd.DataFrame(
        {
            "target": [1.0, 2.0],
            "date": ["2026-01-01", "2026-01-02"],
        }
    )

    with pytest.raises(HTTPException) as error:
        validate_forecasting_target(
            dataframe,
            target_column="target",
            config_json={"date_column": 42},
        )

    assert error.value.status_code == 400


def test_non_owned_analysis_dataset_and_job_return_404(db_session) -> None:
    owner = add_user(db_session, "owner")
    other_user = add_user(db_session, "other")
    dataset = add_dataset(db_session, owner)
    job = AnalysisJob(
        user_id=owner.id,
        dataset_id=dataset.id,
        task_type="classification",
        target_column="target",
    )
    db_session.add(job)
    db_session.commit()

    with pytest.raises(HTTPException) as dataset_error:
        get_owned_dataset(db_session, dataset.id, other_user)
    with pytest.raises(HTTPException) as job_error:
        get_analysis_job(db_session, job.id, other_user)

    assert dataset_error.value.status_code == 404
    assert job_error.value.status_code == 404


def test_job_pagination_has_stable_created_at_and_id_order(db_session) -> None:
    user = add_user(db_session, "owner")
    dataset = add_dataset(db_session, user)
    created_at = datetime(2026, 1, 1)
    jobs = [
        AnalysisJob(
            user_id=user.id,
            dataset_id=dataset.id,
            task_type="classification",
            target_column="target",
            created_at=created_at,
        )
        for _ in range(3)
    ]
    db_session.add_all(jobs)
    db_session.commit()

    first_page = list_analysis_jobs(db_session, user, limit=2, offset=0)
    second_page = list_analysis_jobs(db_session, user, limit=2, offset=2)

    assert [job.id for job in first_page] == [jobs[2].id, jobs[1].id]
    assert [job.id for job in second_page] == [jobs[0].id]


def test_owner_can_fetch_model_result_after_job_completed(db_session) -> None:
    user = add_user(db_session, "result_owner")
    dataset = add_dataset(db_session, user)
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type="classification",
        target_column="target",
        status=JobStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.flush()
    model_result = ModelResult(
        analysis_id=job.id,
        model_name="RandomForestClassifier",
        metrics={"accuracy": 0.9},
        report_json={"class_distribution": {"yes": 10, "no": 10}},
    )
    db_session.add(model_result)
    db_session.commit()

    result = get_analysis_job_result(db_session, job.id, user)

    assert result.id == model_result.id
    assert result.metrics == {"accuracy": 0.9}


def test_non_owner_cannot_fetch_model_result(db_session) -> None:
    owner = add_user(db_session, "result_owner")
    other_user = add_user(db_session, "result_other")
    dataset = add_dataset(db_session, owner)
    job = AnalysisJob(
        user_id=owner.id,
        dataset_id=dataset.id,
        task_type="classification",
        target_column="target",
        status=JobStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(
        ModelResult(
            analysis_id=job.id,
            model_name="RandomForestClassifier",
            metrics={"accuracy": 0.9},
            report_json={},
        )
    )
    db_session.commit()

    with pytest.raises(HTTPException) as error:
        get_analysis_job_result(db_session, job.id, other_user)

    assert error.value.status_code == 404
    assert error.value.detail == "Analysis job not found"


@pytest.mark.parametrize(
    "job_status",
    [JobStatus.COMPLETED, JobStatus.CREATED],
)
def test_job_without_model_result_returns_404(
    db_session,
    job_status: JobStatus,
) -> None:
    user = add_user(db_session, f"no_result_{job_status.value}")
    dataset = add_dataset(db_session, user)
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type="classification",
        target_column="target",
        status=job_status,
    )
    db_session.add(job)
    db_session.commit()

    with pytest.raises(HTTPException) as error:
        get_analysis_job_result(db_session, job.id, user)

    assert error.value.status_code == 404
    assert error.value.detail == "Model result not found for this analysis job"


@pytest.mark.parametrize(
    "job_status",
    [JobStatus.CREATED, JobStatus.RUNNING, JobStatus.FAILED],
)
def test_non_completed_job_result_returns_404_even_if_result_exists(
    db_session,
    job_status: JobStatus,
) -> None:
    user = add_user(db_session, f"unfinished_result_{job_status.value}")
    dataset = add_dataset(db_session, user)
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type="classification",
        target_column="target",
        status=job_status,
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(
        ModelResult(
            analysis_id=job.id,
            model_name="RandomForestClassifier",
            metrics={"accuracy": 0.9},
            report_json={},
        )
    )
    db_session.commit()

    with pytest.raises(HTTPException) as error:
        get_analysis_job_result(db_session, job.id, user)

    assert error.value.status_code == 404
    assert error.value.detail == "Model result not found for this analysis job"
