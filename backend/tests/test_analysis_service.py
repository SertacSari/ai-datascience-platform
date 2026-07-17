from datetime import datetime

import pandas as pd
import pytest
from fastapi import HTTPException

from app.models.analysis_job import AnalysisJob
from app.models.dataset import Dataset
from app.models.user import User
from app.services.dataset_service import get_owned_dataset
from app.services.analysis_service import (
    get_analysis_job,
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
