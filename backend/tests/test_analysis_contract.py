import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.main import app
from app.models.analysis_job import AnalysisJob
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.analysis import AnalysisJobCreate


def add_job_dependencies(db_session) -> tuple[User, Dataset]:
    user = User(
        username="owner",
        email="owner@example.com",
        password_hash="not-used-in-tests",
    )
    db_session.add(user)
    db_session.flush()
    dataset = Dataset(
        user_id=user.id,
        file_name="dataset.csv",
        file_path="/unused.csv",
        row_count=2,
        column_count=1,
    )
    db_session.add(dataset)
    db_session.commit()
    return user, dataset


def test_analysis_schema_default_is_not_shared() -> None:
    first = AnalysisJobCreate(
        dataset_id=1,
        task_type="classification",
        target_column="target",
    )
    second = AnalysisJobCreate(
        dataset_id=1,
        task_type="classification",
        target_column="target",
    )

    first.config_json["changed"] = True

    assert second.config_json == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_id", 0),
        ("task_type", "clustering"),
        ("target_column", ""),
    ],
)
def test_analysis_schema_rejects_invalid_input(field: str, value) -> None:
    payload = {
        "dataset_id": 1,
        "task_type": "classification",
        "target_column": "target",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        AnalysisJobCreate.model_validate(payload)


def test_openapi_declares_pagination_and_positive_job_id_bounds() -> None:
    schema = app.openapi()
    list_parameters = {
        parameter["name"]: parameter["schema"]
        for parameter in schema["paths"]["/analysis/jobs"]["get"]["parameters"]
    }
    job_parameters = {
        parameter["name"]: parameter["schema"]
        for parameter in schema["paths"]["/analysis/jobs/{job_id}"]["get"][
            "parameters"
        ]
    }

    assert list_parameters["limit"]["minimum"] == 1
    assert list_parameters["limit"]["maximum"] == 100
    assert list_parameters["offset"]["minimum"] == 0
    assert job_parameters["job_id"]["exclusiveMinimum"] == 0


def test_analysis_job_python_and_server_config_defaults(db_session) -> None:
    user, dataset = add_job_dependencies(db_session)
    orm_job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type="classification",
        target_column="target",
    )
    db_session.add(orm_job)
    db_session.commit()

    assert orm_job.config_json == {}

    result = db_session.execute(
        text(
            """
            INSERT INTO analysis_jobs
                (user_id, dataset_id, task_type, target_column, status, created_at)
            VALUES
                (:user_id, :dataset_id, 'classification', 'target', 'created', CURRENT_TIMESTAMP)
            RETURNING config_json
            """
        ),
        {"user_id": user.id, "dataset_id": dataset.id},
    ).scalar_one()

    assert result == "{}"


@pytest.mark.parametrize(
    "task_type,target_column,status",
    [
        ("clustering", "target", "created"),
        ("classification", "target", "unknown"),
        ("classification", None, "created"),
    ],
)
def test_analysis_job_database_constraints(
    db_session,
    task_type: str,
    target_column: str | None,
    status: str,
) -> None:
    user, dataset = add_job_dependencies(db_session)
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type=task_type,
        target_column=target_column,
        status=status,
    )
    db_session.add(job)

    with pytest.raises(IntegrityError):
        db_session.commit()
