from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.analysis_job import AnalysisJob
from app.models.dataset import Dataset
from app.models.enums import JobStatus, TaskType
from app.models.user import User
from app.services.dataset_service import get_owned_dataset, read_stored_dataset_file


VALID_TASK_TYPES = {task_type.value for task_type in TaskType}


def get_dataset_path_for_analysis(dataset: Dataset) -> str:
    if dataset.cleaned_file_path:
        return dataset.cleaned_file_path

    return dataset.file_path


def validate_task_type(task_type: str) -> None:
    if task_type not in VALID_TASK_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task type",
        )


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target column does not exist in dataset",
        )

    if df[target_column].isna().any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target column must not contain missing values",
        )

    if pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].isin(
        [float("inf"), float("-inf")]
    ).any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target column must contain only finite values",
        )


def validate_minimum_rows(df: pd.DataFrame) -> None:
    if len(df) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain at least 2 rows for analysis",
        )


def validate_classification_target(df: pd.DataFrame, target_column: str) -> None:
    unique_count = int(df[target_column].nunique())

    if unique_count < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Classification target must have at least 2 classes",
        )


def validate_regression_target(df: pd.DataFrame, target_column: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Regression target column must be numerical",
        )


def validate_forecasting_target(
    df: pd.DataFrame,
    target_column: str,
    config_json: dict[str, Any],
) -> None:
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting target column must be numerical",
        )

    date_column = (config_json or {}).get("date_column")

    if not isinstance(date_column, str) or not date_column.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting requires config_json.date_column",
        )

    if date_column == target_column:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting date column must be different from target column",
        )

    if date_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting date column does not exist in dataset",
        )

    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

    if parsed_dates.isna().any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting date column contains invalid or missing dates",
        )

    if parsed_dates.duplicated().any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecasting date column must contain unique values",
        )


def validate_task_specific_rules(
    df: pd.DataFrame,
    task_type: TaskType,
    target_column: str,
    config_json: dict[str, Any],
) -> None:
    if task_type == "classification":
        validate_classification_target(df, target_column)

    elif task_type == "regression":
        validate_regression_target(df, target_column)

    elif task_type == "forecasting":
        validate_forecasting_target(df, target_column, config_json)


def validate_analysis_request(
    db: Session,
    dataset_id: int,
    task_type: TaskType,
    target_column: str,
    config_json: dict[str, Any],
    current_user: User,
) -> Dataset:
    validate_task_type(task_type)

    dataset = get_owned_dataset(
        db=db,
        dataset_id=dataset_id,
        current_user=current_user,
    )

    dataset_path = get_dataset_path_for_analysis(dataset)

    if not Path(dataset_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )

    df = read_stored_dataset_file(dataset_path)

    validate_target_column(df, target_column)
    validate_minimum_rows(df)

    validate_task_specific_rules(
        df=df,
        task_type=task_type,
        target_column=target_column,
        config_json=config_json,
    )

    return dataset


def create_analysis_job(
    db: Session,
    dataset_id: int,
    task_type: TaskType,
    target_column: str,
    config_json: dict[str, Any],
    current_user: User,
) -> AnalysisJob:
    dataset = validate_analysis_request(
        db=db,
        dataset_id=dataset_id,
        task_type=task_type,
        target_column=target_column,
        config_json=config_json,
        current_user=current_user,
    )

    analysis_job = AnalysisJob(
        user_id=current_user.id,
        dataset_id=dataset.id,
        task_type=task_type,
        target_column=target_column,
        status=JobStatus.CREATED,
        config_json=config_json,
    )

    try:
        db.add(analysis_job)
        db.commit()
        db.refresh(analysis_job)
    except Exception:
        db.rollback()
        raise

    return analysis_job


def list_analysis_jobs(
    db: Session,
    current_user: User,
    limit: int,
    offset: int,
) -> list[AnalysisJob]:
    return (
        db.query(AnalysisJob)
        .filter(AnalysisJob.user_id == current_user.id)
        .order_by(AnalysisJob.created_at.desc(), AnalysisJob.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_analysis_job(
    db: Session,
    job_id: int,
    current_user: User,
) -> AnalysisJob:
    job = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found",
        )

    return job
