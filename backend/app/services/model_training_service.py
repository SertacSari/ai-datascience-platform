from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.analysis_job import AnalysisJob
from app.models.enums import JobStatus, TaskType
from app.models.model_result import ModelResult
from app.models.user import User
from app.services.analysis_service import get_analysis_job, get_dataset_path_for_analysis
from app.services.classification_training_service import (
    build_model_result_payload,
    validate_original_target_was_not_fabricated,
)
from app.services.dataset_service import read_stored_dataset_file
from app.services.forecasting_training_service import build_forecasting_model_result_payload
from app.services.regression_training_service import build_regression_model_result_payload


def get_enum_value(value: str | JobStatus | TaskType) -> str:
    if isinstance(value, (JobStatus, TaskType)):
        return value.value

    return value


def save_model_result(
    db: Session,
    analysis_job: AnalysisJob,
    model_name: str,
    metrics: dict,
    report_json: dict,
) -> ModelResult:
    model_result = (
        db.query(ModelResult)
        .filter(ModelResult.analysis_id == analysis_job.id)
        .first()
    )

    if model_result is None:
        model_result = ModelResult(analysis_id=analysis_job.id)
        db.add(model_result)

    model_result.model_name = model_name
    model_result.metrics = metrics
    model_result.report_json = report_json

    return model_result


def mark_job_status(
    db: Session,
    analysis_job: AnalysisJob,
    status_value: JobStatus,
) -> None:
    analysis_job.status = status_value
    if status_value in {JobStatus.COMPLETED, JobStatus.FAILED}:
        analysis_job.finished_at = datetime.utcnow()
    db.commit()
    db.refresh(analysis_job)


def run_analysis_job(
    db: Session,
    job_id: int,
    current_user: User,
) -> tuple[AnalysisJob, ModelResult]:
    analysis_job = get_analysis_job(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )

    task_type = get_enum_value(analysis_job.task_type)

    supported_task_types = {
        TaskType.CLASSIFICATION.value,
        TaskType.REGRESSION.value,
        TaskType.FORECASTING.value,
    }

    if task_type not in supported_task_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only classification, regression, and forecasting jobs can be run in this phase",
        )

    current_status = get_enum_value(analysis_job.status)

    if current_status != JobStatus.CREATED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only created jobs can be run",
        )

    mark_job_status(db, analysis_job, JobStatus.RUNNING)

    try:
        dataset_path = get_dataset_path_for_analysis(analysis_job.dataset)

        if not Path(dataset_path).is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset file not found on server",
            )

        df = read_stored_dataset_file(dataset_path)
        if task_type == TaskType.CLASSIFICATION.value:
            validate_original_target_was_not_fabricated(
                dataset=analysis_job.dataset,
                target_column=analysis_job.target_column,
            )
            model_name, metrics, report_json = build_model_result_payload(
                df=df,
                target_column=analysis_job.target_column,
            )
        else:
            if task_type == TaskType.REGRESSION.value:
                model_name, metrics, report_json = build_regression_model_result_payload(
                    df=df,
                    target_column=analysis_job.target_column,
                )
            else:
                model_name, metrics, report_json = build_forecasting_model_result_payload(
                    df=df,
                    target_column=analysis_job.target_column,
                    config_json=analysis_job.config_json,
                )
        model_result = save_model_result(
            db=db,
            analysis_job=analysis_job,
            model_name=model_name,
            metrics=metrics,
            report_json=report_json,
        )
        analysis_job.status = JobStatus.COMPLETED
        analysis_job.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(analysis_job)
        db.refresh(model_result)

        return analysis_job, model_result

    except HTTPException:
        db.rollback()
        mark_job_status(db, analysis_job, JobStatus.FAILED)
        raise
    except Exception as exc:
        db.rollback()
        mark_job_status(db, analysis_job, JobStatus.FAILED)
        task_name = {
            TaskType.CLASSIFICATION.value: "Classification",
            TaskType.REGRESSION.value: "Regression",
            TaskType.FORECASTING.value: "Forecasting",
        }.get(task_type, "Model")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{task_name} training failed",
        ) from exc
