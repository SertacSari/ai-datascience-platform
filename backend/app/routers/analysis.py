from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.analysis_job import AnalysisJob
from app.models.user import User
from app.schemas.analysis import AnalysisJobCreate, AnalysisJobResponse
from app.services.analysis_service import (
    create_analysis_job,
    get_analysis_job,
    list_analysis_jobs,
)


router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


@router.get("/health")
def analysis_health_check() -> dict[str, str]:
    return {"message": "Analysis router is working"}


@router.post(
    "/jobs",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_analysis_job_endpoint(
    job_data: AnalysisJobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisJob:
    return create_analysis_job(
        db=db,
        dataset_id=job_data.dataset_id,
        task_type=job_data.task_type,
        target_column=job_data.target_column,
        config_json=job_data.config_json,
        current_user=current_user,
    )


@router.get("/jobs", response_model=list[AnalysisJobResponse])
def list_analysis_jobs_endpoint(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AnalysisJob]:
    return list_analysis_jobs(
        db=db,
        current_user=current_user,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisJob:
    return get_analysis_job(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )
