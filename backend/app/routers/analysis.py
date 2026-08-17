from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.analysis_job import AnalysisJob
from app.models.user import User
from app.schemas.analysis import (
    AIExplanationResponse,
    AnalysisJobCreate,
    AnalysisJobResponse,
    AnalysisJobRunResponse,
    AnalysisReportResponse,
    ModelResultResponse,
)
from app.services.ai_service import create_ai_explanation, get_ai_explanation
from app.services.analysis_service import (
    create_analysis_job,
    get_analysis_job,
    get_analysis_job_result,
    list_analysis_jobs,
)
from app.services.model_training_service import run_analysis_job
from app.services.report_service import (
    create_analysis_report,
    get_analysis_report,
    get_analysis_report_html,
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


@router.post(
    "/jobs/{job_id}/run",
    response_model=AnalysisJobRunResponse,
)
def run_analysis_job_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisJobRunResponse:
    analysis_job, model_result = run_analysis_job(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )

    return AnalysisJobRunResponse(
        job=analysis_job,
        model_result=model_result,
    )


@router.get(
    "/jobs/{job_id}/result",
    response_model=ModelResultResponse,
)
def get_analysis_job_result_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_analysis_job_result(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )


@router.post(
    "/jobs/{job_id}/ai-explanation",
    response_model=AIExplanationResponse,
)
def create_ai_explanation_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AIExplanationResponse:
    return create_ai_explanation(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )


@router.get(
    "/jobs/{job_id}/ai-explanation",
    response_model=AIExplanationResponse,
)
def get_ai_explanation_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AIExplanationResponse:
    return get_ai_explanation(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )


@router.post(
    "/jobs/{job_id}/report",
    response_model=AnalysisReportResponse,
)
def create_analysis_report_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisReportResponse:
    return create_analysis_report(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )


@router.get(
    "/jobs/{job_id}/report",
    response_model=AnalysisReportResponse,
)
def get_analysis_report_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisReportResponse:
    return get_analysis_report(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )


@router.get("/jobs/{job_id}/report/download")
def download_analysis_report_endpoint(
    job_id: Annotated[int, Path(gt=0)],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    report = get_analysis_report_html(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )

    return Response(
        content=report.html_content,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{report.file_name}"',
        },
    )
