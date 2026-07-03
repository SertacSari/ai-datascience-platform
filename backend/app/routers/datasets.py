from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.dataset import (
    CleaningReportResponse,
    CleanDatasetResponse,
    DatasetPreviewResponse,
    DatasetResponse,
)
from app.services.dataset_service import (
    clean_dataset,
    get_cleaning_report,
    get_dataset_preview,
    upload_dataset,
)

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


@router.get("/health")
def datasets_health_check():
    return {"message": "Datasets router is working"}


@router.post("/upload", response_model=DatasetResponse)
def upload_dataset_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return upload_dataset(
        db=db,
        file=file,
        current_user=current_user,
    )


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset_endpoint(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_dataset_preview(
        db=db,
        dataset_id=dataset_id,
        current_user=current_user,
    )


@router.get("/{dataset_id}/cleaning-report", response_model=CleaningReportResponse)
def cleaning_report_endpoint(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_cleaning_report(
        db=db,
        dataset_id=dataset_id,
        current_user=current_user,
    )


@router.post("/{dataset_id}/clean", response_model=CleanDatasetResponse)
def clean_dataset_endpoint(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return clean_dataset(
        db=db,
        dataset_id=dataset_id,
        current_user=current_user,
    )
