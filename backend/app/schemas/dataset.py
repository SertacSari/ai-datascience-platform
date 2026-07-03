from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DatasetResponse(BaseModel):
    id: int
    user_id: int
    file_name: str
    file_path: str
    row_count: int
    column_count: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float


class DatasetPreviewResponse(BaseModel):
    dataset_id: int
    file_name: str
    row_count: int
    column_count: int
    columns: list[str]
    preview: list[dict[str, Any]]
    column_info: list[ColumnInfo]
    duplicate_rows: int
    summary_statistics: dict[str, Any]


class CleaningIssue(BaseModel):
    column: str | None = None
    issue_type: str
    details: str
    recommended_action: str


class CleaningReportResponse(BaseModel):
    dataset_id: int
    file_name: str
    row_count: int
    column_count: int
    column_types: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    issues: list[CleaningIssue]
    ready_for_ml: bool


class CleanDatasetResponse(BaseModel):
    dataset_id: int
    original_file_path: str
    cleaned_file_path: str
    original_row_count: int
    cleaned_row_count: int
    removed_duplicate_rows: int
    message: str
