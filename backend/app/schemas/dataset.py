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
