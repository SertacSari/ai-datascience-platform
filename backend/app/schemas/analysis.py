from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal["classification", "regression", "forecasting"]
AnalysisStatus = Literal["created", "running", "completed", "failed"]


class AnalysisJobCreate(BaseModel):
    dataset_id: int = Field(gt=0)
    task_type: TaskType
    target_column: str = Field(min_length=1, max_length=255)
    config_json: dict[str, Any] = Field(default_factory=dict)


class AnalysisJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    dataset_id: int
    task_type: TaskType
    target_column: str
    status: AnalysisStatus
    config_json: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None
