from datetime import datetime

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
