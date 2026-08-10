from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    text,
)
from sqlalchemy.orm import relationship

from app.database import Base
from app.models.enums import JobStatus, TaskType


DATASET_SOURCE_LABEL_CONFIG_KEY = "_analysis_dataset_source_label"
CLEANED_DATASET_SOURCE_LABEL = "Cleaned dataset"
ORIGINAL_DATASET_SOURCE_LABEL = "Original uploaded file"


def enum_values(enum_class):
    return [member.value for member in enum_class]


def get_current_dataset_source_label(dataset) -> str:
    if dataset is not None and dataset.cleaned_file_path:
        return CLEANED_DATASET_SOURCE_LABEL

    return ORIGINAL_DATASET_SOURCE_LABEL


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        CheckConstraint(
            "task_type IN ('classification', 'regression', 'forecasting')",
            name="ck_analysis_jobs_task_type",
        ),
        CheckConstraint(
            "status IN ('created', 'running', 'completed', 'failed')",
            name="ck_analysis_jobs_status",
        ),
        Index("ix_analysis_jobs_user_created_id", "user_id", "created_at", "id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    task_type = Column(
        Enum(
            TaskType,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=False,
            length=30,
        ),
        nullable=False,
    )
    target_column = Column(String(255), nullable=False)
    status = Column(
        Enum(
            JobStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=False,
            length=30,
        ),
        default=JobStatus.CREATED,
        nullable=False,
    )
    config_json = Column(
        JSON,
        default=dict,
        server_default=text("'{}'"),
        nullable=False,
    )

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="analysis_jobs")
    dataset = relationship("Dataset", back_populates="analysis_jobs")
    model_result = relationship(
        "ModelResult",
        back_populates="analysis_job",
        uselist=False,
    )
    ai_explanation = relationship(
        "AIExplanation",
        back_populates="analysis_job",
        uselist=False,
    )

    @property
    def dataset_file_name(self) -> str | None:
        if self.dataset is None:
            return None

        return self.dataset.file_name

    @property
    def dataset_source_label(self) -> str:
        config_json = self.config_json if isinstance(self.config_json, dict) else {}
        source_label = config_json.get(DATASET_SOURCE_LABEL_CONFIG_KEY)

        if source_label in {
            CLEANED_DATASET_SOURCE_LABEL,
            ORIGINAL_DATASET_SOURCE_LABEL,
        }:
            return source_label

        return get_current_dataset_source_label(self.dataset)
