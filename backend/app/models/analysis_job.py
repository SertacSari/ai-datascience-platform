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


def enum_values(enum_class):
    return [member.value for member in enum_class]


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
