from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.database import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    task_type = Column(String(30), nullable=False)
    target_column = Column(String(255), nullable=True)
    status = Column(String(30), default="created", nullable=False)
    config_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="analysis_jobs")
    dataset = relationship("Dataset", back_populates="analysis_jobs")
    model_result = relationship("ModelResult", back_populates="analysis_job", uselist=False)
    ai_explanation = relationship("AIExplanation", back_populates="analysis_job", uselist=False)