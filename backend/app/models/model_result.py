from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.database import Base


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)

    model_name = Column(String(100), nullable=False)
    metrics = Column(JSON, nullable=True)
    report_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis_job = relationship("AnalysisJob", back_populates="model_result")