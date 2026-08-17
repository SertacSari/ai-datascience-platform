from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class AnalysisReport(Base):
    __tablename__ = "analysis_reports"
    __table_args__ = (
        UniqueConstraint("analysis_id", name="uq_analysis_reports_analysis_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    report_type = Column(String(20), default="html", nullable=False)
    status = Column(String(30), default="ready", nullable=False)
    file_name = Column(String(255), nullable=False)
    html_content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis_job = relationship("AnalysisJob", back_populates="analysis_report")
