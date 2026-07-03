from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    cleaned_file_path = Column(String(500), nullable=True)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="datasets")
    analysis_jobs = relationship("AnalysisJob", back_populates="dataset")