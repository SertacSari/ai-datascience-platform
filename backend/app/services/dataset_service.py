import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.user import User


UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"


def upload_dataset(db: Session, file: UploadFile, current_user: User) -> Dataset:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded",
        )

    allowed_extensions = {".csv", ".xlsx", ".xls"}

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and Excel files are allowed",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid4()}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception:
            file_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file could not be read as a valid CSV or Excel file",
            )

        row_count, column_count = df.shape

        new_dataset = Dataset(
            user_id=current_user.id,
            file_name=file.filename,
            file_path=str(file_path),
            row_count=row_count,
            column_count=column_count,
        )

        try:
            db.add(new_dataset)
            db.commit()
            db.refresh(new_dataset)
        except Exception:
            db.rollback()
            file_path.unlink(missing_ok=True)
            raise

        return new_dataset

    finally:
        file.file.close()
