import json
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.user import User


UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"


def read_dataset_file(file_path: str) -> pd.DataFrame:
    file_extension = Path(file_path).suffix.lower()

    try:
        if file_extension == ".csv":
            return pd.read_csv(file_path)

        if file_extension in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file could not be read",
        ) from exc


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


def get_dataset_preview(
    db: Session,
    dataset_id: int,
    current_user: User,
) -> dict:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    if dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this dataset",
        )

    if not Path(dataset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )

    df = read_dataset_file(dataset.file_path)

    preview_df = df.head(10)

    column_info = []
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = (
            round((missing_count / len(df)) * 100, 2) if len(df) > 0 else 0
        )

        column_info.append(
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            }
        )

    duplicate_rows = int(df.duplicated().sum())

    preview = json.loads(preview_df.to_json(orient="records", date_format="iso"))
    summary_statistics = json.loads(
        df.describe(include="all").to_json(date_format="iso")
    )

    return {
        "dataset_id": dataset.id,
        "file_name": dataset.file_name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [str(column) for column in df.columns],
        "preview": preview,
        "column_info": column_info,
        "duplicate_rows": duplicate_rows,
        "summary_statistics": summary_statistics,
    }
