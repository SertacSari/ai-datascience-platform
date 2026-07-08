from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.models.dataset import Dataset
from app.models.user import User
from app.services import dataset_service
from app.services.dataset_service import (
    DatasetReadError,
    clean_dataset,
    get_cleaning_report,
    get_dataset_preview,
    read_dataset_file,
    read_stored_dataset_file,
    upload_dataset,
)


def add_user(db_session, username: str, user_id: int | None = None) -> User:
    user = User(
        id=user_id,
        username=username,
        email=f"{username}@example.com",
        password_hash="not-used-in-tests",
    )
    db_session.add(user)
    db_session.commit()
    return user


def add_dataset(db_session, user: User, file_path: str) -> Dataset:
    dataset = Dataset(
        user_id=user.id,
        file_name="dataset.csv",
        file_path=file_path,
        row_count=2,
        column_count=1,
    )
    db_session.add(dataset)
    db_session.commit()
    return dataset


def test_actual_csv_duplicate_headers_are_rejected(tmp_path) -> None:
    dataset_path = tmp_path / "duplicate.csv"
    dataset_path.write_text("target,target\n1,2\n", encoding="utf-8")

    with pytest.raises(DatasetReadError, match="duplicate or ambiguous"):
        read_dataset_file(str(dataset_path))


@pytest.mark.parametrize("leading_blank_lines", [1, 3])
def test_actual_csv_duplicate_headers_after_leading_blank_lines_are_rejected(
    tmp_path,
    leading_blank_lines: int,
) -> None:
    dataset_path = tmp_path / "leading-blank-duplicate.csv"
    dataset_path.write_text(
        "\n" * leading_blank_lines + "target,target\n1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetReadError, match="duplicate or ambiguous"):
        read_dataset_file(str(dataset_path))


def test_valid_csv_after_leading_blank_lines_is_accepted(tmp_path) -> None:
    dataset_path = tmp_path / "leading-blank-valid.csv"
    dataset_path.write_text("\n\nfeature,target\n1,2\n", encoding="utf-8")

    dataframe = read_dataset_file(str(dataset_path))

    assert list(dataframe.columns) == ["feature", "target"]
    assert dataframe.to_dict(orient="records") == [{"feature": 1, "target": 2}]


def test_empty_csv_is_rejected(tmp_path) -> None:
    dataset_path = tmp_path / "empty.csv"
    dataset_path.write_bytes(b"")

    with pytest.raises(DatasetReadError, match="could not be read"):
        read_dataset_file(str(dataset_path))


def test_header_only_csv_is_accepted(tmp_path) -> None:
    dataset_path = tmp_path / "header-only.csv"
    dataset_path.write_text("feature,target\n", encoding="utf-8")

    dataframe = read_dataset_file(str(dataset_path))

    assert list(dataframe.columns) == ["feature", "target"]
    assert dataframe.empty


def test_actual_xlsx_integer_string_header_collision_is_rejected(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    dataset_path = tmp_path / "collision.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append([1, "1"])
    worksheet.append([10, 20])
    workbook.save(dataset_path)

    with pytest.raises(DatasetReadError, match="duplicate or ambiguous"):
        read_dataset_file(str(dataset_path))


def test_actual_xls_integer_string_header_collision_is_rejected(tmp_path) -> None:
    xlwt = pytest.importorskip("xlwt")
    pytest.importorskip("xlrd")
    dataset_path = tmp_path / "collision.xls"
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet("Sheet1")
    worksheet.write(0, 0, 1)
    worksheet.write(0, 1, "1")
    worksheet.write(1, 0, 10)
    worksheet.write(1, 1, 20)
    workbook.save(str(dataset_path))

    with pytest.raises(DatasetReadError, match="ambiguous column names"):
        read_dataset_file(str(dataset_path))


def test_valid_numeric_excel_header_becomes_string(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    dataset_path = tmp_path / "numeric-header.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append([2024, "target"])
    worksheet.append([1, 2])
    workbook.save(dataset_path)

    dataframe = read_dataset_file(str(dataset_path))

    assert list(dataframe.columns) == ["2024", "target"]


def test_upload_parse_failure_returns_400(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_service, "UPLOAD_DIR", tmp_path)
    upload = UploadFile(filename="invalid.csv", file=BytesIO(b""))

    with pytest.raises(HTTPException) as error:
        upload_dataset(None, upload, SimpleNamespace(id=1))

    assert error.value.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_upload_memory_limit_failure_returns_413(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(dataset_service, "MAX_DATAFRAME_MEMORY_BYTES", 1)
    upload = UploadFile(filename="large.csv", file=BytesIO(b"value\ncontent\n"))

    with pytest.raises(HTTPException) as error:
        upload_dataset(None, upload, SimpleNamespace(id=1))

    assert error.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["parse", "memory"])
def test_stored_read_failures_return_500(
    tmp_path,
    monkeypatch,
    failure_kind: str,
) -> None:
    dataset_path = tmp_path / "stored.csv"
    if failure_kind == "parse":
        dataset_path.write_bytes(b"")
    else:
        dataset_path.write_text("value\ncontent\n", encoding="utf-8")
        monkeypatch.setattr(dataset_service, "MAX_DATAFRAME_MEMORY_BYTES", 1)

    with pytest.raises(HTTPException) as error:
        read_stored_dataset_file(str(dataset_path))

    assert error.value.status_code == 500
    assert error.value.detail == "Stored dataset could not be read"


def test_missing_owned_stored_file_returns_404(db_session, tmp_path) -> None:
    user = add_user(db_session, "owner")
    dataset = add_dataset(db_session, user, str(tmp_path / "missing.csv"))

    with pytest.raises(HTTPException) as error:
        get_dataset_preview(db_session, dataset.id, user)

    assert error.value.status_code == 404
    assert error.value.detail == "Dataset file not found on server"


@pytest.mark.parametrize(
    "operation",
    [get_dataset_preview, get_cleaning_report, clean_dataset],
)
def test_non_owned_dataset_operations_return_404(
    db_session,
    tmp_path,
    operation,
) -> None:
    owner = add_user(db_session, "owner")
    other_user = add_user(db_session, "other")
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("target\n1\n2\n", encoding="utf-8")
    dataset = add_dataset(db_session, owner, str(dataset_path))

    with pytest.raises(HTTPException) as error:
        operation(db_session, dataset.id, other_user)

    assert error.value.status_code == 404
    assert error.value.detail == "Dataset not found"


def test_csv_concatenation_limit_accounts_for_peak_memory(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("value\n" + "abcdefghij\n" * 20, encoding="utf-8")
    monkeypatch.setattr(dataset_service, "CSV_CHUNK_SIZE_ROWS", 10)

    first_chunk = next(
        dataset_service.pd.read_csv(dataset_path, chunksize=10)
    )
    retained_memory = dataset_service.get_dataframe_memory_usage(first_chunk) * 2
    monkeypatch.setattr(
        dataset_service,
        "MAX_DATAFRAME_MEMORY_BYTES",
        retained_memory + 1,
    )

    with pytest.raises(dataset_service.DatasetMemoryLimitError):
        dataset_service.read_csv_with_memory_limit(str(dataset_path))
