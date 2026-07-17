import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parents[1]


def get_required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required")
    return value


def get_int_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")

    return value


def get_upload_dir() -> Path:
    upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))

    if upload_dir.is_absolute():
        return upload_dir

    return BACKEND_DIR / upload_dir


DATABASE_URL = get_required_setting("DATABASE_URL")
SECRET_KEY = get_required_setting("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = get_int_setting("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
UPLOAD_DIR = get_upload_dir()
MAX_UPLOAD_SIZE_MB = get_int_setting("MAX_UPLOAD_SIZE_MB", 50)
MAX_DATAFRAME_MEMORY_MB = get_int_setting("MAX_DATAFRAME_MEMORY_MB", 200)
