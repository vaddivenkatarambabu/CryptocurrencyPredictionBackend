import os
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _csv_env(name, default):
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _path_env(name, default):
    path = Path(os.getenv(name, default))
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))

    TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

    MODEL_PATH = _path_env("MODEL_PATH", "model.keras")

    COINGECKO_BASE_URL = os.getenv(
        "COINGECKO_BASE_URL",
        "https://api.coingecko.com/api/v3",
    )
    COINGECKO_TIMEOUT_SECONDS = float(os.getenv("COINGECKO_TIMEOUT_SECONDS", "10"))

    HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "365"))
    MAX_PREDICTION_DAYS = int(os.getenv("MAX_PREDICTION_DAYS", "365"))

    CORS_ORIGINS = _csv_env(
        "CORS_ORIGINS",
        "http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:5173,http://localhost:5173",
    )

    RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "200 per hour")
    RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "30 per minute")
    RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

    REQUEST_ID_FACTORY = staticmethod(lambda: str(uuid.uuid4()))
