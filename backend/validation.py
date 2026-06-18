from dataclasses import dataclass
import re

from errors import ApiError

COIN_ID_PATTERN = re.compile(r"^[a-z0-9-]+$")


@dataclass(frozen=True)
class PredictionRequest:
    coin_id: str
    days: int


def validate_prediction_request(payload, max_days):
    if not isinstance(payload, dict):
        raise ApiError(
            message="Request body must be a JSON object.",
            status_code=400,
            code="invalid_json",
        )

    coin_id = payload.get("coinId")
    if not isinstance(coin_id, str) or not coin_id.strip():
        raise ApiError(
            message="coinId is required.",
            status_code=400,
            code="invalid_coin_id",
        )

    normalized_coin_id = coin_id.strip().lower()
    if not COIN_ID_PATTERN.fullmatch(normalized_coin_id):
        raise ApiError(
            message="coinId may only contain lowercase letters, numbers, and hyphens.",
            status_code=400,
            code="invalid_coin_id_format",
        )

    raw_days = payload.get("days", 7)
    if isinstance(raw_days, bool):
        raise ApiError(
            message="days must be an integer.",
            status_code=400,
            code="invalid_days",
        )

    if isinstance(raw_days, int):
        days = raw_days
    elif isinstance(raw_days, str) and raw_days.strip().isdigit():
        days = int(raw_days.strip())
    else:
        raise ApiError(
            message="days must be an integer.",
            status_code=400,
            code="invalid_days",
        )

    if days < 1 or days > max_days:
        raise ApiError(
            message=f"days must be between 1 and {max_days}.",
            status_code=400,
            code="invalid_days_range",
        )

    return PredictionRequest(coin_id=normalized_coin_id, days=days)
