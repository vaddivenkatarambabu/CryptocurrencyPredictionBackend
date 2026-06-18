import pytest

from errors import ApiError
from validation import validate_prediction_request


def test_validate_prediction_request_normalizes_coin_id():
    result = validate_prediction_request({"coinId": " BitCoin ", "days": "7"}, 365)

    assert result.coin_id == "bitcoin"
    assert result.days == 7


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (None, "invalid_json"),
        ({}, "invalid_coin_id"),
        ({"coinId": ""}, "invalid_coin_id"),
        ({"coinId": "../bitcoin"}, "invalid_coin_id_format"),
        ({"coinId": "bitcoin/usd"}, "invalid_coin_id_format"),
        ({"coinId": "bitcoin", "days": "abc"}, "invalid_days"),
        ({"coinId": "bitcoin", "days": "1.5"}, "invalid_days"),
        ({"coinId": "bitcoin", "days": 1.5}, "invalid_days"),
        ({"coinId": "bitcoin", "days": True}, "invalid_days"),
        ({"coinId": "bitcoin", "days": 0}, "invalid_days_range"),
        ({"coinId": "bitcoin", "days": 366}, "invalid_days_range"),
    ],
)
def test_validate_prediction_request_rejects_invalid_payloads(payload, code):
    with pytest.raises(ApiError) as exc:
        validate_prediction_request(payload, 365)

    assert exc.value.code == code
