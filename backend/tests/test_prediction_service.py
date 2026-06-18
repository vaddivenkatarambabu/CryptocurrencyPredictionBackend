import numpy as np
import pytest

from errors import ApiError
from services.prediction import PredictionService


class FakeCoinGeckoClient:
    def __init__(self, prices):
        self.prices = prices

    def fetch_daily_prices(self, coin_id, days):
        return self.prices


class FakeModel:
    input_shape = (None, 3, 1)

    def predict(self, window, verbose=0):
        return np.array([[window[0, -1, 0]]])


class BrokenModel:
    input_shape = (None, 3, 1)

    def predict(self, window, verbose=0):
        raise ValueError("model exploded")


class NegativeModel:
    input_shape = (None, 3, 1)

    def predict(self, window, verbose=0):
        return np.array([[-10]])


def test_prediction_service_generates_expected_response_shape():
    service = PredictionService(
        model=FakeModel(),
        coingecko_client=FakeCoinGeckoClient([100, 110, 120, 130]),
        history_days=365,
        max_prediction_days=365,
    )

    result = service.predict("bitcoin", 2).to_dict()

    assert result["coinId"] == "bitcoin"
    assert result["days"] == 2
    assert len(result["prediction"]["dailyPrices"]) == 2
    assert result["prediction"]["trend"] in {"bullish", "bearish", "neutral"}


def test_prediction_service_supports_365_day_forecast():
    service = PredictionService(
        model=FakeModel(),
        coingecko_client=FakeCoinGeckoClient([100, 110, 120, 130]),
        history_days=365,
        max_prediction_days=365,
    )

    result = service.predict("bitcoin", 365).to_dict()

    assert result["days"] == 365
    assert len(result["prediction"]["dailyPrices"]) == 365


def test_prediction_service_clips_out_of_range_model_predictions():
    service = PredictionService(
        model=NegativeModel(),
        coingecko_client=FakeCoinGeckoClient([100, 110, 120]),
        history_days=365,
        max_prediction_days=365,
    )

    result = service.predict("bitcoin", 1).to_dict()

    assert result["prediction"]["predictedPrice"] > 0


def test_prediction_service_requires_enough_history():
    service = PredictionService(
        model=FakeModel(),
        coingecko_client=FakeCoinGeckoClient([100, 110]),
        history_days=365,
        max_prediction_days=365,
    )

    with pytest.raises(ApiError) as exc:
        service.predict("bitcoin", 1)

    assert exc.value.code == "insufficient_market_history"


def test_prediction_service_rejects_invalid_market_prices():
    service = PredictionService(
        model=FakeModel(),
        coingecko_client=FakeCoinGeckoClient([100, 0, 120]),
        history_days=365,
        max_prediction_days=365,
    )

    with pytest.raises(ApiError) as exc:
        service.predict("bitcoin", 1)

    assert exc.value.code == "invalid_market_data_price"


def test_prediction_service_wraps_model_failures():
    service = PredictionService(
        model=BrokenModel(),
        coingecko_client=FakeCoinGeckoClient([100, 110, 120]),
        history_days=365,
        max_prediction_days=365,
    )

    with pytest.raises(ApiError) as exc:
        service.predict("bitcoin", 1)

    assert exc.value.code == "prediction_failed"
