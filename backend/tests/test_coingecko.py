import pytest
import requests

from errors import ApiError
from services.coingecko import CoinGeckoClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=None):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error
        self.ok = 200 <= status_code < 300

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def get(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.response


def test_fetch_daily_prices_returns_numeric_prices():
    client = CoinGeckoClient("https://example.test", 1)
    client.session = FakeSession(FakeResponse(payload={"prices": [[1, "100.5"]]}))

    assert client.fetch_daily_prices("bitcoin", 365) == [100.5]


def test_fetch_market_coins_returns_coin_list():
    payload = [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}]
    client = CoinGeckoClient("https://example.test", 1)
    client.session = FakeSession(FakeResponse(payload=payload))

    assert client.fetch_market_coins(page=1, per_page=50) == payload


def test_fetch_daily_prices_handles_invalid_json():
    client = CoinGeckoClient("https://example.test", 1)
    client.session = FakeSession(FakeResponse(json_error=ValueError("bad json")))

    with pytest.raises(ApiError) as exc:
        client.fetch_daily_prices("bitcoin", 365)

    assert exc.value.code == "invalid_market_data_json"


def test_fetch_market_coins_rejects_invalid_payload():
    client = CoinGeckoClient("https://example.test", 1)
    client.session = FakeSession(FakeResponse(payload={"id": "bitcoin"}))

    with pytest.raises(ApiError) as exc:
        client.fetch_market_coins()

    assert exc.value.code == "invalid_market_data"


def test_fetch_daily_prices_handles_timeout():
    client = CoinGeckoClient("https://example.test", 1)
    client.session = FakeSession(error=requests.Timeout())

    with pytest.raises(ApiError) as exc:
        client.fetch_daily_prices("bitcoin", 365)

    assert exc.value.code == "market_data_timeout"


@pytest.mark.parametrize("price", [0, -1, "nan"])
def test_fetch_daily_prices_rejects_invalid_prices(price):
    client = CoinGeckoClient("https://example.test", 1)
    client.session = FakeSession(FakeResponse(payload={"prices": [[1, price]]}))

    with pytest.raises(ApiError) as exc:
        client.fetch_daily_prices("bitcoin", 365)

    assert exc.value.code == "invalid_market_data_price"
