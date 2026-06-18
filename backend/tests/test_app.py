from pathlib import Path

import app as app_module


class FakePredictionResult:
    def to_dict(self):
        return {
            "coinId": "bitcoin",
            "coinName": "Bitcoin",
            "symbol": "BITCOIN",
            "days": 1,
            "currentPrice": 100.0,
            "prediction": {
                "predictedPrice": 101.0,
                "dailyPrices": [101.0],
                "confidence": 0.95,
                "changePercent": 1.0,
                "trend": "neutral",
                "analysis": "test",
                "supportLevel": 101.0,
                "resistanceLevel": 101.0,
            },
        }


class FakePredictionService:
    def predict(self, coin_id, days):
        assert coin_id == "bitcoin"
        assert days == 1
        return FakePredictionResult()


class TestConfig:
    TESTING = True
    DEBUG = False
    HOST = "127.0.0.1"
    PORT = 5000
    TRUST_PROXY_HEADERS = False
    MODEL_PATH = Path("unused.keras")
    COINGECKO_BASE_URL = "https://example.test"
    COINGECKO_TIMEOUT_SECONDS = 1
    HISTORY_DAYS = 365
    MAX_PREDICTION_DAYS = 365
    CORS_ORIGINS = ["http://localhost:5173"]
    RATE_LIMIT_DEFAULT = "1000 per minute"
    RATE_LIMIT_PREDICT = "1000 per minute"
    RATE_LIMIT_STORAGE_URI = "memory://"
    REQUEST_ID_FACTORY = staticmethod(lambda: "test-request-id")


def test_health_route(monkeypatch):
    monkeypatch.setattr(
        app_module.PredictionService,
        "from_file",
        lambda **kwargs: FakePredictionService(),
    )
    client = app_module.create_app(TestConfig).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_predict_route_success(monkeypatch):
    monkeypatch.setattr(
        app_module.PredictionService,
        "from_file",
        lambda **kwargs: FakePredictionService(),
    )
    client = app_module.create_app(TestConfig).test_client()

    response = client.post("/predict", json={"coinId": "bitcoin", "days": 1})

    assert response.status_code == 200
    assert response.json["coinId"] == "bitcoin"
    assert response.json["requestId"] == "test-request-id"


def test_coins_route_success(monkeypatch):
    monkeypatch.setattr(
        app_module.PredictionService,
        "from_file",
        lambda **kwargs: FakePredictionService(),
    )
    monkeypatch.setattr(
        app_module.CoinGeckoClient,
        "fetch_market_coins",
        lambda self, page=1, per_page=50: [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}
        ],
    )
    client = app_module.create_app(TestConfig).test_client()

    response = client.get("/coins")

    assert response.status_code == 200
    assert response.json[0]["id"] == "bitcoin"


def test_predict_route_validation_error(monkeypatch):
    monkeypatch.setattr(
        app_module.PredictionService,
        "from_file",
        lambda **kwargs: FakePredictionService(),
    )
    client = app_module.create_app(TestConfig).test_client()

    response = client.post("/predict", json={"coinId": "bitcoin", "days": "bad"})

    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_days"


def test_unknown_route_returns_not_found(monkeypatch):
    monkeypatch.setattr(
        app_module.PredictionService,
        "from_file",
        lambda **kwargs: FakePredictionService(),
    )
    client = app_module.create_app(TestConfig).test_client()

    response = client.get("/missing")

    assert response.status_code == 404
    assert response.json["error"]["code"] == "not_found"
