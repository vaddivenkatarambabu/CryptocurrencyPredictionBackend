from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from errors import ApiError, error_response
from logging_config import configure_logging
from services.coingecko import CoinGeckoClient
from services.prediction import PredictionService
from validation import validate_prediction_request


def create_app(config_class=Config):
    configure_logging()

    app = Flask(__name__)
    app.config.from_object(config_class)
    if app.config["TRUST_PROXY_HEADERS"]:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    CORS(app, resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}})

    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[app.config["RATE_LIMIT_DEFAULT"]],
        storage_uri=app.config["RATE_LIMIT_STORAGE_URI"],
    )

    coingecko_client = CoinGeckoClient(
        base_url=app.config["COINGECKO_BASE_URL"],
        timeout_seconds=app.config["COINGECKO_TIMEOUT_SECONDS"],
    )
    prediction_service = PredictionService.from_file(
        model_path=app.config["MODEL_PATH"],
        coingecko_client=coingecko_client,
        history_days=app.config["HISTORY_DAYS"],
        max_prediction_days=app.config["MAX_PREDICTION_DAYS"],
    )

    @app.before_request
    def attach_request_id():
        g.request_id = request.headers.get("X-Request-ID") or app.config[
            "REQUEST_ID_FACTORY"
        ]()

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(exc):
        app.logger.info(
            "api_error",
            extra={
                "request_id": g.get("request_id"),
                "code": exc.code,
                "status_code": exc.status_code,
            },
        )
        return error_response(exc, request_id=g.get("request_id"))

    @app.errorhandler(429)
    def handle_rate_limit(exc):
        api_error = ApiError(
            message="Too many requests. Please retry later.",
            status_code=429,
            code="rate_limit_exceeded",
        )
        return error_response(api_error, request_id=g.get("request_id"))

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        api_error = ApiError(
            message=exc.description,
            status_code=exc.code or 500,
            code=exc.name.lower().replace(" ", "_"),
        )
        return error_response(api_error, request_id=g.get("request_id"))

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc):
        app.logger.exception(
            "unhandled_exception",
            extra={"request_id": g.get("request_id")},
        )
        api_error = ApiError(
            message="Unexpected server error.",
            status_code=500,
            code="internal_error",
        )
        return error_response(api_error, request_id=g.get("request_id"))

    @app.get("/")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "crypto-prediction-api",
                "requestId": g.get("request_id"),
            }
        )

    @app.get("/coins")
    @limiter.limit(app.config["RATE_LIMIT_DEFAULT"])
    def coins():
        page = _int_query_arg("page", default=1, minimum=1, maximum=100)
        per_page = _int_query_arg("per_page", default=50, minimum=1, maximum=250)
        coins_data = coingecko_client.fetch_market_coins(page=page, per_page=per_page)
        return jsonify(coins_data)

    @app.post("/predict")
    @limiter.limit(app.config["RATE_LIMIT_PREDICT"])
    def predict():
        payload = validate_prediction_request(
            request.get_json(silent=True),
            max_days=app.config["MAX_PREDICTION_DAYS"],
        )
        result = prediction_service.predict(payload.coin_id, payload.days)
        response = result.to_dict()
        response["requestId"] = g.get("request_id")
        return jsonify(response)

    return app


def _int_query_arg(name, default, minimum, maximum):
    raw_value = request.args.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ApiError(
            message=f"{name} must be an integer.",
            status_code=400,
            code=f"invalid_{name}",
        )

    if value < minimum or value > maximum:
        raise ApiError(
            message=f"{name} must be between {minimum} and {maximum}.",
            status_code=400,
            code=f"invalid_{name}_range",
        )
    return value


app = create_app()


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
