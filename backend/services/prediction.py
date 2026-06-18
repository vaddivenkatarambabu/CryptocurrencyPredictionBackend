from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from errors import ApiError


@dataclass(frozen=True)
class PredictionResult:
    coin_id: str
    days: int
    current_price: float
    daily_prices: list[float]

    def to_dict(self):
        _ensure_positive_finite(self.current_price, "current price")
        if not self.daily_prices:
            raise ApiError(
                message="Prediction did not produce any prices.",
                status_code=500,
                code="prediction_failed",
            )
        for daily_price in self.daily_prices:
            _ensure_positive_finite(daily_price, "predicted price")

        predicted_price = self.daily_prices[-1]
        change_percent = round(
            ((predicted_price - self.current_price) / self.current_price) * 100,
            2,
        )
        trend = _trend_from_change(change_percent)

        return {
            "coinId": self.coin_id,
            "coinName": self.coin_id.capitalize(),
            "symbol": self.coin_id.upper(),
            "days": self.days,
            "currentPrice": round(self.current_price, 4),
            "prediction": {
                "predictedPrice": predicted_price,
                "dailyPrices": self.daily_prices,
                "confidence": 0.95,
                "confidenceNote": "Static placeholder. Replace with a calibrated model metric before financial use.",
                "changePercent": change_percent,
                "trend": trend,
                "analysis": f"LSTM predicts {trend} trend over next {self.days} days.",
                "supportLevel": min(self.daily_prices),
                "resistanceLevel": max(self.daily_prices),
            },
        }


class PredictionService:
    def __init__(self, model, coingecko_client, history_days, max_prediction_days):
        self.model = model
        self.coingecko_client = coingecko_client
        self.history_days = history_days
        self.max_prediction_days = max_prediction_days

        input_shape = getattr(model, "input_shape", None)
        if not input_shape or len(input_shape) < 2 or not input_shape[1]:
            raise RuntimeError("Model must define a fixed timestep input shape.")
        self.timesteps = int(input_shape[1])

    @classmethod
    def from_file(cls, model_path, coingecko_client, history_days, max_prediction_days):
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"Model file not found: {path}")
        model = load_model(path)
        return cls(
            model=model,
            coingecko_client=coingecko_client,
            history_days=history_days,
            max_prediction_days=max_prediction_days,
        )

    def predict(self, coin_id, days):
        if days > self.max_prediction_days:
            raise ApiError(
                message=f"days must be between 1 and {self.max_prediction_days}.",
                status_code=400,
                code="invalid_days_range",
            )

        prices = self.coingecko_client.fetch_daily_prices(
            coin_id=coin_id,
            days=self.history_days,
        )

        if len(prices) < self.timesteps:
            raise ApiError(
                message="Not enough historical data for this model.",
                status_code=400,
                code="insufficient_market_history",
            )

        close_prices = np.asarray(prices, dtype=np.float64).reshape(-1, 1)
        if not np.all(np.isfinite(close_prices)) or np.any(close_prices <= 0):
            raise ApiError(
                message="Market data contained invalid prices.",
                status_code=502,
                code="invalid_market_data_price",
            )

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        last_window = scaled_data[-self.timesteps :].reshape(1, self.timesteps, 1)
        future_predictions = []

        for _ in range(days):
            try:
                raw_prediction = np.asarray(
                    self.model.predict(last_window, verbose=0),
                    dtype=np.float64,
                ).reshape(1, 1)
                if not np.all(np.isfinite(raw_prediction)):
                    raise ValueError("model returned a non-finite value")

                next_scaled = np.clip(raw_prediction, 0, 1)
                next_price = float(scaler.inverse_transform(next_scaled)[0][0])
            except Exception as exc:
                raise ApiError(
                    message="Prediction model failed to generate a valid result.",
                    status_code=500,
                    code="prediction_failed",
                ) from exc

            _ensure_positive_finite(next_price, "predicted price")
            future_predictions.append(round(next_price, 4))
            last_window = np.append(
                last_window[:, 1:, :],
                next_scaled.reshape(1, 1, 1),
                axis=1,
            )

        return PredictionResult(
            coin_id=coin_id,
            days=days,
            current_price=float(close_prices[-1][0]),
            daily_prices=future_predictions,
        )


def _trend_from_change(change_percent):
    if change_percent > 1:
        return "bullish"
    if change_percent < -1:
        return "bearish"
    return "neutral"


def _ensure_positive_finite(value, label):
    if not np.isfinite(value) or value <= 0:
        raise ApiError(
            message=f"Prediction produced an invalid {label}.",
            status_code=500,
            code="prediction_failed",
        )
