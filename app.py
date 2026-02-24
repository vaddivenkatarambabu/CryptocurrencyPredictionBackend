from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import traceback

app = Flask(__name__)
CORS(app)

# Load trained LSTM model once
model = load_model("model.keras")


@app.route("/")
def health():
    return jsonify({"message": "Crypto Prediction API running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json()

        coin_id = request_data.get("coinId")
        days = int(request_data.get("days", 7))

        if not coin_id:
            return jsonify({"error": "coinId required"}), 400

        if days < 1 or days > 90:
            return jsonify({"error": "Days must be between 1 and 90"}), 400

        # Fetch historical data (1 year)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "365",
            "interval": "daily"
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return jsonify({"error": f"{coin_id} not supported"}), 400

        cg_data = response.json()
        prices = [p[1] for p in cg_data["prices"]]

        if len(prices) < 365:
            return jsonify({"error": "Not enough historical data"}), 400

        close_prices = np.array(prices).reshape(-1, 1)

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        timesteps = model.input_shape[1]

        # Current price
        current_price = float(close_prices[-1][0])

        # Future prediction
        last_window = scaled_data[-timesteps:].reshape(1, timesteps, 1)

        future_predictions = []

        for _ in range(days):
            next_scaled = model.predict(last_window, verbose=0)
            next_price = float(
                scaler.inverse_transform(next_scaled)[0][0]
            )

            future_predictions.append(round(next_price, 4))

            last_window = np.append(
                last_window[:, 1:, :],
                next_scaled.reshape(1, 1, 1),
                axis=1
            )

        change_percent = (
            (future_predictions[-1] - current_price)
            / current_price
        ) * 100

        change_percent = round(change_percent, 2)

        if change_percent > 1:
            trend = "bullish"
        elif change_percent < -1:
            trend = "bearish"
        else:
            trend = "neutral"

        # Response
        return jsonify({
            "coinId": coin_id,
            "coinName": coin_id.capitalize(),
            "symbol": coin_id.upper(),
            "days": days,
            "currentPrice": round(current_price, 4),
            "prediction": {
                "predictedPrice": future_predictions[-1],
                "dailyPrices": future_predictions,
                "confidence": 0.95,
                "changePercent": change_percent,
                "trend": trend,
                "analysis": f"LSTM predicts {trend} trend over next {days} days.",
                "supportLevel": min(future_predictions),
                "resistanceLevel": max(future_predictions)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)