# Cryptocurrency Prediction

A full-stack cryptocurrency price prediction project.

The project idea is unchanged:

- The frontend shows live cryptocurrency market data.
- The backend fetches data from CoinGecko.
- A TensorFlow/Keras model predicts future prices.
- The frontend displays the prediction, trend, and charts.

## Project Structure

```text
CryptocurrencyPrediction/
  backend/
    app.py
    config.py
    errors.py
    logging_config.py
    validation.py
    model.keras
    requirements.txt
    requirements-dev.txt
    Dockerfile
    services/
      coingecko.py
      prediction.py
    tests/
  frontend/
    index.html
    package.json
    package-lock.json
    vite.config.ts
    src/
      components/
      hooks/
      pages/
      services/
  .env.example
  .gitignore
```

## Backend Setup

```powershell
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest
python app.py
```

Backend URL:

```text
http://127.0.0.1:5000
```

Backend routes:

- `GET /` - health check
- `GET /coins` - market coin list for the frontend
- `POST /predict` - price prediction

Example prediction request:

```json
{
  "coinId": "bitcoin",
  "days": 30
}
```

## Frontend Setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:8080
```

## Environment

Real `.env` files are not required for local development. The default backend
and frontend URLs are already in code.

Use `.env.example` only as a reference when deploying or changing local ports.
Keep real `.env` files private.

## Prediction Limit

The app now supports predictions from 1 to 365 days by default.

If needed, the backend limit can be changed with:

```text
MAX_PREDICTION_DAYS=365
```

Very long prediction ranges are technically possible because the model predicts
one day at a time, but longer ranges are slower and less reliable.
