import math

import requests

from errors import ApiError


class CoinGeckoClient:
    def __init__(self, base_url, timeout_seconds):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def fetch_daily_prices(self, coin_id, days):
        try:
            response = self.session.get(
                f"{self.base_url}/coins/{coin_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": str(days),
                    "interval": "daily",
                },
                timeout=self.timeout_seconds,
            )
        except requests.Timeout:
            raise ApiError(
                message="CoinGecko request timed out.",
                status_code=504,
                code="market_data_timeout",
            )
        except requests.RequestException:
            raise ApiError(
                message="Unable to fetch market data.",
                status_code=502,
                code="market_data_unavailable",
            )

        if response.status_code == 404:
            raise ApiError(
                message=f"{coin_id} is not supported.",
                status_code=400,
                code="unsupported_coin",
            )

        if response.status_code == 429:
            raise ApiError(
                message="CoinGecko rate limit exceeded.",
                status_code=502,
                code="market_data_rate_limited",
            )

        if not response.ok:
            raise ApiError(
                message="CoinGecko returned an unexpected response.",
                status_code=502,
                code="market_data_error",
            )

        try:
            data = response.json()
        except ValueError:
            raise ApiError(
                message="Market data response was not valid JSON.",
                status_code=502,
                code="invalid_market_data_json",
            )

        prices = data.get("prices")
        if not isinstance(prices, list):
            raise ApiError(
                message="Market data response is missing prices.",
                status_code=502,
                code="invalid_market_data",
            )

        daily_prices = []
        for price in prices:
            if not isinstance(price, list) or len(price) <= 1:
                continue
            try:
                daily_price = float(price[1])
            except (TypeError, ValueError):
                raise ApiError(
                    message="Market data contained a non-numeric price.",
                    status_code=502,
                    code="invalid_market_data_price",
                )
            if not math.isfinite(daily_price) or daily_price <= 0:
                raise ApiError(
                    message="Market data contained an invalid price.",
                    status_code=502,
                    code="invalid_market_data_price",
                )
            daily_prices.append(daily_price)

        return daily_prices

    def fetch_market_coins(self, page=1, per_page=50):
        try:
            response = self.session.get(
                f"{self.base_url}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": str(per_page),
                    "page": str(page),
                    "sparkline": "true",
                    "price_change_percentage": "24h",
                },
                timeout=self.timeout_seconds,
            )
        except requests.Timeout:
            raise ApiError(
                message="CoinGecko request timed out.",
                status_code=504,
                code="market_data_timeout",
            )
        except requests.RequestException:
            raise ApiError(
                message="Unable to fetch market data.",
                status_code=502,
                code="market_data_unavailable",
            )

        if response.status_code == 429:
            raise ApiError(
                message="CoinGecko rate limit exceeded.",
                status_code=502,
                code="market_data_rate_limited",
            )

        if not response.ok:
            raise ApiError(
                message="CoinGecko returned an unexpected response.",
                status_code=502,
                code="market_data_error",
            )

        try:
            data = response.json()
        except ValueError:
            raise ApiError(
                message="Market data response was not valid JSON.",
                status_code=502,
                code="invalid_market_data_json",
            )

        if not isinstance(data, list):
            raise ApiError(
                message="Market data response is invalid.",
                status_code=502,
                code="invalid_market_data",
            )

        return data
