"""Integration tests for FastAPI endpoints in timesfm-serve."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "google/timesfm" in data["model_id"]
    assert "version" in data


def test_univariate_forecast_endpoint(client: TestClient) -> None:
    payload = {
        "series": [
            [100.0, 101.5, 102.0, 101.8, 103.2, 104.0, 103.5]
        ],
        "horizon": 5,
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 5
    assert len(data["forecasts"]) == 1

    forecast = data["forecasts"][0]
    assert len(forecast["mean"]) == 5
    assert len(forecast["quantiles"]["q10"]) == 5
    assert len(forecast["quantiles"]["q50"]) == 5
    assert len(forecast["quantiles"]["q90"]) == 5


def test_financial_forecast_endpoint(client: TestClient) -> None:
    bars = [
        {
            "timestamp": f"2026-09-17T10:{i:02d}:00Z",
            "open": 150.0 + i * 0.1,
            "high": 151.0 + i * 0.1,
            "low": 149.5 + i * 0.1,
            "close": 150.5 + i * 0.1,
            "volume": 50000,
        }
        for i in range(15)
    ]
    payload = {
        "symbol": "NVDA",
        "bars": bars,
        "horizon_bars": 6,
        "current_price": 152.0,
    }
    response = client.post("/v1/forecast/financial", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert data["horizon_bars"] == 6
    assert data["signal"] in {"bullish", "bearish", "neutral"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert "trajectory" in data
    assert len(data["trajectory"]["p50"]) == 6


def test_multivariate_forecast_endpoint(client: TestClient) -> None:
    payload = {
        "targets": [
            {"name": "channel_a", "history": [10.0, 11.0, 12.0, 13.0, 14.0]},
            {"name": "channel_b", "history": [20.0, 19.0, 18.0, 17.0, 16.0]},
        ],
        "horizon": 4,
    }
    response = client.post("/v1/forecast/multivariate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 4
    assert len(data["targets"]) == 2
    assert data["targets"][0]["name"] == "channel_a"
    assert len(data["targets"][0]["mean"]) == 4
    assert len(data["targets"][0]["quantiles"]["q10"]) == 4

