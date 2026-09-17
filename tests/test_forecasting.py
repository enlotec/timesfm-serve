"""Integration tests for the forecasting module endpoints."""

from fastapi.testclient import TestClient


def test_univariate_forecast_success(client: TestClient) -> None:
    payload = {
        "series": [
            [100.0, 101.5, 102.0, 101.8, 103.2, 104.0, 103.5]
        ],
        "horizon": 5,
        "frequency": "hourly",
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 5
    assert len(data["forecasts"]) == 1

    forecast = data["forecasts"][0]
    assert len(forecast["mean"]) == 5
    # Verify all 9 quantiles are populated
    for q_idx in range(10, 100, 10):
        key = f"q{q_idx}"
        assert key in forecast["quantiles"]
        assert len(forecast["quantiles"][key]) == 5


def test_univariate_empty_series_validation(client: TestClient) -> None:
    response = client.post("/v1/forecast", json={"series": [], "horizon": 5})
    assert response.status_code == 422


def test_univariate_horizon_exceeds_max(client: TestClient) -> None:
    response = client.post(
        "/v1/forecast",
        json={"series": [[1.0, 2.0, 3.0]], "horizon": 9999},
    )
    assert response.status_code in (400, 422)


def test_univariate_cadence_variations(client: TestClient) -> None:
    for cadence in ["15m", "daily", "weekly", "monthly", "high"]:
        res = client.post(
            "/v1/forecast",
            json={"series": [[10.0, 11.0, 12.0]], "horizon": 2, "frequency": cadence},
        )
        assert res.status_code == 200
        assert len(res.json()["forecasts"][0]["mean"]) == 2


def test_multivariate_forecast_success(client: TestClient) -> None:
    payload = {
        "targets": [
            {"name": "channel_a", "history": [10.0, 11.0, 12.0, 13.0, 14.0]},
            {"name": "channel_b", "history": [20.0, 19.0, 18.0, 17.0, 16.0]},
        ],
        "horizon": 4,
        "frequency": "daily",
    }
    response = client.post("/v1/forecast/multivariate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 4
    assert len(data["targets"]) == 2
    assert data["targets"][0]["name"] == "channel_a"
    assert len(data["targets"][0]["mean"]) == 4
    assert len(data["targets"][0]["quantiles"]["q10"]) == 4
    assert len(data["targets"][0]["quantiles"]["q90"]) == 4


def test_multivariate_with_past_and_future_covariates(client: TestClient) -> None:
    payload = {
        "targets": [
            {"name": "solar_output", "history": [50.0, 60.0, 75.0, 80.0, 90.0]},
        ],
        "past_covariates": [
            {"name": "temperature", "history": [20.0, 21.0, 22.0, 23.0, 24.0]},
        ],
        "future_covariates": [
            {"name": "cloud_cover_forecast", "future_values": [0.1, 0.2, 0.15]},
        ],
        "horizon": 3,
    }
    response = client.post("/v1/forecast/multivariate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 3
    assert len(data["targets"]) == 1
    assert data["targets"][0]["name"] == "solar_output"
    assert len(data["targets"][0]["mean"]) == 3
    assert len(data["targets"][0]["quantiles"]["q50"]) == 3


def test_engine_not_initialized_error(client: TestClient) -> None:
    original_engine = client.app.state.engine
    client.app.state.engine = None
    try:
        res = client.post("/v1/forecast", json={"series": [[1, 2, 3]], "horizon": 2})
        assert res.status_code == 503
        assert "not initialized" in res.json()["detail"]
    finally:
        client.app.state.engine = original_engine


def test_univariate_inference_error_500(client: TestClient, monkeypatch) -> None:
    def broken_forecast(*args, **kwargs):
        raise RuntimeError("GPU OOM or internal crash")

    monkeypatch.setattr(client.app.state.engine, "forecast_univariate", broken_forecast)
    res = client.post("/v1/forecast", json={"series": [[1, 2, 3]], "horizon": 2})
    assert res.status_code == 500
    assert "Inference error" in res.json()["detail"]


def test_multivariate_inference_error_500(client: TestClient, monkeypatch) -> None:
    def broken_multi(*args, **kwargs):
        raise RuntimeError("Covariate alignment crash")

    monkeypatch.setattr(client.app.state.engine, "forecast_multivariate", broken_multi)
    res = client.post("/v1/forecast/multivariate", json={"targets": [{"name": "a", "history": [1, 2]}]})
    assert res.status_code == 500
    assert "Multivariate error" in res.json()["detail"]

