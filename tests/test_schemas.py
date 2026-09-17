"""Unit tests for Pydantic schemas in timesfm-serve."""

from timesfm_serve.schemas import (
    FinancialBar,
    FinancialForecastRequest,
    ForecastRequest,
    HealthResponse,
)


def test_forecast_request_validation() -> None:
    req = ForecastRequest(
        series=[[1.0, 2.0, 3.0, 4.0, 5.0]],
        horizon=10,
    )
    assert req.horizon == 10
    assert len(req.series[0]) == 5


def test_financial_request_validation() -> None:
    bars = [
        FinancialBar(open=100.0, high=101.0, low=99.0, close=100.5, volume=1000)
        for _ in range(12)
    ]
    req = FinancialForecastRequest(
        symbol="AAPL",
        bars=bars,
        horizon_bars=5,
        current_price=100.5,
    )
    assert req.symbol == "AAPL"
    assert len(req.bars) == 12
    assert req.horizon_bars == 5


def test_health_response() -> None:
    health = HealthResponse(
        status="healthy",
        model_id="google/timesfm-3.0-pytorch",
        device="cpu",
        torch_version="2.4.0",
        multivariate_enabled=True,
        version="0.1.0",
    )
    assert health.status == "healthy"
    assert health.multivariate_enabled is True
