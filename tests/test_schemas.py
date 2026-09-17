"""Unit tests for Pydantic schemas in timesfm-serve."""

from timesfm_serve.modules.diagnostics.schemas import HealthResponse
from timesfm_serve.modules.forecasting.schemas import (
    ForecastRequest,
    FutureCovariate,
    MultivariateForecastRequest,
    NamedSeries,
    QuantileBands,
)


def test_forecast_request_validation() -> None:
    req = ForecastRequest(
        series=[[1.0, 2.0, 3.0, 4.0, 5.0]],
        horizon=10,
        frequency="hourly",
    )
    assert req.horizon == 10
    assert len(req.series[0]) == 5
    assert req.frequency == "hourly"


def test_multivariate_request_validation() -> None:
    targets = [
        NamedSeries(name="load_a", history=[10.0, 11.0, 12.0, 13.0]),
        NamedSeries(name="load_b", history=[20.0, 21.0, 22.0, 23.0]),
    ]
    past_covs = [
        NamedSeries(name="temp", history=[15.0, 15.5, 16.0, 16.5]),
    ]
    future_covs = [
        FutureCovariate(name="is_holiday", future_values=[0.0, 0.0, 1.0]),
    ]
    req = MultivariateForecastRequest(
        targets=targets,
        past_covariates=past_covs,
        future_covariates=future_covs,
        horizon=3,
        frequency="15m",
    )
    assert len(req.targets) == 2
    assert req.horizon == 3
    assert req.frequency == "15m"
    assert len(req.future_covariates[0].future_values) == 3


def test_quantile_bands_all_nine() -> None:
    bands = QuantileBands(
        q10=[1.0],
        q20=[1.2],
        q30=[1.4],
        q40=[1.6],
        q50=[1.8],
        q60=[2.0],
        q70=[2.2],
        q80=[2.4],
        q90=[2.6],
    )
    assert bands.q10 == [1.0]
    assert bands.q50 == [1.8]
    assert bands.q90 == [2.6]


def test_health_response() -> None:
    health = HealthResponse(
        status="healthy",
        model_id="google/timesfm-3.0-pytorch",
        device="cpu",
        torch_version="2.14.0",
        multivariate_enabled=True,
        version="0.1.0",
    )
    assert health.status == "healthy"
    assert health.multivariate_enabled is True
