"""Unit and integration tests for Model Context Protocol (MCP) server."""

import pytest

from timesfm_serve.modules.mcp.server import (
    _resolve_engine,
    forecast_multivariate,
    forecast_univariate,
    mcp,
)


def test_mcp_server_initialization() -> None:
    assert mcp.name == "TimesFM Serve"
    sse_app = mcp.sse_app()
    assert sse_app is not None


def test_resolve_engine_helper(mock_engine) -> None:
    engine = _resolve_engine()
    assert engine is not None


def test_resolve_engine_lazy_creation(monkeypatch) -> None:
    import timesfm_serve.core.engine as eng
    from timesfm_serve.core.engine import TimesFmEngine
    eng._global_engine = None
    monkeypatch.setattr(TimesFmEngine, "load_model", lambda self: None)
    engine = _resolve_engine()
    assert engine is not None
    assert eng._global_engine is engine


@pytest.mark.asyncio
async def test_mcp_forecast_univariate_tool(mock_engine) -> None:
    res = await forecast_univariate(
        series=[[10.0, 11.0, 12.0, 13.0]],
        horizon=3,
        frequency="hourly",
    )
    assert "forecasts" in res
    assert len(res["forecasts"]) == 1
    forecast = res["forecasts"][0]
    assert len(forecast["mean"]) == 3
    for q_idx in range(10, 100, 10):
        assert f"q{q_idx}" in forecast["quantiles"]


@pytest.mark.asyncio
async def test_mcp_forecast_multivariate_tool(mock_engine) -> None:
    res = await forecast_multivariate(
        targets=[{"name": "cpu_util", "history": [25.0, 30.0, 45.0, 50.0]}],
        horizon=2,
        future_covariates=[{"name": "scheduled_job", "future_values": [1.0, 0.0]}],
        frequency="daily",
    )
    assert "targets" in res
    assert len(res["targets"]) == 1
    assert res["targets"][0]["name"] == "cpu_util"
    assert len(res["targets"][0]["mean"]) == 2
    assert "quantiles" in res["targets"][0]
