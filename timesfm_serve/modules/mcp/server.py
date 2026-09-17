"""Model Context Protocol (MCP) server for TimesFM Serve."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server import MCPServer

from timesfm_serve.core.config import get_settings
from timesfm_serve.core.engine import TimesFmEngine, get_global_engine, set_global_engine
from timesfm_serve.modules.forecasting.schemas import FutureCovariate, NamedSeries

mcp = MCPServer("TimesFM Serve")


def _resolve_engine() -> TimesFmEngine:
    """Return the global engine, initializing it lazily if running standalone."""
    try:
        return get_global_engine()
    except RuntimeError:
        settings = get_settings()
        engine = TimesFmEngine(settings)
        engine.load_model()
        set_global_engine(engine)
        return engine


@mcp.tool()
async def forecast_univariate(
    series: list[list[float]],
    horizon: int = 32,
    frequency: str | None = None,
) -> dict[str, Any]:
    """Forecast single or batch 1D time series with probabilistic quantile bands using Google TimesFM.

    Args:
        series: A list of 1D time series (each being a list of floats) to forecast.
        horizon: Number of future time steps to forecast.
        frequency: Optional sampling cadence (e.g. 'hourly', 'daily', '15m').
    """
    engine = _resolve_engine()
    results = engine.forecast_univariate(series, horizon=horizon, frequency=frequency)
    return {
        "forecasts": [res.model_dump() for res in results]
    }


@mcp.tool()
async def forecast_multivariate(
    targets: list[dict[str, Any]],
    horizon: int = 32,
    past_covariates: list[dict[str, Any]] | None = None,
    future_covariates: list[dict[str, Any]] | None = None,
    frequency: str | None = None,
) -> dict[str, Any]:
    """Multivariate forecasting across multiple interrelated channels with dynamic covariates.

    Args:
        targets: List of dicts with 'name' (str) and 'history' (list of floats).
        horizon: Number of future time steps to forecast.
        past_covariates: List of dicts with 'name' and 'history' matching the target length.
        future_covariates: List of dicts with 'name' and 'future_values' (and optional 'history').
        frequency: Optional sampling cadence (e.g. 'hourly', 'daily', '15m').
    """
    engine = _resolve_engine()

    # Parse dicts to schemas
    parsed_targets = [NamedSeries(**t) for t in targets]
    parsed_past_cov = [NamedSeries(**c) for c in past_covariates] if past_covariates else None
    parsed_future_cov = [FutureCovariate(**c) for c in future_covariates] if future_covariates else None

    results = engine.forecast_multivariate(
        targets=parsed_targets,
        past_covariates=parsed_past_cov,
        future_covariates=parsed_future_cov,
        horizon=horizon,
        frequency=frequency,
    )
    return {
        "targets": [res.model_dump() for res in results]
    }


def run_stdio() -> None:
    """Run the MCP server over standard I/O for Claude Desktop, Cursor, and AI agents."""
    _resolve_engine()
    asyncio.run(mcp.run_stdio_async())
