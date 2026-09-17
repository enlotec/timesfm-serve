"""Model Context Protocol (MCP) server for TimesFM Serve."""

from mcp.server.fastmcp import FastMCP
from timesfm_serve.core.engine import get_global_engine
from timesfm_serve.modules.forecasting.schemas import NamedSeries, FutureCovariate

mcp = FastMCP("TimesFM Serve")

@mcp.tool()
async def forecast_univariate(series: list[list[float]], horizon: int = 32) -> dict:
    """
    Forecast single or batch 1D time series with probabilistic quantile bands using Google TimesFM.
    
    Args:
        series: A list of 1D time series (each being a list of floats) to forecast.
        horizon: Number of future time steps to forecast.
    """
    engine = get_global_engine()
    results = engine.forecast_univariate(series, horizon=horizon)
    return {
        "forecasts": [res.model_dump() for res in results]
    }

@mcp.tool()
async def forecast_multivariate(
    targets: list[dict],
    horizon: int = 32,
    past_covariates: list[dict] | None = None,
    future_covariates: list[dict] | None = None,
) -> dict:
    """
    Multivariate forecasting across multiple interrelated channels with dynamic covariates.
    
    Args:
        targets: List of dicts with 'name' (str) and 'history' (list of floats).
        horizon: Number of future time steps to forecast.
        past_covariates: List of dicts with 'name' and 'history' matching the target length.
        future_covariates: List of dicts with 'name' and 'future_values' matching the horizon length.
    """
    engine = get_global_engine()
    
    # Parse dicts to schemas
    parsed_targets = [NamedSeries(**t) for t in targets]
    parsed_past_cov = [NamedSeries(**c) for c in past_covariates] if past_covariates else None
    parsed_future_cov = [FutureCovariate(**c) for c in future_covariates] if future_covariates else None
    
    results = engine.forecast_multivariate(
        targets=parsed_targets,
        past_covariates=parsed_past_cov,
        future_covariates=parsed_future_cov,
        horizon=horizon,
    )
    return {
        "targets": [res.model_dump() for res in results]
    }
