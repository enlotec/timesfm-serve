"""Pydantic schemas for TimesFM Serve REST API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. Univariate Forecast Models
# ------------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    series: list[list[float]] = Field(
        ...,
        min_length=1,
        description="List of 1D time series to forecast (e.g., historical values).",
        examples=[[[10.1, 10.5, 11.2, 10.9, 12.1, 12.5, 13.0]]],
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        le=512,
        description="Number of future time steps to forecast. Defaults to server setting.",
    )
    frequency: str | None = Field(
        default=None,
        description="Optional sampling cadence (e.g. 'hourly', 'daily', '15m').",
    )


class QuantileBands(BaseModel):
    q10: list[float] = Field(..., description="10th percentile forecast (lower bound).")
    q20: list[float] | None = None
    q30: list[float] | None = None
    q40: list[float] | None = None
    q50: list[float] = Field(..., description="50th percentile forecast (median).")
    q60: list[float] | None = None
    q70: list[float] | None = None
    q80: list[float] | None = None
    q90: list[float] = Field(..., description="90th percentile forecast (upper bound).")


class SeriesForecast(BaseModel):
    series_index: int = Field(..., description="Index of the series from the request.")
    mean: list[float] = Field(..., description="Mean/point trajectory forecast.")
    quantiles: QuantileBands = Field(..., description="Probabilistic quantile bands.")


class ForecastResponse(BaseModel):
    model_id: str
    horizon: int
    forecasts: list[SeriesForecast]


# ------------------------------------------------------------------------------
# 2. Multivariate & Covariate Models
# ------------------------------------------------------------------------------

class NamedSeries(BaseModel):
    name: str = Field(..., description="Channel name or identifier.")
    history: list[float] = Field(..., min_length=2, description="Historical observed values.")


class FutureCovariate(BaseModel):
    name: str = Field(..., description="Covariate channel name.")
    future_values: list[float] = Field(..., description="Known future values matching forecast horizon.")


class MultivariateForecastRequest(BaseModel):
    targets: list[NamedSeries] = Field(
        ...,
        min_length=1,
        description="Primary target channels to forecast.",
    )
    past_covariates: list[NamedSeries] | None = Field(
        default=None,
        description="Dynamic historical covariates observed up to current timestamp.",
    )
    future_covariates: list[FutureCovariate] | None = Field(
        default=None,
        description="Known future covariates across the forecast horizon.",
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        le=512,
        description="Forecast horizon length.",
    )


class TargetForecastResult(BaseModel):
    name: str
    mean: list[float]
    quantiles: QuantileBands


class MultivariateForecastResponse(BaseModel):
    model_id: str
    horizon: int
    targets: list[TargetForecastResult]


# ------------------------------------------------------------------------------
# 3. Specialized Financial Time-Series Models
# ------------------------------------------------------------------------------

class FinancialBar(BaseModel):
    timestamp: str | None = None
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class FinancialForecastRequest(BaseModel):
    symbol: str = Field(..., description="Asset ticker symbol, e.g. 'AAPL' or 'SPY'.")
    bars: list[FinancialBar] = Field(
        ...,
        min_length=10,
        description="Ordered historical candlestick bars.",
    )
    horizon_bars: int = Field(
        default=12,
        ge=1,
        le=128,
        description="Number of future bars to forecast.",
    )
    current_price: float | None = Field(
        default=None,
        description="Current quote mid/last price. Defaults to last bar close.",
    )
    market_sentiment_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional Fear & Greed sentiment index (0..100).",
    )


class TrajectoryBands(BaseModel):
    p10: list[float] = Field(..., description="10th percentile simulated price path.")
    p50: list[float] = Field(..., description="Median (50th percentile) expected price path.")
    p90: list[float] = Field(..., description="90th percentile simulated price path.")


class FinancialForecastResponse(BaseModel):
    symbol: str
    horizon_bars: int
    current_price: float
    predicted_price_p50: float = Field(..., description="Expected price at end of horizon.")
    expected_return_bps: float = Field(
        ...,
        description="Expected basis point return ((predicted_price - current_price) / current_price * 10000).",
    )
    signal: str = Field(..., description="'bullish', 'bearish', or 'neutral'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Statistical signal confidence.")
    uncertainty_spread_bps: float = Field(
        ...,
        description="Width of 80% credible interval in basis points ((p90 - p10) / current_price * 10000).",
    )
    downside_var_p10_bps: float = Field(
        ...,
        description="10th percentile tail return in bps (Value at Risk proxy).",
    )
    trajectory: TrajectoryBands


# ------------------------------------------------------------------------------
# 4. System & Health Models
# ------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    torch_version: str
    multivariate_enabled: bool
    version: str
