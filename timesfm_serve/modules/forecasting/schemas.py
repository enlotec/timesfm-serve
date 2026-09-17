"""Pydantic schemas for TimesFM Serve Forecasting API."""

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
