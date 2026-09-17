"""API router defining forecasting endpoints for TimesFM Serve."""

from fastapi import APIRouter, HTTPException, Request

from timesfm_serve.modules.forecasting.schemas import (
    ForecastRequest,
    ForecastResponse,
    MultivariateForecastRequest,
    MultivariateForecastResponse,
)

router = APIRouter()

@router.post("/v1/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast_univariate(
    payload: ForecastRequest,
    request: Request,
) -> ForecastResponse:
    """Forecast single or batch 1D time series with probabilistic quantile bands."""
    engine = getattr(request.app.state, "engine", None)
    settings = getattr(request.app.state, "settings", None)
    if engine is None or settings is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    horizon = payload.horizon or settings.default_horizon
    if horizon > settings.max_horizon:
        raise HTTPException(
            status_code=400,
            detail=f"Requested horizon {horizon} exceeds maximum allowed {settings.max_horizon}",
        )

    try:
        forecasts = engine.forecast_univariate(payload.series, horizon=horizon)
        return ForecastResponse(
            model_id=settings.timesfm_model_id,
            horizon=horizon,
            forecasts=forecasts,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@router.post(
    "/v1/forecast/multivariate",
    response_model=MultivariateForecastResponse,
    tags=["Forecasting"],
)
async def forecast_multivariate(
    payload: MultivariateForecastRequest,
    request: Request,
) -> MultivariateForecastResponse:
    """Multivariate forecasting across multiple interrelated channels with dynamic covariates."""
    engine = getattr(request.app.state, "engine", None)
    settings = getattr(request.app.state, "settings", None)
    if engine is None or settings is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    horizon = payload.horizon or settings.default_horizon

    try:
        targets = engine.forecast_multivariate(
            targets=payload.targets,
            past_covariates=payload.past_covariates,
            future_covariates=payload.future_covariates,
            horizon=horizon,
        )

        return MultivariateForecastResponse(
            model_id=settings.timesfm_model_id,
            horizon=horizon,
            targets=targets,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multivariate error: {exc}") from exc
