"""API router defining endpoints for TimesFM Serve."""

from fastapi import APIRouter, HTTPException, Request

try:
    import torch
    TORCH_VERSION = torch.__version__
except ImportError:
    torch = None  # type: ignore
    TORCH_VERSION = "unavailable"

from timesfm_serve import __version__
from timesfm_serve.schemas import (
    FinancialForecastRequest,
    FinancialForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MultivariateForecastRequest,
    MultivariateForecastResponse,
    QuantileBands,
    TargetForecastResult,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def get_health(request: Request) -> HealthResponse:
    """Check readiness, active model checkpoint, and device acceleration."""
    engine = getattr(request.app.state, "engine", None)
    settings = getattr(request.app.state, "settings", None)

    return HealthResponse(
        status="healthy" if engine is not None else "starting",
        model_id=settings.timesfm_model_id if settings else "unknown",
        device=engine.device if engine else "unknown",
        torch_version=TORCH_VERSION,
        multivariate_enabled=engine.is_v3 if engine else False,
        version=__version__,
    )


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


@router.post(
    "/v1/forecast/financial",
    response_model=FinancialForecastResponse,
    tags=["Financial Forecasting"],
)
async def forecast_financial(
    payload: FinancialForecastRequest,
    request: Request,
) -> FinancialForecastResponse:
    """Domain-specialized endpoint for historical OHLCV bars.

    Returns forward price trajectories, expected basis-point return, uncertainty bands,
    downside Value at Risk (VaR), and directional bias ('bullish', 'bearish', 'neutral').
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    try:
        return engine.forecast_financial(
            symbol=payload.symbol,
            bars=payload.bars,
            horizon_bars=payload.horizon_bars,
            current_price=payload.current_price,
            market_sentiment_score=payload.market_sentiment_score,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Financial forecast error: {exc}") from exc
