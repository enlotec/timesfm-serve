"""API router defining diagnostic endpoints for TimesFM Serve."""

from fastapi import APIRouter, Request

try:
    import torch
    TORCH_VERSION = torch.__version__
except ImportError:
    torch = None  # type: ignore
    TORCH_VERSION = "unavailable"

from timesfm_serve import __version__
from timesfm_serve.modules.diagnostics.schemas import HealthResponse

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
