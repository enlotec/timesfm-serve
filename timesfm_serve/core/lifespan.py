"""FastAPI lifespan context manager for TimesFM Serve."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
import structlog

from timesfm_serve.core.config import get_settings
from timesfm_serve.core.engine import TimesFmEngine

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Initializing TimesFM engine during startup...")
    engine = TimesFmEngine(settings)
    engine.load_model()
    
    from timesfm_serve.core.engine import set_global_engine
    set_global_engine(engine)
    
    app.state.engine = engine
    app.state.settings = settings
    logger.info("TimesFM engine ready to serve requests.")
    yield
    logger.info("Shutting down TimesFM engine.")
