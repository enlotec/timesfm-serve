"""Main application entrypoint for TimesFM Serve."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
import uvicorn

from timesfm_serve import __version__
from timesfm_serve.modules.forecasting.router import router as forecasting_router
from timesfm_serve.modules.diagnostics.router import router as diagnostics_router
from timesfm_serve.modules.mcp.server import mcp
from timesfm_serve.core.config import get_settings
from timesfm_serve.core.lifespan import lifespan

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="TimesFM Serve",
        description=(
            "### Production-grade REST API, MCP Server, and Docker container for Google's TimesFM\n\n"
            "Developed by **enlotec** under the **Apache License, Version 2.0**.\n\n"
            "This service provides a pure wrapper around the Google TimesFM foundation model, "
            "exposing Univariate, Multivariate, and Covariate forecasting capabilities over REST and Model Context Protocol (MCP)."
        ),
        version=__version__,
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        contact={
            "name": "enlotec",
            "url": "https://github.com/enlotec/timesfm-serve",
        },
        lifespan=lifespan,
    )

    # Enable CORS for web dashboards, Studio frontends, and notebooks
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(diagnostics_router)
    app.include_router(forecasting_router)
    
    # Mount the FastMCP Starlette app
    app.mount("/mcp", mcp.get_starlette_app())
    
    return app


app = create_app()


def cli() -> None:
    """CLI entrypoint for running the server directly."""
    settings = get_settings()
    uvicorn.run(
        "timesfm_serve.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    cli()
