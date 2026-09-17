"""Main application entrypoint for TimesFM Serve."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
import uvicorn

from timesfm_serve import __version__
from timesfm_serve.api import router
from timesfm_serve.config import get_settings
from timesfm_serve.lifespan import lifespan

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="TimesFM Serve",
        description=(
            "### Production-grade REST API and Docker container for Google's TimesFM\n\n"
            "Developed by **enlotec** under the **Apache License, Version 2.0**.\n\n"
            "**LEGAL & FINANCIAL DISCLAIMER:**\n"
            "The forecasts, trajectories, return estimates, and directional indicators provided "
            "by this service are for computational research and informational purposes only. "
            "Nothing herein constitutes financial, investment, trading, legal, or tax advice. "
            "Under no circumstances shall enlotec or its contributors be liable for any financial "
            "losses, lost profits, or trading damages arising from the use of this service."
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

    app.include_router(router)
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
