"""Integration tests for the diagnostics module."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint_healthy(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "google/timesfm" in data["model_id"]
    assert "torch_version" in data
    assert "version" in data
    assert "device" in data


def test_health_endpoint_starting(client: TestClient) -> None:
    # Save original engine and temporarily set to None
    original_engine = client.app.state.engine
    client.app.state.engine = None
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "starting"
    finally:
        client.app.state.engine = original_engine


@pytest.mark.asyncio
async def test_lifespan_lifecycle(monkeypatch) -> None:
    from fastapi import FastAPI

    from timesfm_serve.core.engine import TimesFmEngine
    from timesfm_serve.core.lifespan import lifespan

    # Mock load_model so it doesn't trigger weights download
    monkeypatch.setattr(TimesFmEngine, "load_model", lambda self: None)

    app = FastAPI()
    async with lifespan(app):
        assert app.state.engine is not None
        assert app.state.settings is not None
