"""Test fixtures and configuration for timesfm-serve."""

import pytest
from fastapi.testclient import TestClient

from timesfm_serve.config import Settings
from timesfm_serve.engine import TimesFmEngine
from timesfm_serve.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        timesfm_model_id="google/timesfm-2.0-500m-pytorch",
        device="cpu",
        default_horizon=5,
        max_horizon=64,
        batch_size=4,
    )


@pytest.fixture
def mock_engine(test_settings: Settings) -> TimesFmEngine:
    engine = TimesFmEngine(test_settings)
    engine._is_mock = True
    return engine


@pytest.fixture
def client(test_settings: Settings, mock_engine: TimesFmEngine) -> TestClient:
    app = create_app()
    app.state.settings = test_settings
    app.state.engine = mock_engine
    return TestClient(app)
