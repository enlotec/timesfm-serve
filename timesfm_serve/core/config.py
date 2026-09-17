"""Configuration settings for TimesFM Serve."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model & Checkpoint
    timesfm_model_id: str = Field(
        default="google/timesfm-3.0-pytorch",
        description="Hugging Face repo or local path for TimesFM model weights.",
    )
    device: str = Field(
        default="auto",
        description="Execution device: 'auto' (detect CUDA if present), 'cuda', or 'cpu'.",
    )
    
    # Inference parameters
    default_horizon: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Default forecast horizon length if not specified in request.",
    )
    max_horizon: int = Field(
        default=512,
        ge=1,
        description="Maximum allowed forecast horizon length.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum batch size for inference.",
    )
    normalize_inputs: bool = Field(
        default=True,
        description="Apply internal z-score / instance scaling on series.",
    )
    
    # Server & Telemetry
    host: str = Field(default="0.0.0.0", description="Host to bind the server.")
    port: int = Field(default=8088, description="Port to bind the server.")
    log_level: str = Field(default="info", description="Logging verbosity.")
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face API token if needed for rate limit avoidance.",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
