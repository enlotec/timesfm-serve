"""Pydantic schemas for TimesFM Serve Diagnostics API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    torch_version: str
    multivariate_enabled: bool
    version: str
