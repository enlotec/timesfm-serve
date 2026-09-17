"""Unit tests for configuration in timesfm_serve.core.config."""

from timesfm_serve.core.config import Settings, get_settings


def test_default_settings() -> None:
    settings = Settings()
    assert "timesfm" in settings.timesfm_model_id
    assert settings.device in ("auto", "cpu", "cuda")
    assert settings.default_horizon == 32
    assert settings.max_horizon == 512
    assert settings.batch_size == 32
    assert settings.port == 8088
    assert settings.normalize_inputs is True


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
    monkeypatch.setenv("DEFAULT_HORIZON", "64")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("DEVICE", "cpu")

    custom_settings = Settings()
    assert custom_settings.timesfm_model_id == "google/timesfm-2.5-200m-pytorch"
    assert custom_settings.default_horizon == 64
    assert custom_settings.port == 9090
    assert custom_settings.device == "cpu"


def test_get_settings_singleton() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_main_cli_execution(monkeypatch) -> None:
    import uvicorn

    from timesfm_serve.main import cli

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("sys.argv", ["timesfm-serve"])
    cli()
    assert len(calls) == 1
    assert calls[0][0][0] == "timesfm_serve.main:app"


def test_main_cli_mcp_flag(monkeypatch) -> None:
    import timesfm_serve.modules.mcp.server as mcp_mod
    from timesfm_serve.main import cli

    mcp_calls = []
    monkeypatch.setattr(mcp_mod, "run_stdio", lambda: mcp_calls.append(True))
    monkeypatch.setattr("sys.argv", ["timesfm-serve", "--mcp-stdio"])
    cli()
    assert len(mcp_calls) == 1

