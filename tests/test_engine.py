"""Unit tests for TimesFM engine in timesfm_serve.core.engine."""

import numpy as np
import pytest

from timesfm_serve.core.config import Settings
from timesfm_serve.core.engine import (
    TimesFmEngine,
    _extract_quantile_bands,
    get_global_engine,
    map_frequency,
    set_global_engine,
)
from timesfm_serve.modules.forecasting.schemas import FutureCovariate, NamedSeries


def test_map_frequency() -> None:
    assert map_frequency(None) == 0
    assert map_frequency(0) == 0
    assert map_frequency(2) == 2
    assert map_frequency("hourly") == 0
    assert map_frequency("15m") == 0
    assert map_frequency("daily") == 1
    assert map_frequency("weekly") == 2
    assert map_frequency("monthly") == 3
    assert map_frequency("quarterly") == 4
    assert map_frequency("yearly") == 5
    assert map_frequency("unknown_string") == 0


def test_device_resolution() -> None:
    assert TimesFmEngine._resolve_device("cpu") == "cpu"
    # Auto will return cpu or cuda depending on host
    assert TimesFmEngine._resolve_device("auto") in ("cpu", "cuda")


def test_extract_quantile_bands_nine() -> None:
    # Shape: (horizon=3, quantiles=9)
    matrix = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
    ])
    bands = _extract_quantile_bands(matrix, horizon=3)
    assert bands.q10 == [0.1, 1.1, 2.1]
    assert bands.q50 == [0.5, 1.5, 2.5]
    assert bands.q90 == [0.9, 1.9, 2.9]


def test_extract_quantile_bands_ten() -> None:
    # Shape: (horizon=2, quantiles=10) where index 0 is point forecast and 1..9 are quantiles
    matrix = np.array([
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
    ])
    bands = _extract_quantile_bands(matrix, horizon=2)
    assert bands.q10 == [0.1, 1.1]
    assert bands.q50 == [0.5, 1.5]
    assert bands.q90 == [0.9, 1.9]


def test_mock_engine_univariate() -> None:
    settings = Settings(device="cpu", default_horizon=3)
    engine = TimesFmEngine(settings)
    engine._is_mock = True

    series = [[10.0, 10.5, 11.0, 11.5]]
    results = engine.forecast_univariate(series, horizon=3, frequency="daily")
    assert len(results) == 1
    assert len(results[0].mean) == 3
    assert len(results[0].quantiles.q10) == 3
    assert len(results[0].quantiles.q50) == 3
    assert len(results[0].quantiles.q90) == 3
    # Check monotonicity of quantiles: q10 <= q50 <= q90
    for h in range(3):
        assert results[0].quantiles.q10[h] <= results[0].quantiles.q50[h] <= results[0].quantiles.q90[h]


def test_mock_engine_multivariate() -> None:
    settings = Settings(device="cpu")
    engine = TimesFmEngine(settings)
    engine._is_mock = True

    targets = [
        NamedSeries(name="load_1", history=[100.0, 102.0, 104.0]),
        NamedSeries(name="load_2", history=[200.0, 201.0, 203.0]),
    ]
    past_covs = [
        NamedSeries(name="temp", history=[22.0, 22.5, 23.0]),
    ]
    future_covs = [
        FutureCovariate(name="cloud", future_values=[0.1, 0.2]),
    ]

    results = engine.forecast_multivariate(
        targets=targets,
        past_covariates=past_covs,
        future_covariates=future_covs,
        horizon=2,
        frequency="hourly",
    )
    assert len(results) == 2
    assert results[0].name == "load_1"
    assert len(results[0].mean) == 2
    assert results[1].name == "load_2"
    assert len(results[1].mean) == 2


def test_global_engine_lifecycle() -> None:
    settings = Settings(device="cpu")
    engine = TimesFmEngine(settings)
    set_global_engine(engine)
    assert get_global_engine() is engine


def test_global_engine_uninitialized_raises() -> None:
    import timesfm_serve.core.engine as eng
    eng._global_engine = None
    with pytest.raises(RuntimeError, match="not initialized"):
        get_global_engine()


def test_cuda_fallback_resolution(monkeypatch) -> None:
    import timesfm_serve.core.engine as eng
    monkeypatch.setattr(eng.torch.cuda, "is_available", lambda: False)
    assert TimesFmEngine._resolve_device("cuda") == "cpu"

