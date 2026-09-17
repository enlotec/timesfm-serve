"""Inference engine managing TimesFM model loading and forecast execution."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import structlog

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

from timesfm_serve.core.config import Settings
from timesfm_serve.modules.forecasting.schemas import (
    FutureCovariate,
    NamedSeries,
    QuantileBands,
    SeriesForecast,
    TargetForecastResult,
)

logger = structlog.get_logger(__name__)

# Standard z-multipliers for 9 quantile percentiles:
# [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
_Z_QUANTILES = [
    -1.2816,  # q10
    -0.8416,  # q20
    -0.5244,  # q30
    -0.2533,  # q40
    0.0000,   # q50
    0.2533,   # q60
    0.5244,   # q70
    0.8416,   # q80
    1.2816,   # q90
]


def map_frequency(freq: str | int | None) -> int:
    """Map string frequency alias to TimesFM frequency integer identifier.

    0: high frequency (<= hourly, minutely, seconds)
    1: daily
    2: weekly
    3: monthly
    4: quarterly
    5: yearly
    """
    if freq is None:
        return 0
    if isinstance(freq, int):
        return max(0, min(5, freq))
    f_clean = str(freq).strip().lower()
    if f_clean in ("0", "high", "hourly", "1h", "minutely", "1m", "15m", "30m", "second", "s"):
        return 0
    elif f_clean in ("1", "daily", "day", "1d", "d"):
        return 1
    elif f_clean in ("2", "weekly", "week", "1w", "w"):
        return 2
    elif f_clean in ("3", "monthly", "month", "1m", "m"):
        return 3
    elif f_clean in ("4", "quarterly", "quarter", "1q", "q"):
        return 4
    elif f_clean in ("5", "yearly", "annual", "1y", "y"):
        return 5
    return 0


def _extract_quantile_bands(q_matrix: np.ndarray, horizon: int) -> QuantileBands:
    """Extract all 9 quantile bands (q10 to q90) from a quantile step matrix.

    Accepts shape (horizon, num_quantiles) where num_quantiles is 9 (TimesFM 3.0)
    or 10 (TimesFM 2.x where index 0 is median/point forecast and indices 1..9 are quantiles).
    """
    num_q = q_matrix.shape[-1]
    if num_q == 9:
        # TimesFM 3.0: quantiles are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_cols = list(range(9))
    elif num_q >= 10:
        # TimesFM 2.x: index 0 is point/median, indices 1..9 are 0.1..0.9
        q_cols = list(range(1, 10))
    else:
        # Fallback for fewer quantiles
        q_cols = [0] * 9

    def get_band(col_idx: int) -> list[float]:
        return [round(float(q_matrix[step, col_idx]), 4) for step in range(horizon)]

    return QuantileBands(
        q10=get_band(q_cols[0]),
        q20=get_band(q_cols[1]),
        q30=get_band(q_cols[2]),
        q40=get_band(q_cols[3]),
        q50=get_band(q_cols[4]),
        q60=get_band(q_cols[5]),
        q70=get_band(q_cols[6]),
        q80=get_band(q_cols[7]),
        q90=get_band(q_cols[8]),
    )


class TimesFmEngine:
    """Manages TimesFM foundation model loading, tensor conversion, and inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.model: Any = None
        self.is_v3: bool = "3.0" in settings.timesfm_model_id
        self._is_mock: bool = not HAS_TORCH

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    def load_model(self) -> None:
        """Load model weights from Hugging Face checkpoint or local directory."""
        logger.info(
            "Loading TimesFM model",
            model_id=self.settings.timesfm_model_id,
            device=self.device,
        )
        if self.settings.hf_token:
            os.environ["HF_TOKEN"] = self.settings.hf_token

        try:
            if self.is_v3:
                try:
                    # TimesFM 3.0 official evaluator / forecaster
                    from timesfm3 import ModelConfig, TimesFM3Evaluator

                    config = ModelConfig(
                        checkpoint_path=self.settings.timesfm_model_id,
                        per_core_batch_size=self.settings.batch_size,
                        device=self.device,
                    )
                    self.model = TimesFM3Evaluator(config)
                    self._is_mock = False
                    logger.info("TimesFM 3.0 evaluator initialized successfully.")
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("timesfm3 initialization failed, checking fallback timesfm...", error=str(exc))

            # Fallback / TimesFM 2.x import
            import timesfm

            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                # TimesFM 2.5 PyTorch
                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    self.settings.timesfm_model_id,
                    device=self.device,
                )
                self._is_mock = False
                logger.info("TimesFM 2.5 model loaded successfully.")
                return
            elif hasattr(timesfm, "TimesFm"):
                # TimesFM 2.0
                self.model = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend=self.device,
                        per_core_batch_size=self.settings.batch_size,
                        horizon_len=self.settings.default_horizon,
                    ),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id=self.settings.timesfm_model_id
                    ),
                )
                self._is_mock = False
                logger.info("TimesFM 2.0 model loaded successfully.")
                return

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TimesFM library or weights not found/downloaded yet. Enabling simulation mode.",
                error=str(exc),
            )
            self._is_mock = True

    def forecast_univariate(
        self,
        series_list: list[list[float]],
        horizon: int,
        frequency: str | int | None = None,
    ) -> list[SeriesForecast]:
        """Perform zero-shot forecasting on univariate series with all 9 quantile bands."""
        freq_code = map_frequency(frequency)
        results: list[SeriesForecast] = []

        if self._is_mock or self.model is None:
            # High-fidelity probabilistic simulation fallback when weights not downloaded
            for idx, series in enumerate(series_list):
                last_val = series[-1] if series else 100.0
                returns = np.diff(series) / np.array(series[:-1]) if len(series) > 1 else [0.0]
                drift = float(np.mean(returns)) if len(returns) > 0 else 0.0
                vol = float(np.std(returns)) if len(returns) > 1 else 0.01

                mean_path: list[float] = []
                q_bands: dict[str, list[float]] = {f"q{p*10}": [] for p in range(1, 10)}

                curr = last_val
                for h in range(1, horizon + 1):
                    step_mean = curr * (1.0 + drift)
                    sigma = curr * vol * math.sqrt(h)
                    mean_path.append(round(step_mean, 4))
                    for p_idx, z in enumerate(_Z_QUANTILES, start=1):
                        q_bands[f"q{p_idx * 10}"].append(round(step_mean + z * sigma, 4))
                    curr = step_mean

                results.append(
                    SeriesForecast(
                        series_index=idx,
                        mean=mean_path,
                        quantiles=QuantileBands(**q_bands),
                    )
                )
            return results

        # Real TimesFM model execution
        try:
            inputs = [np.array(s, dtype=np.float32) for s in series_list]

            # TimesFM 3.0 API
            if hasattr(self.model, "predict_batch"):
                outs = list(
                    self.model.predict_batch(
                        contexts=inputs,
                        horizon=horizon,
                        return_quantiles=True,
                        univariate=True,
                    )
                )
                for idx, out in enumerate(outs):
                    f_pts = [round(float(v), 4) for v in out.forecast[:horizon]]
                    if out.quantiles is not None:
                        quantiles = _extract_quantile_bands(out.quantiles[:horizon], horizon)
                    else:
                        quantiles = QuantileBands(
                            q10=[round(v * 0.95, 4) for v in f_pts],
                            q50=f_pts,
                            q90=[round(v * 1.05, 4) for v in f_pts],
                        )
                    results.append(
                        SeriesForecast(
                            series_index=idx,
                            mean=f_pts,
                            quantiles=quantiles,
                        )
                    )
                return results

            # TimesFM 2.x API
            if hasattr(self.model, "forecast"):
                point_forecast, raw_quantiles = self.model.forecast(
                    inputs,
                    freq=[freq_code] * len(inputs),
                )
                for idx in range(len(series_list)):
                    pts = [round(float(v), 4) for v in point_forecast[idx][:horizon]]
                    q_matrix = raw_quantiles[idx][:horizon]
                    quantiles = _extract_quantile_bands(q_matrix, horizon)
                    results.append(
                        SeriesForecast(
                            series_index=idx,
                            mean=pts,
                            quantiles=quantiles,
                        )
                    )
                return results

        except Exception as exc:
            logger.error("Forecast execution failed", error=str(exc))
            raise RuntimeError(f"TimesFM inference failed: {exc}") from exc

        return results

    def forecast_multivariate(
        self,
        targets: list[NamedSeries],
        past_covariates: list[NamedSeries] | None,
        future_covariates: list[FutureCovariate] | None,
        horizon: int,
        frequency: str | int | None = None,
    ) -> list[TargetForecastResult]:
        """Perform multivariate forecasting across multiple target series factoring in dynamic covariates."""
        freq_code = map_frequency(frequency)
        target_histories = [t.history for t in targets]
        context_len = len(targets[0].history)

        # In real mode with loaded model:
        if not self._is_mock and self.model is not None:
            try:
                # TimesFM 3.0 Multivariate execution
                if hasattr(self.model, "predict_batch"):
                    target_matrix = np.stack(
                        [np.array(t.history, dtype=np.float32) for t in targets],
                        axis=0,
                    )

                    # Build past-only covariates matrix
                    po_matrix = None
                    if past_covariates:
                        po_arrs = [np.array(c.history, dtype=np.float32) for c in past_covariates]
                        po_matrix = np.stack(po_arrs, axis=0)

                    # Build past-future covariates matrix (must span context_len + horizon)
                    pf_matrix = None
                    if future_covariates:
                        past_map = {c.name: c.history for c in (past_covariates or [])}
                        pf_arrs = []
                        for fc in future_covariates:
                            # Context history
                            if fc.history is not None:
                                past_part = fc.history[-context_len:]
                            elif fc.name in past_map:
                                past_part = past_map[fc.name][-context_len:]
                            else:
                                # Pad with first future value if historical context not provided
                                past_part = [fc.future_values[0]] * context_len

                            fut_part = fc.future_values[:horizon]
                            full_cov = np.array(list(past_part) + list(fut_part), dtype=np.float32)
                            pf_arrs.append(full_cov)

                        if pf_arrs:
                            pf_matrix = np.stack(pf_arrs, axis=0)

                    outs = list(
                        self.model.predict_batch(
                            contexts=[target_matrix],
                            horizon=horizon,
                            past_only_covariates=[po_matrix] if po_matrix is not None else None,
                            past_future_covariates=[pf_matrix] if pf_matrix is not None else None,
                            return_quantiles=True,
                            univariate=False,
                        )
                    )
                    out = outs[0]
                    results: list[TargetForecastResult] = []
                    for idx, target in enumerate(targets):
                        f_pts = [round(float(v), 4) for v in out.forecast[idx][:horizon]]
                        if out.quantiles is not None:
                            quantiles = _extract_quantile_bands(out.quantiles[idx][:horizon], horizon)
                        else:
                            quantiles = QuantileBands(
                                q10=[round(v * 0.95, 4) for v in f_pts],
                                q50=f_pts,
                                q90=[round(v * 1.05, 4) for v in f_pts],
                            )
                        results.append(
                            TargetForecastResult(
                                name=target.name,
                                mean=f_pts,
                                quantiles=quantiles,
                            )
                        )
                    return results

                # TimesFM 2.x native covariate execution
                if hasattr(self.model, "forecast_with_covariates"):
                    cov_dict = {}
                    if past_covariates:
                        for c in past_covariates:
                            # In 2.x, dynamic numerical covariates must be context_len + horizon
                            last_val = c.history[-1]
                            extended = list(c.history) + [last_val] * horizon
                            cov_dict[c.name] = np.array(extended, dtype=np.float32)

                    if future_covariates:
                        past_map = {c.name: c.history for c in (past_covariates or [])}
                        for fc in future_covariates:
                            past_part = (
                                fc.history
                                if fc.history is not None
                                else past_map.get(fc.name, [fc.future_values[0]] * context_len)
                            )
                            cov_dict[fc.name] = np.array(
                                list(past_part) + list(fc.future_values[:horizon]),
                                dtype=np.float32,
                            )

                    inputs = [np.array(h, dtype=np.float32) for h in target_histories]
                    points, q_array = self.model.forecast_with_covariates(
                        inputs=inputs,
                        dynamic_numerical_covariates=cov_dict,
                        freq=[freq_code] * len(inputs),
                    )
                    results = []
                    for idx, target in enumerate(targets):
                        f_pts = [round(float(v), 4) for v in points[idx][:horizon]]
                        quantiles = _extract_quantile_bands(q_array[idx][:horizon], horizon)
                        results.append(
                            TargetForecastResult(
                                name=target.name,
                                mean=f_pts,
                                quantiles=quantiles,
                            )
                        )
                    return results

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Native covariate forecast failed; falling back to multichannel forecast",
                    error=str(exc),
                )

        # Fallback multichannel forecast
        forecasts = self.forecast_univariate(target_histories, horizon=horizon, frequency=frequency)
        return [
            TargetForecastResult(
                name=targets[i].name,
                mean=forecasts[i].mean,
                quantiles=forecasts[i].quantiles,
            )
            for i in range(len(targets))
        ]


_global_engine: TimesFmEngine | None = None


def set_global_engine(engine: TimesFmEngine) -> None:
    global _global_engine
    _global_engine = engine


def get_global_engine() -> TimesFmEngine:
    if _global_engine is None:
        raise RuntimeError("TimesFM Engine not initialized globally.")
    return _global_engine
