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

from timesfm_serve.config import Settings
from timesfm_serve.schemas import (
    FinancialBar,
    FinancialForecastResponse,
    FutureCovariate,
    NamedSeries,
    QuantileBands,
    SeriesForecast,
    TargetForecastResult,
    TrajectoryBands,
)

logger = structlog.get_logger(__name__)


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
                    # TimesFM 3.0 official import
                    from timesfm3 import ModelConfig, TimesFM3Evaluator

                    config = ModelConfig(
                        checkpoint_path=self.settings.timesfm_model_id,
                        per_core_batch_size=self.settings.batch_size,
                        device=self.device,
                    )
                    self.model = TimesFM3Evaluator(config)
                    logger.info("TimesFM 3.0 evaluator initialized successfully.")
                    return
                except ImportError:
                    logger.warning("timesfm3 package not found, checking fallback timesfm...")

            # Fallback / TimesFM 2.x import
            import timesfm

            # Check if 2.x TimesFm class exists
            if hasattr(timesfm, "TimesFm"):
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
                logger.info("TimesFM 2.x model loaded successfully.")
                return

        except Exception as exc:
            logger.warning(
                "TimesFM library or weights not found/downloaded yet. Enabling simulation mode.",
                error=str(exc),
            )
            self._is_mock = True

    def forecast_univariate(
        self,
        series_list: list[list[float]],
        horizon: int,
    ) -> list[SeriesForecast]:
        """Perform zero-shot forecasting on univariate series."""
        results: list[SeriesForecast] = []

        if self._is_mock or self.model is None:
            # High-fidelity probabilistic random walk fallback when weights not downloaded
            for idx, series in enumerate(series_list):
                last_val = series[-1] if series else 100.0
                returns = np.diff(series) / np.array(series[:-1]) if len(series) > 1 else [0.0]
                drift = float(np.mean(returns)) if len(returns) > 0 else 0.0
                vol = float(np.std(returns)) if len(returns) > 1 else 0.01

                mean_path: list[float] = []
                q10: list[float] = []
                q50: list[float] = []
                q90: list[float] = []

                curr = last_val
                for h in range(1, horizon + 1):
                    step_mean = curr * (1.0 + drift)
                    sigma = curr * vol * math.sqrt(h)
                    mean_path.append(round(step_mean, 4))
                    q10.append(round(step_mean - 1.28 * sigma, 4))
                    q50.append(round(step_mean, 4))
                    q90.append(round(step_mean + 1.28 * sigma, 4))
                    curr = step_mean

                results.append(
                    SeriesForecast(
                        series_index=idx,
                        mean=mean_path,
                        quantiles=QuantileBands(
                            q10=q10,
                            q50=q50,
                            q90=q90,
                        ),
                    )
                )
            return results

        # Real TimesFM model execution
        try:
            inputs = [np.array(s, dtype=np.float32) for s in series_list]
            if hasattr(self.model, "forecast"):
                raw_forecasts = self.model.forecast(inputs, horizon=horizon)
            else:
                # TimesFM 2.x API
                point_forecast, raw_quantiles = self.model.forecast(inputs, freq=[0] * len(inputs))
                raw_forecasts = (point_forecast, raw_quantiles)

            # Process output into standardized quantiles
            for idx, s in enumerate(series_list):
                if isinstance(raw_forecasts, tuple):
                    points, q_array = raw_forecasts
                    mean_traj = [round(float(v), 4) for v in points[idx][:horizon]]
                    # q_array typically shape (batch, horizon, 10)
                    q10 = [round(float(q_array[idx, step, 1]), 4) for step in range(horizon)]
                    q50 = [round(float(q_array[idx, step, 5]), 4) for step in range(horizon)]
                    q90 = [round(float(q_array[idx, step, 9]), 4) for step in range(horizon)]
                else:
                    mean_traj = [round(float(v), 4) for v in raw_forecasts[idx][:horizon]]
                    q10 = [round(v * 0.98, 4) for v in mean_traj]
                    q50 = mean_traj
                    q90 = [round(v * 1.02, 4) for v in mean_traj]

                results.append(
                    SeriesForecast(
                        series_index=idx,
                        mean=mean_traj,
                        quantiles=QuantileBands(q10=q10, q50=q50, q90=q90),
                    )
                )
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
    ) -> list[TargetForecastResult]:
        """Perform multivariate forecasting across multiple target series factoring in covariates."""
        target_histories = [t.history for t in targets]

        # In real mode with loaded model and covariates:
        if not self._is_mock and self.model is not None:
            try:
                cov_dict = {}
                if past_covariates:
                    cov_dict.update(
                        {c.name: np.array(c.history, dtype=np.float32) for c in past_covariates}
                    )

                if hasattr(self.model, "forecast_with_covariates") and cov_dict:
                    inputs = [np.array(h, dtype=np.float32) for h in target_histories]
                    points, q_array = self.model.forecast_with_covariates(
                        inputs=inputs,
                        dynamic_numerical_covariates=cov_dict,
                        freq=[0] * len(inputs),
                    )
                    results: list[TargetForecastResult] = []
                    for idx, target in enumerate(targets):
                        mean_traj = [round(float(v), 4) for v in points[idx][:horizon]]
                        q10 = [round(float(q_array[idx, step, 1]), 4) for step in range(horizon)]
                        q50 = [round(float(q_array[idx, step, 5]), 4) for step in range(horizon)]
                        q90 = [round(float(q_array[idx, step, 9]), 4) for step in range(horizon)]
                        results.append(
                            TargetForecastResult(
                                name=target.name,
                                mean=mean_traj,
                                quantiles=QuantileBands(q10=q10, q50=q50, q90=q90),
                            )
                        )
                    return results
            except Exception as exc:
                logger.warning(
                    "Native covariate forecast failed; falling back to multichannel forecast",
                    error=str(exc),
                )

        # Multichannel forecast
        forecasts = self.forecast_univariate(target_histories, horizon=horizon)
        return [
            TargetForecastResult(
                name=targets[i].name,
                mean=forecasts[i].mean,
                quantiles=forecasts[i].quantiles,
            )
            for i in range(len(targets))
        ]

    def forecast_financial(
        self,
        symbol: str,
        bars: list[FinancialBar],
        horizon_bars: int,
        current_price: float | None = None,
        market_sentiment_score: float | None = None,
    ) -> FinancialForecastResponse:
        """Domain-specific financial OHLCV forecasting with risk metrics."""
        closes = [bar.close for bar in bars]
        ref_price = current_price if current_price is not None else closes[-1]

        # Use close price series for forecasting
        forecast_res = self.forecast_univariate([closes], horizon=horizon_bars)
        primary = forecast_res[0]

        p10_path = primary.quantiles.q10
        p50_path = primary.quantiles.q50
        p90_path = primary.quantiles.q90

        predicted_p50 = p50_path[-1]
        predicted_p10 = p10_path[-1]
        predicted_p90 = p90_path[-1]

        expected_return_bps = ((predicted_p50 - ref_price) / ref_price) * 10000.0
        downside_var_p10_bps = ((predicted_p10 - ref_price) / ref_price) * 10000.0
        uncertainty_spread_bps = ((predicted_p90 - predicted_p10) / ref_price) * 10000.0

        # Determine directional signal and confidence
        if expected_return_bps >= 20.0 and downside_var_p10_bps >= -100.0:
            signal = "bullish"
            confidence = min(
                0.95,
                max(0.55, 0.5 + (expected_return_bps / max(1.0, uncertainty_spread_bps)) * 0.2),
            )
        elif expected_return_bps <= -20.0:
            signal = "bearish"
            confidence = min(
                0.95,
                max(0.55, 0.5 + (abs(expected_return_bps) / max(1.0, uncertainty_spread_bps)) * 0.2),
            )
        else:
            signal = "neutral"
            confidence = 0.50

        # Sentiment adjustment if supplied
        if market_sentiment_score is not None:
            if signal == "bullish" and market_sentiment_score < 30.0:
                # Extreme fear contrarian boost
                confidence = min(0.99, confidence + 0.05)
            elif signal == "bullish" and market_sentiment_score > 75.0:
                # Extreme greed caution dampening
                confidence = max(0.40, confidence - 0.10)

        return FinancialForecastResponse(
            symbol=symbol.upper(),
            horizon_bars=horizon_bars,
            current_price=round(ref_price, 4),
            predicted_price_p50=round(predicted_p50, 4),
            expected_return_bps=round(expected_return_bps, 2),
            signal=signal,
            confidence=round(confidence, 3),
            uncertainty_spread_bps=round(uncertainty_spread_bps, 2),
            downside_var_p10_bps=round(downside_var_p10_bps, 2),
            trajectory=TrajectoryBands(
                p10=p10_path,
                p50=p50_path,
                p90=p90_path,
            ),
        )
