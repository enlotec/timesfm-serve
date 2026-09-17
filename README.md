<div align="center">

# ⚡ TimesFM Serve

**Production-grade, lightweight REST API and Docker serving container for Google's TimesFM (Time Series Foundation Model).**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/google/timesfm-3.0-pytorch)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)

[Features](#-key-features) •
[Quickstart](#-quickstart) •
[Docker Deployment](#-docker-deployment) •
[API Reference](#-api-reference) •
[Model Context Protocol (MCP)](#-model-context-protocol-mcp) •
[Configuration](#-configuration) •
[Licensing & Legal](#-licensing--legal-notice)

</div>

---

## 📖 Overview

Google Research's **TimesFM** (Time Series Foundation Model) is a state-of-the-art decoder-only foundation model pre-trained on over **1 trillion time points**. It delivers top-tier zero-shot forecasting performance across diverse domains (finance, retail demand, energy, IoT, traffic) without requiring task-specific fine-tuning.

However, Google does not provide an official, ready-to-run public Docker container on Docker Hub or GitHub Container Registry for self-hosted deployments.

**TimesFM Serve** bridges this gap. It packages Google's TimesFM into a high-performance, containerized microservice exposing clean RESTful endpoints, OpenAPI documentation, and full probabilistic quantile uncertainty bands ($q_{10}$ to $q_{90}$) for both CPU and NVIDIA GPU hardware.

---

## ✨ Key Features

- **Zero-Shot Forecasting**: Predict any time series without training, fine-tuning, or parameter search.
- **TimesFM 3.0 & 2.x Support**: Seamlessly run the latest multivariate TimesFM 3.0 or the permissive Apache-2.0 TimesFM 2.5 / 2.0 checkpoints.
- **Multivariate & Covariate Support**: Jointly forecast multi-channel series with dynamic past covariates (volume, technical indicators) and future known events (market hours, holidays).
- **Probabilistic Quantile Forecasts**: Outputs 9 quantiles ($q_{10}, q_{20}, \dots, q_{90}$) at every forecast step for complete uncertainty estimation and risk budgeting.
- **Model Context Protocol (MCP) Server**: Built-in MCP endpoints allowing AI agents (like Claude Desktop) to natively execute zero-shot forecasting directly on your data.
- **Multi-Arch Docker Images**: Pre-configured Docker setups for both lightweight CPU execution and CUDA-accelerated GPU inference.
- **Persistent Model Caching**: Hugging Face checkpoints are cached in a persistent volume—download once, start in seconds thereafter.
- **Observability & Healthchecks**: Native `/health` readiness probes and `/metrics` instrumentation.

---

## 🚀 Quickstart

### Option 1: Run with Docker (Recommended)

#### CPU Mode (Lightweight)
```bash
docker run -d \
  --name timesfm-serve \
  -p 8088:8088 \
  -v timesfm_cache:/root/.cache/huggingface \
  -e TIMESFM_MODEL_ID=google/timesfm-3.0-pytorch \
  ghcr.io/enlotec/timesfm-serve:latest
```

#### NVIDIA GPU Mode (CUDA Acceleration)
```bash
docker run -d \
  --name timesfm-serve \
  --gpus all \
  -p 8088:8088 \
  -v timesfm_cache:/root/.cache/huggingface \
  -e TIMESFM_MODEL_ID=google/timesfm-3.0-pytorch \
  -e DEVICE=cuda \
  ghcr.io/enlotec/timesfm-serve:latest-gpu
```

Interactive API documentation will be available at **`http://localhost:8088/docs`**.

---

### Option 2: Docker Compose

Clone this repository and launch with Docker Compose:

```bash
git clone https://github.com/enlotec/timesfm-serve.git
cd timesfm-serve

# Start in CPU mode
docker compose up -d

# Or start with NVIDIA GPU acceleration
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
```

---

### Option 3: Local Python Environment

```bash
git clone https://github.com/your-username/timesfm-serve.git
cd timesfm-serve

# Create virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run the server
uvicorn timesfm_serve.main:app --host 0.0.0.0 --port 8088 --reload
```

---

## 📡 API Reference

### 1. Univariate Forecast (`POST /v1/forecast`)

Forecast single or batch time series. Returns point forecast (mean/median) along with all 9 quantiles.

#### Request:
```bash
curl -X POST "http://localhost:8088/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [
      [102.5, 103.1, 102.8, 104.2, 105.0, 104.7, 106.1, 107.0, 106.8, 108.2]
    ],
    "horizon": 5,
    "frequency": "hourly"
  }'
```

#### Response:
```json
{
  "model_id": "google/timesfm-3.0-pytorch",
  "horizon": 5,
  "forecasts": [
    {
      "series_index": 0,
      "mean": [108.65, 109.12, 109.45, 109.80, 110.15],
      "quantiles": {
        "q10": [107.20, 107.45, 107.60, 107.85, 108.05],
        "q50": [108.60, 109.10, 109.40, 109.78, 110.10],
        "q90": [110.10, 110.80, 111.30, 111.75, 112.25]
      }
    }
  ]
}
```

---

### 2. Multivariate & Covariates (`POST /v1/forecast/multivariate`)

Forecast multiple correlated channels jointly using TimesFM 3.0's native cross-variate attention, factoring in past and future covariates.

#### Request:
```bash
curl -X POST "http://localhost:8088/v1/forecast/multivariate" \
  -H "Content-Type: application/json" \
  -d '{
    "targets": [
      {
        "name": "channel_load",
        "history": [45.2, 47.1, 49.3, 52.0, 50.8, 53.4, 55.1]
      }
    ],
    "past_covariates": [
      {
        "name": "temperature",
        "history": [21.0, 21.5, 22.1, 23.0, 22.8, 23.4, 24.1]
      }
    ],
    "horizon": 3
  }'
```

---

### 3. Model Context Protocol (MCP)

TimesFM Serve includes a built-in Anthropic **Model Context Protocol (MCP)** server exposed via Server-Sent Events (SSE). This allows AI agents to directly use Google's TimesFM to forecast data without needing to write code.

The MCP server exposes two tools:
1. `forecast_univariate`
2. `forecast_multivariate`

#### Connecting an AI Agent
Point your MCP-compatible client (e.g., Claude Desktop or Cursor) to the SSE endpoints:
- **SSE Transport**: `http://localhost:8088/mcp/sse`
- **Messages Route**: `http://localhost:8088/mcp/messages`

---

### 4. Health & Diagnostics (`GET /health`)

```bash
curl "http://localhost:8088/health"
```

```json
{
  "status": "healthy",
  "model_id": "google/timesfm-3.0-pytorch",
  "device": "cuda:0",
  "torch_version": "2.4.0",
  "multivariate_enabled": true
}
```

---

## 🐍 Python Client Example

You can easily call **TimesFM Serve** from any Python codebase (e.g. trading bots, backtesting scripts, data pipelines):

```python
import httpx

client = httpx.Client(base_url="http://localhost:8088", timeout=30.0)

# 1. Health check
health = client.get("/health").json()
print("Connected to:", health["model_id"], "on", health["device"])

# 2. Simple forecast
response = client.post("/v1/forecast", json={
    "series": [[10.0, 10.5, 11.2, 10.8, 11.9, 12.3, 12.1]],
    "horizon": 4
})
result = response.json()
print("Expected next 4 steps:", result["forecasts"][0]["mean"])
print("10th percentile (lower bound):", result["forecasts"][0]["quantiles"]["q10"])
```

---

## ⚙️ Configuration

Configure the server via environment variables in `.env` or Docker:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TIMESFM_MODEL_ID` | `google/timesfm-3.0-pytorch` | Hugging Face model repository ID |
| `DEVICE` | `auto` | Execution device (`auto`, `cuda`, `cpu`) |
| `DEFAULT_HORIZON` | `32` | Default forecast horizon length if omitted |
| `MAX_HORIZON` | `512` | Upper limit for forecast horizon |
| `BATCH_SIZE` | `32` | Max inference batch size |
| `NORMALIZE_INPUTS` | `true` | Apply internal z-score / instance scaling |
| `HOST` | `0.0.0.0` | Bind host address |
| `PORT` | `8088` | Bind port |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warning`) |
| `HF_TOKEN` | *(None)* | Optional Hugging Face token for rate limits |

---

## ⚖️ Legal Disclaimers & Limitation of Liability

### 1. Software License & Limitation of Liability
The software source code of **TimesFM Serve** is licensed under the **Apache License, Version 2.0**.

```text
Copyright 2026 enlotec and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**EXPRESS DISCLAIMER OF WARRANTIES AND DAMAGES:**
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NONINFRINGEMENT. IN NO EVENT SHALL `ENLOTEC`, ITS AFFILIATES, OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, NEGLIGENCE, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### 2. Google TimesFM Model Weights & Intellectual Property
- **No Weight Redistribution:** Neither `enlotec` nor this repository hosts, mirrors, sells, or redistributes Google's pre-trained model weights. Model checkpoints are fetched on-demand directly from the public [Hugging Face Model Hub](https://huggingface.co/google) by the user's local container instance.
- **Model Licensing Terms:** End-users are solely responsible for verifying and complying with the respective model checkpoint licenses:
  - **TimesFM 3.0 (`google/timesfm-3.0-pytorch`)**: Released by Google under the **`timesfm-non-commercial-license-v1.0`** (restricted to non-commercial research, academic study, and personal evaluation).
  - **TimesFM 2.5 / 2.0 (`google/timesfm-2.0-500m-pytorch`)**: Released by Google under the **Apache License, Version 2.0** (permissive for commercial and production use).
  - To ensure commercial compliance, set `TIMESFM_MODEL_ID=google/timesfm-2.0-500m-pytorch`.
- **Third-Party Trademarks:** *Google*, *TimesFM*, *Vertex AI*, and *BigQuery* are trademarks or registered trademarks of Google LLC. This project is an independent open-source contribution developed by `enlotec` and is not affiliated with, sponsored by, or endorsed by Google LLC.

---

## 🤝 Contributing

Contributions are warmly welcomed! Please submit an issue or pull request:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 Acknowledgments

- Google Research for developing and open-sourcing the [TimesFM](https://github.com/google-research/timesfm) architecture.
- The Hugging Face team for hosting model checkpoints and transformer infrastructure.
