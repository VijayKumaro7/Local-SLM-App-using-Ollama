# Local SLM Benchmark with Ollama

> Run small language models entirely offline via [Ollama](https://ollama.ai). Generate text with real-time streaming and live throughput metrics, benchmark multiple models on standardized prompts, and compare speed-vs-quality tradeoffs — all privately on your own hardware.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B">
  <img alt="Ollama" src="https://img.shields.io/badge/Inference-Ollama-black">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Supported Models](#supported-models)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Command-Line Benchmarking](#command-line-benchmarking)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Programmatic Use](#programmatic-use)
- [Constraint Analysis](#constraint-analysis)
- [Hardware Recommendations](#hardware-recommendations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**Local SLM Benchmark** is a Streamlit application for running and evaluating small language models on your own machine through Ollama. It provides an interactive playground for text generation, a standardized benchmark suite for measuring inference performance, and a side-by-side comparison view to help you pick the right model for your use case.

Because every model runs locally, no prompt or response ever leaves your hardware — making the tool well suited for sensitive data, offline environments, and cost-sensitive, high-volume inference.

### Why local inference?

| Dimension | Benefit |
|-----------|---------|
| **Privacy** | Prompts and responses never leave your machine — no third-party APIs. |
| **Cost** | No per-token billing; you pay only for hardware and electricity. |
| **Latency** | No network round-trips; response time is bound only by your hardware. |
| **Availability** | Works fully offline once models are downloaded. |

---

## Features

- **Interactive text generation** with **token-by-token streaming** output.
- **Accurate performance metrics** — throughput is computed from Ollama's real `eval_count` / `eval_duration`, not a word-count estimate.
- **Live connection status** — the sidebar shows whether Ollama is reachable and marks each model as installed (✅) or not pulled (⬇️).
- **One-click model pulling** — download a missing model directly from the UI.
- **Standardized benchmark suite** — evaluate every configured model on the same set of prompts and persist the results.
- **Comparison dashboard** — throughput, latency, quality ratings, and parameter counts side by side.
- **Graceful error handling** — clear, actionable messages when Ollama is down or a model isn't installed.
- **Configurable host** — point at a remote Ollama instance via the `OLLAMA_HOST` environment variable / secret.
- **CLI benchmarking** — run the full or a quick benchmark headlessly, no UI required.

---

## Screenshots

The app is a three-page Streamlit dashboard: **Generate**, **Benchmark**, and **Comparison**.

### Generate — Interactive Text Generation
Pick a model, enter a prompt, and watch the response stream in with live latency, token count, and throughput.

![Text generation page showing streamed output and performance metrics](output_images/generate.png)

### Benchmark — Run the Full Suite
Evaluate every configured model on the standardized prompt set and view per-model throughput and average latency.

![Benchmark suite results across models](output_images/benchmark.png)

### Comparison — Side-by-Side Tradeoffs
Compare speed, latency, quality ratings, and parameter counts in a single view.

![Model comparison table with speed, latency, and quality tradeoffs](output_images/comparison.png)

---

## Supported Models

Models are defined in [`config.py`](config.py). The two lightweight models are listed first and make excellent quick-start / low-resource options; the larger models deliver higher quality at the cost of speed and memory.

| Model | Parameters | Approx. Download | Relative Speed | Quality | Best For |
|-------|-----------|-----------------|----------------|---------|----------|
| **`qwen2.5:0.5b`** *(default)* | 0.5B | ~400 MB | Very Fast | 5/10 | Smoke tests, low-resource devices |
| **`llama3.2:1b`** | 1B | ~1.3 GB | Very Fast | 6/10 | Lightweight chat, quick demos |
| **`llama2:7b`** | 7B | ~3.8 GB | Fast | 7/10 | Real-time chat, responsive UX |
| **`mistral:7b`** | 7B | ~4.1 GB | Fast | 8/10 | Balanced tasks, code generation |
| **`llama2:13b`** | 13B | ~7.4 GB | Moderate | 9/10 | Complex reasoning, analysis |

> Quality ratings are qualitative guidance for relative comparison, not formal evaluation scores. Throughput depends heavily on your hardware (GPU vs. CPU).

To add or remove models, edit the `MODELS` dictionary in `config.py`.

---

## Getting Started

### Prerequisites

- [**Ollama**](https://ollama.ai) installed and running
- **Python 3.9+**
- Sufficient memory for your chosen models (a few hundred MB for the lightweight models; 8 GB+ RAM/VRAM recommended for 7B–13B models)

### Installation

1. **Start the Ollama server** (in its own terminal):
   ```bash
   ollama serve
   ```

2. **Pull at least one model.** The default is small and quick:
   ```bash
   ollama pull qwen2.5:0.5b
   ```
   Pull additional models as needed (`ollama pull mistral:7b`, `ollama pull llama2:13b`, …).

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```
   Then open <http://localhost:8501> in your browser.

> **Tip:** Run `python setup.py` to verify prerequisites (Ollama running, Python version) before starting.

---

## Usage

### Generate
- Select a model from the sidebar. The status indicator shows whether Ollama is connected and whether the model is installed; pull a missing model with one click.
- Enter a prompt and adjust **temperature** and **max tokens**.
- The response streams in token-by-token, followed by live metrics: elapsed time, tokens generated (real `eval_count`), and throughput (tokens/sec).

### Benchmark
- Runs every configured model against a fixed set of standardized prompts covering knowledge, reasoning, coding, creativity, Q&A, and summarization.
- Measures throughput, latency, and consistency, then persists results to `benchmark_results.json`.
- Depending on how many models are configured and your hardware, a full run can take several minutes.

### Comparison
- Presents throughput, average latency, quality ratings, and parameter counts side by side to help you identify the best model for your workload.

---

## Command-Line Benchmarking

Run benchmarks headlessly, without the Streamlit UI:

```bash
python benchmark_cli.py                    # Full benchmark across all configured models
python benchmark_cli.py --model llama2:7b  # Benchmark a single model
python benchmark_cli.py --quick            # Quick run on a reduced prompt set
```

Results are written to `benchmark_results.json` and printed as a ranked summary.

---

## Configuration

### Ollama host

By default the app connects to `http://localhost:11434`. Override it with the `OLLAMA_HOST` environment variable to target a remote or tunneled Ollama instance:

```bash
export OLLAMA_HOST="http://192.168.1.50:11434"
streamlit run app.py
```

### Models and prompts

- **Models:** edit the `MODELS` dictionary in `config.py`.
- **Benchmark prompts:** edit the `BENCHMARK_PROMPTS` list in `config.py`.

### App settings

Streamlit defaults live in [`.streamlit/config.toml`](.streamlit/config.toml).

---

## Deployment

The repository includes an entry point for [Streamlit Community Cloud](https://share.streamlit.io): **`streamlit_app.py`** (auto-detected), plus `.streamlit/config.toml`.

> **Important:** Streamlit Community Cloud has **no Ollama server and no GPU**, so the default `localhost:11434` is unreachable there. The UI will load, but generation requires an Ollama instance the cloud app can reach over the internet.

To enable live inference on a hosted deployment:

1. Run `ollama serve` on a machine you control and pull your models.
2. Expose it publicly with a tunnel (e.g. [ngrok](https://ngrok.com) or a Cloudflare Tunnel). Set `OLLAMA_HOST=0.0.0.0:11434` and `OLLAMA_ORIGINS=*` on the Ollama host so it accepts tunneled requests.
3. In **Streamlit Cloud → Settings → Secrets**, set the tunnel URL:
   ```toml
   OLLAMA_HOST = "https://your-tunnel-url.example"
   ```
4. Reboot the app. The sidebar should switch to **🟢 Ollama connected**.

> **Security note:** A public tunnel exposes your Ollama server to anyone with the URL. Restrict access (auth, IP allowlist) for anything beyond a short-lived demo.

---

## Project Structure

```
.
├── app.py                # Streamlit UI (Generate / Benchmark / Comparison)
├── streamlit_app.py      # Deployment entry point (runs app.py)
├── inference.py          # OllamaInference client: generate, streaming, model management
├── benchmark.py          # Benchmark runner and result persistence
├── benchmark_cli.py      # Headless CLI benchmark runner
├── config.py             # Models, prompts, and OLLAMA_HOST configuration
├── setup.py              # Prerequisite checker
├── requirements.txt      # Python dependencies
├── .streamlit/           # Streamlit configuration
└── output_images/        # Screenshots used in this README
```

---

## Programmatic Use

The `OllamaInference` client can be used directly in your own code:

```python
from inference import OllamaInference

client = OllamaInference()

# Non-streaming generation
result = client.generate(
    model="qwen2.5:0.5b",
    prompt="Explain AI in one sentence.",
    temperature=0.7,
    num_predict=50,
)
print(result["text"])
print(result["tokens_generated"], "tokens")

# Streaming generation
stats = {}
for chunk in client.generate_stream("qwen2.5:0.5b", "Write a haiku.", stats=stats):
    print(chunk, end="", flush=True)
print("\n", stats)  # {'tokens_generated': ..., 'eval_seconds': ...}
```

Helper methods include `is_available()`, `list_models()`, `has_model(name)`, and `pull_model(name)`.

---

## Constraint Analysis

**Privacy** — All models run locally; prompts and responses never leave your machine. Ideal for sensitive, medical, legal, or personal content.

**Latency** — No network round-trips; response time is bound by model size and hardware. Rough guidance:

| Setup | Typical throughput |
|-------|--------------------|
| 7B model on GPU | 20–40 tok/s |
| 7B model on CPU | 5–15 tok/s |
| 13B model on GPU | 10–20 tok/s |

**Cost** — No API fees. Costs are a one-time hardware investment plus electricity (~100–300 W during inference). For high-volume or latency-critical workloads, local inference typically pays for itself over time.

---

## Hardware Recommendations

| Hardware | 0.5B–1B | 7B models | 13B models |
|----------|---------|-----------|------------|
| **CPU only** | ✅ Fast | ✅ (5–10 tok/s) | ⚠️ Slow |
| **Integrated GPU** | ✅ Very fast | ✅ (15–25 tok/s) | ⚠️ Limited |
| **Dedicated GPU (8 GB+)** | ✅ Very fast | ✅ (25–35+ tok/s) | ✅ (12–20 tok/s) |

**Recommended:** an NVIDIA GPU with 8 GB+ VRAM for the 7B/13B models. On CPU-only or low-memory machines, stick to the lightweight `qwen2.5:0.5b` / `llama3.2:1b` models.

---

## Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| **"Cannot reach Ollama at http://localhost:11434"** | Ollama isn't running. Start it with `ollama serve`. On a hosted deployment, set the `OLLAMA_HOST` secret to a reachable instance (see [Deployment](#deployment)). |
| **Model shows "⬇️ not pulled"** | Pull it with `ollama pull <model>` or the sidebar's Pull button. |
| **Slow generation** | Expected on CPU. Use a smaller model or a GPU-backed host. |
| **`403 Forbidden` through a tunnel** | Set `OLLAMA_ORIGINS=*` (and `OLLAMA_HOST=0.0.0.0:11434`) on the Ollama host, then restart it. |

For more, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## License

Released under the **MIT License**.

---

<p align="center"><sub>Built with Streamlit and Ollama · Local-first, privacy-first inference</sub></p>
