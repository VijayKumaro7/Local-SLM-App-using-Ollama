# Architecture Overview

## Project Structure

```
E:\Local SLM\
├── app.py                      # Main Streamlit web UI
├── benchmark.py                # Benchmark suite and utilities
├── benchmark_cli.py            # Command-line benchmark runner
├── inference.py                # Ollama API client
├── config.py                   # Model definitions and settings
├── setup.py                    # Setup and verification script
│
├── README.md                   # Full documentation
├── QUICKSTART.md               # Quick setup guide
├── ARCHITECTURE.md             # This file
│
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
│
└── benchmark_results.json      # Generated benchmark data (after running)
```

## Core Components

### 1. `config.py` — Configuration
**Purpose**: Centralized model definitions and benchmark setup

**Key exports**:
- `MODELS`: Dictionary of models with metadata (params, speed, quality)
- `BENCHMARK_PROMPTS`: 13 standardized prompts for evaluation
- `CONSTRAINTS`: Documentation of privacy/latency/cost constraints

**Usage**: Imported by all other modules for consistent configuration

### 2. `inference.py` — Ollama Client
**Purpose**: Wrapper around Ollama HTTP API for text generation

**Class**: `OllamaInference`
- `generate()`: Generate text from a prompt with configurable parameters
- `list_models()`: Check available models
- `pull_model()`: Download models from registry

**Design choice**: Simple wrapper instead of direct `ollama` Python library for flexibility and explicit error handling

### 3. `benchmark.py` — Benchmark Suite
**Purpose**: Run standardized evaluation across all models

**Key functions**:
- `run_benchmark()`: Full benchmark on all models with all prompts
- `save_benchmark_results()`: Persist results to JSON
- `load_benchmark_results()`: Load previous results
- `print_benchmark_summary()`: CLI summary output

**Metrics tracked**:
- Throughput (tokens/sec)
- Latency (seconds per response)
- Total tokens generated
- Min/max/average across runs

### 4. `app.py` — Streamlit UI
**Purpose**: Interactive interface with 3 main tabs

**Tab 1: Generate**
- Select model and prompt
- Adjust temperature, max_tokens
- Real-time performance metrics
- Stores in session history

**Tab 2: Benchmark**
- Run full benchmark suite
- Shows progress
- Saves results automatically

**Tab 3: Comparison**
- View side-by-side metrics
- Quality vs Speed analysis
- Best-use recommendations

**Design**: Stateful UI with `st.session_state` for caching benchmark results

### 5. `benchmark_cli.py` — CLI Benchmark
**Purpose**: Non-UI benchmark for automation and batch runs

**Features**:
- `--model`: Benchmark single model
- `--quick`: 5-prompt quick run vs full 13-prompt
- `--output`: Custom output filename
- Formatted table output

**Use case**: Automated pipelines, CI/CD, reproducible benchmarks

### 6. `setup.py` — Setup Verification
**Purpose**: Verify environment and download models

**Checks**:
1. Python version (3.8+)
2. Python dependencies installed
3. Ollama running and reachable
4. Required models downloaded

**Offer**: Interactive model pulling if missing

## Data Flow

### Inference Request
```
User Prompt (Streamlit)
    ↓
OllamaInference.generate()
    ↓
POST /api/generate to Ollama (localhost:11434)
    ↓
Local Model Inference
    ↓
Response with metrics (time, tokens)
    ↓
Display in UI + History
```

### Benchmark Run
```
run_benchmark()
    ↓
For each model in MODELS:
    ├─ For each prompt in BENCHMARK_PROMPTS:
    │  ├─ Call inference.generate()
    │  ├─ Measure time & tokens
    │  └─ Store metrics
    └─ Aggregate stats (avg, min, max)
    ↓
Save to benchmark_results.json
    ↓
Display in Streamlit Comparison tab
```

## Quality vs Speed Tradeoffs

### Implemented Metrics

1. **Speed (throughput)**
   - Tokens per second
   - Lower = slower but may be higher quality
   - Directly impacts user experience (latency)

2. **Quality (capability)**
   - 1-10 rating based on reasoning, knowledge, nuance
   - 7B models: 7-8/10 (good for most tasks)
   - 13B models: 9/10 (excellent reasoning)

3. **Efficiency (quality per token/sec)**
   - Quality ÷ Speed
   - Shows "bang for buck"
   - Helps choose best model for constraints

### Model Positioning

```
        Quality
           ↑
        9  │      Llama 2 13B (best quality, slowest)
           │
        8  │         Mistral 7B (balanced)
           │
        7  │      Llama 2 7B (fastest)
           │
           └─────────────────────→ Speed
           10 tok/s  20 tok/s  30 tok/s
```

## Constraints Documentation

### Privacy
- ✅ All processing local
- ✅ No API calls to external services
- ✅ Models downloaded once, reused indefinitely

### Latency
- ⚡ No network round-trip (vs API calls)
- ⚠️ Bounded by model size and hardware (GPU/CPU)
- 📊 Measurable via benchmark

### Cost
- 💰 No per-token billing
- 💰 One-time hardware investment
- 💰 Electricity cost during inference
- 📊 Break-even calculator: Cost/month of API vs GPU + electricity

## Technology Choices

### Why Streamlit?
- Fast to prototype
- Built-in state management
- Great for data visualization
- No frontend boilerplate

### Why Ollama?
- Simplest local inference
- No CUDA/PyTorch setup complexity
- Works on CPU and GPU
- Easy model management

### Why requests library?
- Lightweight
- No additional ML dependencies
- Direct HTTP API control
- Explicit error handling

### Why separate CLI?
- Automation-friendly
- No UI overhead
- Scripting support
- CI/CD integration

## Extension Points

### Add New Models
1. Add to `MODELS` dict in `config.py`
2. Run `ollama pull <model-name>`
3. UI automatically includes it

### Add Custom Prompts
1. Edit `BENCHMARK_PROMPTS` in `config.py`
2. Re-run benchmark for fresh metrics

### Add Metrics
1. Extend response parsing in `inference.py`
2. Add metrics to `run_benchmark()` aggregation
3. Display in `app.py` Comparison tab

### Add Inference Types
1. Create new method in `OllamaInference`
2. Call from app tabs (e.g., chat mode, streaming)

## Performance Considerations

### Benchmarking Accuracy
- **Warm-up**: First run loads model into VRAM, may be slower
- **Standardized**: All models get same prompts for fair comparison
- **Runs**: Multiple prompts reduce variance
- **Hardware variation**: Results vary 2-3x based on GPU/CPU

### UI Responsiveness
- Benchmark is blocking (expected 5-10 min)
- Single inference < 10 seconds (shows progress spinner)
- Results cached in `st.session_state` to avoid re-running

### Memory Usage
- 7B models: ~5GB VRAM
- 13B models: ~8GB VRAM
- Running multiple simultaneously not supported in this version

## Future Enhancements

1. **Streaming responses** for real-time token display
2. **Chat mode** with conversation history
3. **Fine-tuning** support for domain-specific tasks
4. **Quantization comparison** (Q4 vs Q5 vs FP16)
5. **Hardware profiling** (CPU/GPU/memory during inference)
6. **Multi-model serving** (parallel inference)
7. **API server** wrapping local models as remote API
8. **Cost calculator** (electricity + GPU amortization)

---

**Architecture Review**: This is a simple, single-process design focused on clarity and benchmarking. For production serving, consider:
- Multi-worker inference server (vLLM, TensorRT-LLM)
- Message queue for batch processing
- Caching layer for repeated queries
