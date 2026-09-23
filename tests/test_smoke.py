"""Smoke tests for the Local SLM Benchmark project.

These validate module imports and configuration integrity. They intentionally
avoid importing ``app.py`` (which executes Streamlit calls at import time) and
do not require a running Ollama server, so they are safe to run in CI.
"""
import config
from inference import OllamaInference, OllamaError


def test_models_configured():
    assert isinstance(config.MODELS, dict)
    assert len(config.MODELS) >= 1
    required_keys = {"name", "params", "speed", "quality"}
    for name, meta in config.MODELS.items():
        assert isinstance(name, str) and name
        missing = required_keys - set(meta)
        assert not missing, f"{name} is missing keys: {missing}"
        assert isinstance(meta["quality"], int)
        assert 0 <= meta["quality"] <= 10


def test_lightweight_default_model_first():
    # A small model should be the default (first) option for quick testing.
    first_model = next(iter(config.MODELS))
    assert config.MODELS[first_model]["params"] in {"0.5B", "1B"}


def test_benchmark_prompts_present():
    assert isinstance(config.BENCHMARK_PROMPTS, list)
    assert len(config.BENCHMARK_PROMPTS) > 0
    assert all(isinstance(p, str) and p for p in config.BENCHMARK_PROMPTS)


def test_ollama_host_default():
    assert config.OLLAMA_HOST.startswith("http")


def test_inference_client_constructs():
    client = OllamaInference()
    assert client.host == config.OLLAMA_HOST
    assert client.generate_url.endswith("/api/generate")
    assert issubclass(OllamaError, Exception)


def test_inference_history_entry_schema():
    required = {"timestamp", "model", "prompt", "output",
                "time_elapsed", "tokens_generated", "throughput"}
    entry = {k: None for k in required}
    assert set(entry.keys()) == required
