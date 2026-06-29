import requests
import json
from typing import Dict, Any, Iterator
from config import OLLAMA_HOST


class OllamaError(Exception):
    """User-friendly error raised when an Ollama request cannot be completed."""


class OllamaInference:
    """Interface for communicating with Ollama."""

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.generate_url = f"{host}/api/generate"

    def _friendly_error(self, exc: Exception, model: str = None) -> "OllamaError":
        """Translate a low-level request error into an actionable message."""
        if isinstance(exc, requests.exceptions.ConnectionError):
            return OllamaError(
                f"Cannot reach Ollama at {self.host}. Is it running? "
                f"Start it with `ollama serve`."
            )
        if isinstance(exc, requests.exceptions.Timeout):
            return OllamaError(
                "Ollama timed out. The model may still be loading, or the "
                "prompt/token count is too large."
            )
        if isinstance(exc, requests.exceptions.HTTPError):
            status = getattr(exc.response, "status_code", None)
            if status == 404 and model:
                return OllamaError(
                    f"Model '{model}' isn't available locally. "
                    f"Pull it with `ollama pull {model}`."
                )
            return OllamaError(f"Ollama returned an error (HTTP {status}).")
        return OllamaError(f"Ollama request failed: {exc}")

    def _options(self, temperature, num_predict, top_p, top_k) -> Dict[str, Any]:
        return {
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": top_p,
            "top_k": top_k,
        }

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 256,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama (non-streaming).

        Returns dict with 'text', 'model', 'tokens_generated' (real eval_count),
        and 'time_seconds' (eval_duration) keys.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature, num_predict, top_p, top_k),
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=300)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise self._friendly_error(exc, model) from exc

        data = response.json()
        return {
            "text": data.get("response", ""),
            "model": model,
            "tokens_generated": data.get("eval_count", 0),
            "time_seconds": data.get("eval_duration", 0) / 1e9,
        }

    def generate_stream(
        self,
        model: str,
        prompt: str,
        stats: Dict[str, Any],
        temperature: float = 0.7,
        num_predict: int = 256,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Iterator[str]:
        """
        Stream text from Ollama token-by-token.

        Yields text chunks (suitable for ``st.write_stream``). Final metrics
        (tokens_generated, eval_seconds, total_seconds) are written into the
        provided ``stats`` dict once generation completes.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": self._options(temperature, num_predict, top_p, top_k),
        }

        try:
            with requests.post(
                self.generate_url, json=payload, timeout=300, stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        stats["tokens_generated"] = data.get("eval_count", 0)
                        stats["eval_seconds"] = data.get("eval_duration", 0) / 1e9
                        stats["total_seconds"] = data.get("total_duration", 0) / 1e9
        except requests.exceptions.RequestException as exc:
            raise self._friendly_error(exc, model) from exc

    def list_models(self) -> list:
        """List available models in Ollama. Returns [] if Ollama is unreachable."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.exceptions.RequestException:
            return []

    def is_available(self) -> bool:
        """Return True if the Ollama server responds."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def has_model(self, model: str) -> bool:
        """Return True if ``model`` is installed locally (tolerant of :latest)."""
        installed = self.list_models()
        if model in installed:
            return True
        base = model.split(":")[0]
        return any(m.split(":")[0] == base for m in installed)

    def pull_model(self, model: str) -> bool:
        """Download a model from the Ollama registry (blocks until complete)."""
        try:
            payload = {"name": model, "stream": False}
            response = requests.post(
                f"{self.host}/api/pull",
                json=payload,
                timeout=None,  # No timeout for long download
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as exc:
            print(f"Error pulling model: {exc}")
            return False
