import requests
import json
from typing import Dict, Any
from config import OLLAMA_HOST

class OllamaInference:
    """Interface for communicating with Ollama."""

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.generate_url = f"{host}/api/generate"

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
        Generate text using Ollama.

        Returns dict with 'text' and 'model' keys.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": top_p,
            "top_k": top_k,
        }

        response = requests.post(self.generate_url, json=payload, timeout=300)
        response.raise_for_status()

        data = response.json()
        return {
            "text": data.get("response", ""),
            "model": model,
            "tokens_generated": data.get("eval_count", 0),
            "time_seconds": data.get("eval_duration", 0) / 1e9,
        }

    def list_models(self) -> list:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """Download a model from the Ollama registry."""
        try:
            payload = {"name": model}
            response = requests.post(
                f"{self.host}/api/pull",
                json=payload,
                timeout=None,  # No timeout for long download
                stream=True
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
