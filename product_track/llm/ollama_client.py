"""
Local Ollama API client for grounded clinical LLM inference.
"""

from __future__ import annotations

import time
from typing import Any
import requests


class OllamaClient:
    """
    Client for interacting with local Ollama instance (100% offline).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        default_temperature: float = 0.2,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if local Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of locally downloaded model names."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Send generation request to local Ollama.

        Returns a dictionary with:
            - response: text string
            - model: model used
            - total_duration_s: float
            - eval_count: number of generated tokens
            - eval_rate_tok_s: tokens per second
        """
        model_name = model or self.default_model
        temp = self.default_temperature if temperature is None else temperature

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }
        if system:
            payload["system"] = system

        start_time = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed connecting to {self.base_url}: {e}") from e

        duration = time.perf_counter() - start_time
        eval_count = data.get("eval_count", 0)
        tok_s = (eval_count / duration) if duration > 0 and eval_count > 0 else 0.0

        return {
            "response": data.get("response", "").strip(),
            "model": model_name,
            "total_duration_s": round(duration, 3),
            "eval_count": eval_count,
            "eval_rate_tok_s": round(tok_s, 1),
            "done": data.get("done", True),
        }
