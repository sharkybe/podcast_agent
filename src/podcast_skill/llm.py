"""Text generation for the free answer path (`ask`): a small LLM interface and a Gemini client.

Only used by `ask`. The Claude skill path (SKILL.md step 4) needs none of this: there Claude
itself is the generator.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Protocol

from podcast_skill.config import section

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"


class LLMError(RuntimeError):
    """The model could not be reached or gave no usable answer (rate limit, network, bad JSON)."""


class LLM(Protocol):
    model: str

    def generate(self, system: str, prompt: str, *, json_output: bool = False,
                 max_output_tokens: int = 2048, thinking_budget: Optional[int] = None) -> str:
        """Return the model's text reply. Raises LLMError on any failure."""
        ...


class GeminiLLM:
    """Gemini API through the official `google-genai` SDK (free tier works with a key from AI Studio)."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key_env: str = DEFAULT_API_KEY_ENV,
                 temperature: float = 0.2, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 retries: int = 2, client: Any = None):
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._api_key_env = api_key_env
        self._client = client

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise LLMError(
                    f"Environment variable {self._api_key_env} is not set. Get a free key at "
                    f"https://aistudio.google.com/apikey and put {self._api_key_env}=... in .env"
                )
            from google import genai
            from google.genai import types

            # A timeout so a stuck call becomes an LLMError (and the episode-list fallback) instead
            # of hanging; a couple of retries for "high demand" (503) and rate limits (429).
            self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
                timeout=int(self.timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=self.retries + 1, initial_delay=1.0, max_delay=8.0,
                                                     http_status_codes=[429, 500, 502, 503, 504]),
            ))
        return self._client

    def generate(self, system: str, prompt: str, *, json_output: bool = False,
                 max_output_tokens: int = 2048, thinking_budget: Optional[int] = None) -> str:
        from google.genai import errors, types

        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.temperature,
            "max_output_tokens": max_output_tokens,
        }
        if json_output:
            kwargs["response_mime_type"] = "application/json"
        if thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        try:
            response = self._get_client().models.generate_content(
                model=self.model, contents=prompt, config=types.GenerateContentConfig(**kwargs)
            )
        except errors.APIError as exc:   # 429 = free-tier rate limit, 5xx = Google side
            raise LLMError(f"Gemini error {exc.code}: {exc.message}") from exc
        except Exception as exc:   # network errors and the like
            raise LLMError(f"Gemini request failed: {exc}") from exc
        text = response.text or ""
        if not text.strip():
            raise LLMError("Gemini returned an empty reply")
        return text


def parse_json(text: str) -> Any:
    """Parse a JSON reply, tolerating a ```json fence around it."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model reply is not valid JSON: {text[:200]!r}") from exc


def build_llm(config: dict[str, Any], model: Optional[str] = None) -> LLM:
    cfg = section(config, "ask")
    provider = (cfg.get("llm_provider") or "gemini").lower()
    if provider != "gemini":
        raise RuntimeError(f"Unknown ask.llm_provider {provider!r}; only 'gemini' is implemented")
    model = model or cfg.get("llm_model") or DEFAULT_MODEL
    free_models = cfg.get("free_models") or []
    if free_models and model not in free_models:
        import sys

        print(f"Note: {model} is not in ask.free_models; it may need a paid Gemini key.", file=sys.stderr)
    return GeminiLLM(
        model=model,
        api_key_env=cfg.get("llm_api_key_env") or DEFAULT_API_KEY_ENV,
        temperature=float(cfg.get("temperature", 0.2)),
        timeout_seconds=float(cfg.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
        retries=int(cfg.get("retries", 2)),
    )
