from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests


DEFAULT_ENV_PATH = Path("/Volumes/외장 2TB/cpu2026/common/code/.env")


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_openai_text(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()
    parts = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(str(content["text"]))
    return "\n".join(parts).strip()


def extract_gemini_text(data: Dict[str, Any]) -> str:
    parts = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if "text" in part:
                parts.append(str(part["text"]))
    return "\n".join(parts).strip()


class LLMClient:
    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        timeout: int = 240,
        env_path: str | Path = DEFAULT_ENV_PATH,
    ) -> None:
        load_env_file(env_path)
        self.provider = self.resolve_provider(provider)
        self.model = model or self.default_model(self.provider)
        self.timeout = timeout

    @staticmethod
    def resolve_provider(provider: str) -> str:
        provider = (provider or "auto").lower().strip()
        if provider != "auto":
            return provider
        preferred = os.environ.get("PATENT_PRO_PROVIDER", "").lower().strip()
        if preferred:
            return preferred
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        return "ollama"

    @staticmethod
    def default_model(provider: str) -> str:
        if provider == "openai":
            return os.environ.get("OPENAI_MODEL", "gpt-5.1")
        if provider == "gemini":
            return os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        return os.environ.get("OLLAMA_MODEL", "qwen3:14b")

    def generate(
        self,
        prompt: str,
        instructions: str = "",
        max_tokens: int = 1800,
        temperature: float = 0.1,
    ) -> str:
        if self.provider == "openai":
            return self.generate_openai(prompt, instructions, max_tokens, temperature)
        if self.provider == "gemini":
            return self.generate_gemini(prompt, instructions, max_tokens, temperature)
        if self.provider == "ollama":
            return self.generate_ollama(prompt, instructions, max_tokens, temperature)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_openai(self, prompt: str, instructions: str, max_tokens: int, temperature: float) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "instructions": instructions,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        # Some reasoning models ignore or reject temperature; keep it opt-in for compatibility.
        if os.environ.get("OPENAI_USE_TEMPERATURE", "").lower() in {"1", "true", "yes"}:
            payload["temperature"] = temperature
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        text = extract_openai_text(r.json())
        if not text:
            raise RuntimeError("OpenAI response contained no text")
        return text

    def generate_gemini(self, prompt: str, instructions: str, max_tokens: int, temperature: float) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        text = f"{instructions.strip()}\n\n{prompt}".strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        r = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        text = extract_gemini_text(r.json())
        if not text:
            raise RuntimeError("Gemini response contained no text")
        return text

    def generate_ollama(self, prompt: str, instructions: str, max_tokens: int, temperature: float) -> str:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": self.model,
            "prompt": f"{instructions.strip()}\n\n{prompt}".strip(),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 12000,
                "num_predict": max_tokens,
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return str(r.json().get("response", "")).strip()


def json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)
