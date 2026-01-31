"""Backend adapters for deterministic LLM integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from .utils import optional_import


class LLMInterface(Protocol):
    provider: str
    model: str
    tokenizer: Any

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    def stream(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        ...


class CallableLLM:
    """Simple wrapper to adapt callables into an LLM-like interface."""

    def __init__(
        self,
        generate_fn: Callable[..., str],
        stream_fn: Optional[Callable[..., Iterable[str]]] = None,
        provider: str = "callable",
        model: str = "callable",
    ):
        self._generate = generate_fn
        self._stream = stream_fn
        self.provider = provider
        self.model = model
        self.tokenizer = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._generate(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        if self._stream is None:
            yield self.generate(prompt, **kwargs)
        else:
            yield from self._stream(prompt, **kwargs)


@dataclass
class BackendCapabilities:
    supports_torch_deterministic: bool = False
    supports_fixed_batch_repeatability: bool = False
    supports_score_equality: bool = False


class BackendAdapter:
    """Minimal backend adapter interface for reproducibility checks."""

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def generate(self, prompts: List[str], tier: int, **kwargs: Any) -> List[str]:
        raise NotImplementedError("BackendAdapter.generate must be implemented")


class CloudBackend(BackendAdapter):
    """Cloud backend adapter implementing Tier 0 measurement only."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        request_fn: Optional[Callable[..., str]] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._request_fn = request_fn

        self._client = self._create_client(provider, api_key) if request_fn is None else None
        self._capabilities = BackendCapabilities(
            supports_torch_deterministic=False,
            supports_fixed_batch_repeatability=False,
            supports_score_equality=provider.lower() == "openai",
        )

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def _create_client(self, provider: str, api_key: Optional[str]) -> Any:
        provider = provider.lower()
        if provider == "openai":
            openai = optional_import("openai")
            if openai and api_key:
                openai.api_key = api_key
                return openai
        if provider == "anthropic":
            anthropic = optional_import("anthropic")
            if anthropic and api_key:
                return anthropic.Anthropic(api_key=api_key)
        if provider == "google":
            genai = optional_import("google.generativeai")
            if genai and api_key:
                genai.configure(api_key=api_key)
                return genai
        return None

    def _call_api(self, prompt: str, **kwargs: Any) -> str:
        if self._request_fn is not None:
            return self._request_fn(prompt=prompt, model=self.model, **kwargs)
        if self._client is None:
            return f"[cloud:{self.provider}] {prompt}"
        if self.provider.lower() == "openai":
            response = self._client.completions.create(model=self.model, prompt=prompt, **kwargs)
            return response.choices[0].text if response.choices else ""
        if self.provider.lower() == "anthropic":
            response = self._client.messages.create(model=self.model, max_tokens=512, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text if response.content else ""
        if self.provider.lower() == "google":
            model = self._client.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return getattr(response, "text", "")
        return f"[cloud:{self.provider}] {prompt}"

    def generate(self, prompts: List[str], tier: int, **kwargs: Any) -> List[str]:
        return [self._call_api(prompt, **kwargs) for prompt in prompts]


class LocalBackend(BackendAdapter):
    """Local backend adapter for simple deterministic generation."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self._capabilities = BackendCapabilities(
            supports_torch_deterministic=True,
            supports_fixed_batch_repeatability=True,
            supports_score_equality=True,
        )

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def generate(self, prompts: List[str], tier: int, **kwargs: Any) -> List[str]:
        return [self.llm.generate(prompt, **kwargs) for prompt in prompts]
