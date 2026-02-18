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
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self._request_fn = request_fn

        self._client = self._create_client(self.provider, api_key) if request_fn is None else None
        
        # Cloud backends typically only support Tier 0 measurement
        # OpenAI supports score equality if logprobs are enabled
        self._capabilities = BackendCapabilities(
            supports_torch_deterministic=False,
            supports_fixed_batch_repeatability=False,
            supports_score_equality=self.provider == "openai",
        )

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def _create_client(self, provider: str, api_key: Optional[str]) -> Any:
        if provider == "openai":
            openai = optional_import("openai")
            if openai and api_key:
                try:
                    return openai.OpenAI(api_key=api_key)
                except AttributeError:
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
        
        try:
            if self.provider == "openai":
                # Modern OpenAI client
                if hasattr(self._client, "chat"):
                    response = self._client.chat.completions.create(
                        model=self.model, 
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        **kwargs
                    )
                    return response.choices[0].message.content if response.choices else ""
                # Legacy or mock
                response = self._client.completions.create(model=self.model, prompt=prompt, temperature=0, **kwargs)
                return response.choices[0].text if response.choices else ""
                
            if self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model, 
                    max_tokens=1024, 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return response.content[0].text if response.content else ""
                
            if self.provider == "google":
                model = self._client.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return getattr(response, "text", "")
        except Exception as exc:
            return f"[error:{self.provider}] {exc}"
            
        return f"[cloud:{self.provider}] {prompt}"

    def generate(self, prompts: List[str], tier: int, **kwargs: Any) -> List[str]:
        # Cloud LLMs only support Tier 0 measurement
        if tier > 0:
            # We still allow it but log that it's just measurement
            pass
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
