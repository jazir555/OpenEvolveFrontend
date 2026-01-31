"""LLM adapters for cloud and local backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .utils import deterministic_seed, optional_import


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    seed: Optional[int] = None
    device: str = "cpu"
    dtype: str = "auto"


class BaseLLM:
    provider: str
    model: str
    tokenizer: Any = None
    model_obj: Any = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        yield self.generate(prompt, **kwargs)

    def get_outlines_model(self):
        outlines = optional_import("outlines")
        if outlines is None:
            return None
        return None


class OpenAIChatLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.provider = "openai"
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.seed = config.seed
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        openai = optional_import("openai")
        if openai is None:
            raise RuntimeError("openai package not available")
        if hasattr(openai, "OpenAI"):
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            openai.api_key = self.api_key
            if self.base_url:
                openai.base_url = self.base_url
            self._client = openai
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        client = self._get_client()
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            )
            return response.output_text
        if hasattr(client, "chat"):
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return response.choices[0].message.content if response.choices else ""
        response = client.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.choices[0].text if response.choices else ""

    def get_outlines_model(self):
        outlines = optional_import("outlines")
        if outlines is None:
            return None
        try:
            return outlines.models.from_openai(self.model, api_key=self.api_key, base_url=self.base_url)
        except Exception:
            return None


class AnthropicLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.provider = "anthropic"
        self.model = config.model
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        anthropic = optional_import("anthropic")
        if anthropic is None:
            raise RuntimeError("anthropic package not available")
        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    def get_outlines_model(self):
        outlines = optional_import("outlines")
        if outlines is None:
            return None
        try:
            return outlines.models.from_anthropic(self.model, api_key=self.api_key)
        except Exception:
            return None


class GoogleLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.provider = "google"
        self.model = config.model
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        genai = optional_import("google.generativeai")
        if genai is None:
            raise RuntimeError("google.generativeai package not available")
        genai.configure(api_key=self.api_key)
        self._client = genai
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        client = self._get_client()
        model = client.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return getattr(response, "text", "")

    def get_outlines_model(self):
        outlines = optional_import("outlines")
        if outlines is None:
            return None
        try:
            return outlines.models.from_gemini(self.model, api_key=self.api_key)
        except Exception:
            return None


class HFLocalLLM(BaseLLM):
    def __init__(self, model_path: str, device: str = "cpu", dtype: str = "auto", seed: Optional[int] = None):
        self.provider = "hf"
        self.model = model_path
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self._model = None
        self.tokenizer = None
        self.model_obj = None

    def _load(self):
        if self._model is not None:
            return
        transformers = optional_import("transformers")
        if transformers is None:
            raise RuntimeError("transformers package not available")
        torch = optional_import("torch")
        dtype = None
        if torch and self.dtype != "auto":
            dtype = getattr(torch, self.dtype, None)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model, use_fast=True)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=dtype)
        self.model_obj = self._model
        if torch:
            self._model.to(self.device)
            self._model.eval()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._load()
        transformers = optional_import("transformers")
        torch = optional_import("torch")
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)
        with deterministic_seed(self.seed or 0):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def get_outlines_model(self):
        outlines = optional_import("outlines")
        if outlines is None:
            return None
        try:
            self._load()
            return outlines.models.from_transformers(self._model, self.tokenizer)
        except Exception:
            return None


def build_llm(config: LLMConfig) -> BaseLLM:
    provider = config.provider.lower()
    if provider == "openai":
        return OpenAIChatLLM(config)
    if provider == "anthropic":
        return AnthropicLLM(config)
    if provider == "google":
        return GoogleLLM(config)
    if provider == "hf":
        return HFLocalLLM(config.model, device=config.device, dtype=config.dtype, seed=config.seed)
    raise ValueError(f"Unsupported provider: {provider}")
