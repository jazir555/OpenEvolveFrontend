"""
Provider Catalogue

A dependency-light registry and lookup for the external services, LLM backends,
embedding providers and executors that workflow steps and the RLM integration
rely on.

Design goals:
- Importable with NO external services available (stdlib only).
- Every provider is registered with an ``available`` flag that is only set to
  ``True`` when its backing package can actually be imported. Nothing here will
  raise at import time or at lookup time.
- ``resolve()`` lazily instantiates a provider through an optional ``factory``
  callable and *guards* external services: any failure (missing package,
  missing credentials, network error) is caught and surfaced as ``None`` plus a
  recorded error rather than an exception.

This is the single source of truth for "which providers exist and can I use
them right now" across the orchestration layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProviderKind(str, Enum):
    """Category of a registered provider."""

    LLM = "llm"
    EMBEDDING = "embedding"
    SERVICE = "service"
    EXECUTOR = "executor"
    STORAGE = "storage"


@dataclass
class ProviderInfo:
    """Metadata describing a single provider/backend."""

    name: str
    kind: ProviderKind
    capabilities: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None
    auth_env: Optional[str] = None
    available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    factory: Optional[Callable[..., Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "capabilities": list(self.capabilities),
            "endpoint": self.endpoint,
            "auth_env": self.auth_env,
            "available": self.available,
            "metadata": dict(self.metadata),
        }


@dataclass
class ProviderResolution:
    """Result of attempting to resolve (instantiate) a provider."""

    name: str
    available: bool
    instance: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.available and self.instance is not None


class ProviderCatalogue:
    """
    In-memory registry of providers with safe lookup and resolution.

    Usage::

        cat = ProviderCatalogue()
        cat.register(ProviderInfo(name="openai", kind=ProviderKind.LLM,
                                  capabilities=["chat", "completion"]))
        info = cat.get("openai")
        providers = cat.lookup(capability="chat")
        resolved = cat.resolve("openai", api_key=...)
    """

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderInfo] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(self, provider: ProviderInfo) -> ProviderInfo:
        """Register a provider, replacing any existing entry with the same name."""
        self._providers[provider.name] = provider
        logger.debug("Registered provider %s (%s)", provider.name, provider.kind.value)
        return provider

    def register_many(self, providers: List[ProviderInfo]) -> None:
        for provider in providers:
            self.register(provider)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def get(self, name: str) -> Optional[ProviderInfo]:
        """Return provider metadata by exact name (case-sensitive)."""
        return self._providers.get(name)

    def get_or_raise(self, name: str) -> ProviderInfo:
        info = self.get(name)
        if info is None:
            raise KeyError(f"Provider '{name}' is not registered")
        return info

    def lookup(self, capability: Optional[str] = None,
               kind: Optional[ProviderKind] = None) -> List[ProviderInfo]:
        """
        Find providers, optionally filtered by a capability and/or kind.

        Args:
            capability: Required capability substring (matched case-insensitively).
            kind: Required :class:`ProviderKind`.

        Returns:
            List of matching :class:`ProviderInfo` (may be empty).
        """
        results: List[ProviderInfo] = []
        cap = capability.lower() if capability else None
        for provider in self._providers.values():
            if kind is not None and provider.kind != kind:
                continue
            if cap is not None and not any(cap in c.lower() for c in provider.capabilities):
                continue
            results.append(provider)
        return results

    def list_providers(self, kind: Optional[ProviderKind] = None) -> List[ProviderInfo]:
        """List all providers, optionally filtered by kind."""
        if kind is None:
            return list(self._providers.values())
        return [p for p in self._providers.values() if p.kind == kind]

    def available_providers(self, kind: Optional[ProviderKind] = None) -> List[ProviderInfo]:
        """List only providers that are currently available."""
        return [p for p in self.list_providers(kind) if p.available]

    def is_available(self, name: str) -> bool:
        """Return True if a provider is registered and currently available."""
        info = self.get(name)
        return info is not None and info.available

    # ------------------------------------------------------------------ #
    # Resolution (guarded external service instantiation)
    # ------------------------------------------------------------------ #
    def resolve(self, name: str, **kwargs: Any) -> ProviderResolution:
        """
        Attempt to instantiate a provider via its registered factory.

        External services are *guarded*: any exception during import or
        instantiation is captured in the returned resolution rather than raised.

        Args:
            name: Provider name.
            **kwargs: Passed through to the provider's factory callable.

        Returns:
            :class:`ProviderResolution` describing the outcome.
        """
        info = self.get(name)
        if info is None:
            return ProviderResolution(name=name, available=False,
                                      error=f"Provider '{name}' not registered")
        if not info.available:
            return ProviderResolution(
                name=name, available=False,
                error=f"Provider '{name}' is registered but not available")

        if info.factory is None:
            # No instantiation logic, but the provider is available.
            return ProviderResolution(name=name, available=True, instance=None)

        try:
            instance = info.factory(**kwargs)
            return ProviderResolution(name=name, available=True, instance=instance)
        except Exception as exc:  # noqa: BLE001 - guard external services
            logger.warning("Failed to resolve provider '%s': %s", name, exc)
            return ProviderResolution(
                name=name, available=True, instance=None,
                error=f"{type(exc).__name__}: {exc}")

    def capabilities_for(self, name: str) -> List[str]:
        """Return the capability list for a provider (empty if unknown)."""
        info = self.get(name)
        return list(info.capabilities) if info else []


# ---------------------------------------------------------------------- #
# Default catalogue
# ---------------------------------------------------------------------- #
def _safe_import(name: str) -> bool:
    """Return True if a module can be imported (no side effects / no raise)."""
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _default_providers() -> List[ProviderInfo]:
    """Build the built-in provider catalogue entries."""
    providers: List[ProviderInfo] = []

    # --- LLM backends ------------------------------------------------- #
    llm_backends = [
        ("openai", "openai", ["chat", "completion", "embeddings"]),
        ("anthropic", "anthropic", ["chat", "completion"]),
        ("azure-openai", "azure", ["chat", "completion", "embeddings"]),
        ("groq", "groq", ["chat", "completion"]),
        ("ollama", "ollama", ["chat", "completion", "embeddings"]),
        ("local", "local", ["chat", "completion"]),
    ]
    for name, auth_env, caps in llm_backends:
        pkg = name.split("-")[0]
        providers.append(ProviderInfo(
            name=name, kind=ProviderKind.LLM, capabilities=caps,
            auth_env=f"{auth_env.upper()}_API_KEY" if auth_env != "local" else None,
            available=_safe_import(pkg),
            metadata={"package": pkg},
        ))

    # --- Embedding providers ------------------------------------------ #
    providers.append(ProviderInfo(
        name="hash-embeddings", kind=ProviderKind.EMBEDDING,
        capabilities=["embed", "similarity"],
        available=True,  # pure-python, no external package required
        metadata={"deterministic": True},
    ))
    providers.append(ProviderInfo(
        name="openai-embeddings", kind=ProviderKind.EMBEDDING,
        capabilities=["embed", "similarity"],
        auth_env="OPENAI_API_KEY", available=_safe_import("openai"),
    ))
    providers.append(ProviderInfo(
        name="huggingface-embeddings", kind=ProviderKind.EMBEDDING,
        capabilities=["embed", "similarity"],
        available=_safe_import("sentence_transformers")
        or _safe_import("transformers"),
    ))

    # --- RLM execution backends --------------------------------------- #
    # RLM is itself guarded: only available if the `rlm` package is on path.
    def _make_rlm(backend: str = "openai", **kw: Any):
        from rlm import RLM  # type: ignore
        from rlm.clients import OpenAIClient, AnthropicClient  # type: ignore
        client_cls = OpenAIClient if backend == "openai" else AnthropicClient
        return RLM(backend=backend, client=client_cls(**kw))

    rlm_available = _safe_import("rlm")
    for backend in ("openai", "anthropic"):
        providers.append(ProviderInfo(
            name=f"rlm-{backend}", kind=ProviderKind.EXECUTOR,
            capabilities=["code-execution", "recursive-reasoning", "sub-lm"],
            auth_env="OPENAI_API_KEY" if backend == "openai" else "ANTHROPIC_API_KEY",
            available=rlm_available,
            factory=(lambda b=backend, **kw: _make_rlm(b, **kw)),
            metadata={"engine": "rlm"},
        ))

    # --- Workflow service tasks --------------------------------------- #
    # These represent external services invoked by `service` workflow steps.
    # They are not available by default; the orchestrator guards their use.
    service_providers = [
        ("evaluation_service", ["evaluate", "score", "fitness"]),
        ("knowledge_service", ["query", "store", "retrieve"]),
        ("vector_search", ["search", "index", "similarity"]),
        ("notification_service", ["notify", "alert"]),
    ]
    for name, caps in service_providers:
        providers.append(ProviderInfo(
            name=name, kind=ProviderKind.SERVICE, capabilities=caps,
            available=False,
            metadata={"external": True},
        ))

    return providers


_DEFAULT_CATALOGUE: Optional[ProviderCatalogue] = None


def get_provider_catalogue() -> ProviderCatalogue:
    """
    Return the process-wide default provider catalogue, populated lazily with
    the built-in providers. Safe to call any number of times.
    """
    global _DEFAULT_CATALOGUE
    if _DEFAULT_CATALOGUE is None:
        cat = ProviderCatalogue()
        cat.register_many(_default_providers())
        _DEFAULT_CATALOGUE = cat
    return _DEFAULT_CATALOGUE


def reset_provider_catalogue() -> ProviderCatalogue:
    """Reset and return a fresh default catalogue (primarily for tests)."""
    global _DEFAULT_CATALOGUE
    cat = ProviderCatalogue()
    cat.register_many(_default_providers())
    _DEFAULT_CATALOGUE = cat
    return cat
