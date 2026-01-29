"""
Cloud API Client Integration for Adaptive MDAP.

Provides unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash)
"""

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from adaptive_mdap.utils.logger import get_logger

logger = get_logger("integrations.cloud_api")


class Provider(Enum):
    """Supported API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class APIResponse:
    """Response from API call."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    cost: float


@dataclass
class APIConfig:
    """Configuration for API client."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


class BaseAPIClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
    
    @abstractmethod
    def call(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> APIResponse:
        """Make single API call."""
        pass
    
    @abstractmethod
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Estimate cost for token usage."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
        }


class OpenAIClient(BaseAPIClient):
    """OpenAI API client."""
    
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4": {"input": 0.03, "output": 0.06},
    }
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except ImportError:
            logger.warning("openai package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def call(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> APIResponse:
        """Make OpenAI API call."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        start_time = time.time()
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = self.estimate_cost(input_tokens, output_tokens, model)
                
                # Update stats
                self._call_count += 1
                self._total_tokens += input_tokens + output_tokens
                self._total_cost += cost
                
                return APIResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    latency_ms=latency_ms,
                    cost=cost,
                )
                
            except Exception as e:
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for OpenAI models."""
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class AnthropicClient(BaseAPIClient):
    """Anthropic API client."""
    
    PRICING = {
        "claude-3-5-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except ImportError:
            logger.warning("anthropic package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def call(
        self,
        prompt: str,
        model: str = "claude-3-5-haiku",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> APIResponse:
        """Make Anthropic API call."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                content = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = self.estimate_cost(input_tokens, output_tokens, model)
                
                self._call_count += 1
                self._total_tokens += input_tokens + output_tokens
                self._total_cost += cost
                
                return APIResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    latency_ms=latency_ms,
                    cost=cost,
                )
                
            except Exception as e:
                logger.warning(f"Anthropic API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for Anthropic models."""
        pricing = self.PRICING.get(model, self.PRICING["claude-3-5-haiku"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class CloudAPIClient:
    """
    Unified cloud API client for multiple providers.
    
    Automatically selects appropriate client based on model name.
    """
    
    def __init__(self):
        self._clients: Dict[Provider, BaseAPIClient] = {}
        self._model_to_provider = {
            "gpt-4o-mini": Provider.OPENAI,
            "gpt-4o": Provider.OPENAI,
            "gpt-4": Provider.OPENAI,
            "claude-3-5-haiku": Provider.ANTHROPIC,
            "claude-3-5-sonnet": Provider.ANTHROPIC,
            "claude-3-opus": Provider.ANTHROPIC,
        }
    
    def register_client(self, provider: Provider, client: BaseAPIClient) -> None:
        """Register a client for a provider."""
        self._clients[provider] = client
    
    def call(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> APIResponse:
        """
        Make API call to appropriate provider.
        
        Args:
            prompt: The prompt to send
            model: Model name
            **kwargs: Additional arguments
            
        Returns:
            APIResponse
        """
        provider = self._model_to_provider.get(model)
        
        if not provider:
            raise ValueError(f"Unknown model: {model}")
        
        client = self._clients.get(provider)
        
        if not client:
            raise RuntimeError(f"No client registered for {provider.value}")
        
        return client.call(prompt, model, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all clients."""
        stats = {}
        for provider, client in self._clients.items():
            stats[provider.value] = client.get_stats()
        
        # Calculate totals
        total_calls = sum(s["call_count"] for s in stats.values())
        total_tokens = sum(s["total_tokens"] for s in stats.values())
        total_cost = sum(s["total_cost"] for s in stats.values())
        
        stats["total"] = {
            "call_count": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
        
        return stats
    
    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a model."""
        provider = self._model_to_provider.get(model)
        
        if not provider:
            raise ValueError(f"Unknown model: {model}")
        
        client = self._clients.get(provider)
        
        if not client:
            # Return rough estimate
            return (input_tokens + output_tokens) / 1000 * 0.01
        
        return client.estimate_cost(input_tokens, output_tokens, model)


def create_client(
    provider: Provider,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseAPIClient:
    """
    Factory function to create API clients.
    
    Args:
        provider: The provider to create client for
        api_key: API key (reads from env if not provided)
        **kwargs: Additional configuration
        
    Returns:
        Configured API client
    """
    import os
    
    if not api_key:
        env_var = f"{provider.value.upper()}_API_KEY"
        api_key = os.getenv(env_var)
    
    config = APIConfig(api_key=api_key, **kwargs)
    
    if provider == Provider.OPENAI:
        return OpenAIClient(config)
    elif provider == Provider.ANTHROPIC:
        return AnthropicClient(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
