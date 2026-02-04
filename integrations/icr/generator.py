"""Initial output generation for ICR.

Generates base outputs for iterative refinement through multiple strategies
including direct generation and variant generation for diversity.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class GenerationStrategy(Enum):
    """Strategies for content generation."""
    DIRECT = "direct"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    VARIETY_SAMPLING = "variety_sampling"


@dataclass
class GenerationMetadata:
    """Metadata for a generation operation."""
    strategy: GenerationStrategy
    model: Optional[str] = None
    temperature: float = 0.7
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of a generation operation.
    
    Attributes:
        content: The generated content string
        confidence: Confidence score between 0 and 1
        generation_time: Time taken for generation in seconds
        tokens_used: Total tokens consumed (input + output)
        metadata: Additional metadata about the generation
    """
    content: str
    confidence: float = 0.0
    generation_time: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the generation result."""
        if not 0 <= self.confidence <= 1:
            logger.warning(f"Confidence {self.confidence} outside [0,1], clamping")
            self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "generation_time": self.generation_time,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """Create from dictionary representation."""
        return cls(
            content=data["content"],
            confidence=data.get("confidence", 0.0),
            generation_time=data.get("generation_time", 0.0),
            tokens_used=data.get("tokens_used", 0),
            metadata=data.get("metadata", {}),
        )


class Generator:
    """Base generator for initial output generation.
    
    The Generator creates initial content that will be refined through
    the ICR loop. It supports multiple generation strategies and can
    produce diverse variants for comparison.
    
    Example:
        >>> generator = Generator()
        >>> result = generator.generate(
        ...     prompt="Write a Python function to calculate factorial",
        ...     context={"language": "python", "style": "functional"}
        ... )
        >>> print(result.content)
    """
    
    def __init__(
        self,
        default_strategy: GenerationStrategy = GenerationStrategy.DIRECT,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize the generator.
        
        Args:
            default_strategy: Default generation strategy to use
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens per generation
        """
        self.default_strategy = default_strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._generation_count = 0
        self._backend: Optional[Callable] = None
        
        logger.info(f"Initialized Generator with strategy={default_strategy.value}")
    
    def set_backend(self, backend: Callable[[str, Dict[str, Any]], str]) -> None:
        """Set the generation backend (e.g., LLM API).
        
        Args:
            backend: Callable that takes (prompt, params) and returns content string
        """
        self._backend = backend
        logger.debug("Generation backend registered")
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[GenerationStrategy] = None,
    ) -> GenerationResult:
        """Generate initial output.
        
        Args:
            prompt: The generation prompt
            context: Additional context for generation
            strategy: Override the default strategy
            
        Returns:
            GenerationResult containing the generated content and metadata
        """
        start_time = time.time()
        context = context or {}
        strategy = strategy or self.default_strategy
        
        logger.info(f"Generating with strategy={strategy.value}", extra={
            "correlation_id": context.get("correlation_id"),
            "strategy": strategy.value,
        })
        
        try:
            # Use backend if available, otherwise use fallback
            if self._backend:
                content = self._backend(prompt, {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    **context,
                })
            else:
                content = self._fallback_generate(prompt, context, strategy)
            
            generation_time = time.time() - start_time
            self._generation_count += 1
            
            # Estimate confidence based on generation characteristics
            confidence = self._estimate_confidence(content, strategy)
            
            result = GenerationResult(
                content=content,
                confidence=confidence,
                generation_time=generation_time,
                tokens_used=self._estimate_tokens(prompt + content),
                metadata={
                    "strategy": strategy.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "generation_number": self._generation_count,
                    **context,
                },
            )
            
            logger.debug(f"Generation completed in {generation_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise GenerationError(f"Failed to generate: {e}") from e
    
    def generate_variants(
        self,
        prompt: str,
        n: int = 3,
        context: Optional[Dict[str, Any]] = None,
        diversity_temperature: bool = True,
    ) -> List[GenerationResult]:
        """Generate N variant outputs for diversity.
        
        Args:
            prompt: The generation prompt
            n: Number of variants to generate
            context: Additional context for generation
            diversity_temperature: Vary temperature for diversity
            
        Returns:
            List of GenerationResult objects
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        
        logger.info(f"Generating {n} variants")
        
        variants = []
        base_temp = self.temperature
        
        for i in range(n):
            # Vary temperature for diversity if enabled
            if diversity_temperature:
                # Range from base - 0.2 to base + 0.3
                variation = (i / max(n - 1, 1)) * 0.5 - 0.2
                self.temperature = max(0.1, min(1.0, base_temp + variation))
            
            variant_context = {
                **(context or {}),
                "variant_index": i,
                "total_variants": n,
            }
            
            result = self.generate(prompt, variant_context)
            variants.append(result)
        
        # Restore original temperature
        self.temperature = base_temp
        
        logger.info(f"Generated {len(variants)} variants")
        return variants
    
    def _fallback_generate(
        self,
        prompt: str,
        context: Dict[str, Any],
        strategy: GenerationStrategy,
    ) -> str:
        """Fallback generation when no backend is set.
        
        This creates a placeholder response. In production, set a proper backend.
        """
        logger.warning("Using fallback generation - set a backend for real generation")
        
        if strategy == GenerationStrategy.CHAIN_OF_THOUGHT:
            return f"[CoT] Let me think step by step about: {prompt}\n\n1. First, we need to understand...\n2. Then, we should consider...\n3. Finally, the solution is..."
        elif strategy == GenerationStrategy.FEW_SHOT:
            return f"[Few-Shot] Based on similar examples, here's a response to: {prompt}\n\nExample 1: ...\nExample 2: ...\nResponse: ..."
        else:
            return f"[Generated] Response to: {prompt}\n\nThis is a placeholder generation. Please configure a generation backend."
    
    def _estimate_confidence(self, content: str, strategy: GenerationStrategy) -> float:
        """Estimate confidence based on content characteristics."""
        base_confidence = 0.7
        
        # Adjust based on content length
        if len(content) < 50:
            base_confidence -= 0.1
        elif len(content) > 500:
            base_confidence += 0.05
        
        # Adjust based on strategy
        strategy_multipliers = {
            GenerationStrategy.DIRECT: 1.0,
            GenerationStrategy.CHAIN_OF_THOUGHT: 1.1,
            GenerationStrategy.FEW_SHOT: 1.15,
            GenerationStrategy.VARIETY_SAMPLING: 0.95,
        }
        
        confidence = base_confidence * strategy_multipliers.get(strategy, 1.0)
        return max(0.0, min(1.0, confidence))
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (assuming ~4 chars per token)."""
        return len(text) // 4
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "total_generations": self._generation_count,
            "default_strategy": self.default_strategy.value,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_backend": self._backend is not None,
        }


class GenerationError(Exception):
    """Error during generation operation."""
    pass
