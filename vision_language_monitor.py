"""
Vision Language Monitor for ICR (Iterative Contextual Refinements)

This module provides VLM (Vision Language Model) analysis capabilities for
UI heatmap composites, enabling automated insights about user interaction patterns,
cognitive friction points, and UI refinements.

Provider-specific optimizations are implemented for:
- OpenAI: GPT-4V/GPT-4o with optimized prompts and token management
- Anthropic: Claude 3.5 Vision with proper image encoding
- Google: Gemini Vision with specific API parameters
- Azure: Azure OpenAI Vision with proper authentication
"""

import os
import base64
import logging
import asyncio
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import json
from functools import wraps

logger = logging.getLogger(__name__)


class VLMProvider(str, Enum):
    """Supported VLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    MOCK = "mock"  # For testing without API


class AnalysisType(str, Enum):
    """Types of analysis to perform."""
    LAYOUT_ANALYSIS = "layout_analysis"
    INTERACTION_PATTERNS = "interaction_patterns"
    FRICTION_DETECTION = "friction_detection"
    HEATMAP_INTERPRETATION = "heatmap_interpretation"
    COMPREHENSIVE = "comprehensive"


# ============================================================================
# Provider-Specific Configuration
# ============================================================================

@dataclass
class ProviderOptimalConfig:
    """Optimal configuration settings for a specific provider."""
    temperature: float
    max_tokens: int
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    image_detail: str = "high"  # low, high, auto
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    use_streaming: bool = False


@dataclass
class ProviderPricing:
    """Pricing information for a provider (USD per 1K tokens)."""
    input_cost: float
    output_cost: float
    image_cost: float  # per image


@dataclass
class ProviderCapabilities:
    """Capabilities of a provider."""
    max_image_size: int  # in pixels
    supported_formats: List[str]
    supports_streaming: bool
    supports_function_calling: bool
    supports_system_messages: bool
    max_context_tokens: int
    recommended_for: List[str]


# Provider-specific configurations
PROVIDER_CONFIGS: Dict[VLMProvider, ProviderOptimalConfig] = {
    VLMProvider.OPENAI: ProviderOptimalConfig(
        temperature=0.1,  # Lower for more consistent UI analysis
        max_tokens=2048,
        top_p=0.95,
        image_detail="high",
        timeout=60,
        max_retries=3,
        retry_delay=2.0,
        use_streaming=False
    ),
    VLMProvider.ANTHROPIC: ProviderOptimalConfig(
        temperature=0.0,  # Anthropic works best with 0 for deterministic outputs
        max_tokens=4096,  # Claude has larger context
        top_p=0.999,
        image_detail="high",
        timeout=60,
        max_retries=3,
        retry_delay=1.5,
        use_streaming=False
    ),
    VLMProvider.GOOGLE: ProviderOptimalConfig(
        temperature=0.2,
        max_tokens=2048,
        top_p=0.95,
        top_k=40,
        image_detail="high",
        timeout=60,
        max_retries=2,
        retry_delay=1.0,
        use_streaming=False
    ),
    VLMProvider.AZURE: ProviderOptimalConfig(
        temperature=0.1,
        max_tokens=2048,
        top_p=0.95,
        image_detail="high",
        timeout=60,
        max_retries=3,
        retry_delay=2.0,
        use_streaming=False
    ),
    VLMProvider.MOCK: ProviderOptimalConfig(
        temperature=0.0,
        max_tokens=1024,
        timeout=5,
        max_retries=1,
        retry_delay=0.1,
        use_streaming=False
    ),
}


# Provider pricing (as of 2024 - update as needed)
PROVIDER_PRICING: Dict[VLMProvider, ProviderPricing] = {
    VLMProvider.OPENAI: ProviderPricing(
        input_cost=0.0025,  # GPT-4o
        output_cost=0.0100,
        image_cost=0.00765  # per image
    ),
    VLMProvider.ANTHROPIC: ProviderPricing(
        input_cost=0.0015,  # Claude 3.5 Sonnet
        output_cost=0.0075,
        image_cost=0.00375
    ),
    VLMProvider.GOOGLE: ProviderPricing(
        input_cost=0.00125,  # Gemini 1.5 Pro
        output_cost=0.0050,
        image_cost=0.0025
    ),
    VLMProvider.AZURE: ProviderPricing(
        input_cost=0.0025,  # GPT-4o via Azure
        output_cost=0.0100,
        image_cost=0.00765
    ),
    VLMProvider.MOCK: ProviderPricing(
        input_cost=0.0,
        output_cost=0.0,
        image_cost=0.0
    ),
}


# Provider capabilities
PROVIDER_CAPABILITIES: Dict[VLMProvider, ProviderCapabilities] = {
    VLMProvider.OPENAI: ProviderCapabilities(
        max_image_size=2048,
        supported_formats=["png", "jpeg", "webp", "gif"],
        supports_streaming=True,
        supports_function_calling=True,
        supports_system_messages=True,
        max_context_tokens=128000,
        recommended_for=["general_analysis", "detailed_insights", "cost_sensitive"]
    ),
    VLMProvider.ANTHROPIC: ProviderCapabilities(
        max_image_size=4096,
        supported_formats=["png", "jpeg", "webp", "gif"],
        supports_streaming=True,
        supports_function_calling=True,
        supports_system_messages=True,
        max_context_tokens=200000,
        recommended_for=["complex_analysis", "large_images", "high_accuracy"]
    ),
    VLMProvider.GOOGLE: ProviderCapabilities(
        max_image_size=8192,
        supported_formats=["png", "jpeg", "webp", "heic"],
        supports_streaming=True,
        supports_function_calling=True,
        supports_system_messages=True,
        max_context_tokens=1000000,
        recommended_for=["large_context", "multi_image", "video_analysis"]
    ),
    VLMProvider.AZURE: ProviderCapabilities(
        max_image_size=2048,
        supported_formats=["png", "jpeg", "webp", "gif"],
        supports_streaming=True,
        supports_function_calling=True,
        supports_system_messages=True,
        max_context_tokens=128000,
        recommended_for=["enterprise", "compliance", "regional_deployment"]
    ),
    VLMProvider.MOCK: ProviderCapabilities(
        max_image_size=1024,
        supported_formats=["png", "jpeg"],
        supports_streaming=False,
        supports_function_calling=False,
        supports_system_messages=False,
        max_context_tokens=4096,
        recommended_for=["testing", "development"]
    ),
}


# ============================================================================
# Provider-Specific Prompt Templates
# ============================================================================

PROVIDER_PROMPT_TEMPLATES: Dict[VLMProvider, Dict[str, str]] = {
    VLMProvider.OPENAI: {
        "base": """You are an expert UX/UI analyst specializing in heatmap interpretation and cognitive friction detection.

The heatmap overlay shows user interaction patterns:
- Red/hot areas: High interaction frequency or long dwell time
- Yellow/warm areas: Moderate interaction
- Blue/cool areas: Low or no interaction

Analyze the image and provide insights in JSON format:
{{
    "summary": "Brief overall analysis (1-2 sentences)",
    "insights": ["Key observation 1", "Key observation 2", "Key observation 3"],
    "friction_points": ["Area with friction 1", "Area with friction 2"],
    "recommendations": ["Actionable suggestion 1", "Actionable suggestion 2"],
    "confidence": 0.0-1.0
}}

Focus on identifying:
1. Unexpected interaction patterns
2. Areas of user confusion or hesitation
3. Elements that may be causing cognitive overload
4. Opportunities for UI optimization
""",
        "layout_analysis": "Focus specifically on layout patterns, element placement, and spatial relationships between UI components.",
        "interaction_patterns": "Analyze the user flow, interaction sequences, and how users navigate through the interface.",
        "friction_detection": "Identify specific areas where users struggle, hesitate, or show signs of confusion based on interaction patterns.",
        "heatmap_interpretation": "Interpret the heatmap intensity, distribution, and what it reveals about user behavior.",
        "comprehensive": "Provide a complete analysis covering layout, interactions, friction points, and actionable recommendations.",
    },
    VLMProvider.ANTHROPIC: {
        "base": """You are Claude, an expert UX/UI analyst with deep expertise in heatmap interpretation and cognitive friction analysis.

The heatmap overlay visualizes user interaction patterns:
- Red/hot areas: High interaction frequency or long dwell time
- Yellow/warm areas: Moderate interaction
- Blue/cool areas: Low or no interaction

Please analyze the image and provide your insights in JSON format:
{{
    "summary": "Concise overall analysis",
    "insights": ["Observation 1", "Observation 2", "Observation 3"],
    "friction_points": ["Friction area 1", "Friction area 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "confidence": 0.0-1.0
}}

Your analysis should consider:
- Visual hierarchy and attention flow
- Cognitive load implications
- Accessibility considerations
- Design system consistency
""",
        "layout_analysis": "Examine the layout structure, grid alignment, whitespace usage, and how elements are positioned relative to each other.",
        "interaction_patterns": "Trace the user journey through the interface, noting interaction sequences and navigation patterns.",
        "friction_detection": "Pinpoint specific UI elements or areas where users experience difficulty, confusion, or hesitation.",
        "heatmap_interpretation": "Analyze the heatmap data to understand user behavior patterns, attention distribution, and engagement levels.",
        "comprehensive": "Deliver a thorough analysis encompassing layout, interactions, friction points, with detailed, actionable recommendations.",
    },
    VLMProvider.GOOGLE: {
        "base": """You are a specialized UX/UI analyst for heatmap interpretation and cognitive friction detection.

The heatmap shows user interaction intensity:
- Red/hot: High interaction or dwell time
- Yellow/warm: Moderate interaction
- Blue/cool: Low interaction

Provide analysis in JSON format:
{{
    "summary": "Brief analysis summary",
    "insights": ["Insight 1", "Insight 2", "Insight 3"],
    "friction_points": ["Friction point 1", "Friction point 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "confidence": 0.0-1.0
}}

Key focus areas:
1. Interaction hotspots and coldspots
2. User journey optimization
3. UI element effectiveness
4. Conversion optimization opportunities
""",
        "layout_analysis": "Analyze the UI layout structure, element placement, and spatial organization.",
        "interaction_patterns": "Study user interaction flows, sequences, and behavioral patterns.",
        "friction_detection": "Identify friction points where users encounter difficulties or confusion.",
        "heatmap_interpretation": "Interpret heatmap intensity, distribution, and behavioral insights.",
        "comprehensive": "Provide complete analysis covering all aspects with actionable insights.",
    },
    VLMProvider.AZURE: {
        "base": """You are an enterprise UX/UI analyst specializing in heatmap interpretation for enterprise applications.

The heatmap overlay shows:
- Red/hot areas: High interaction frequency or long dwell time
- Yellow/warm areas: Moderate interaction
- Blue/cool areas: Low or no interaction

Analyze and provide JSON output:
{{
    "summary": "Analysis summary",
    "insights": ["Insight 1", "Insight 2", "Insight 3"],
    "friction_points": ["Friction point 1", "Friction point 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "confidence": 0.0-1.0
}}

Consider enterprise-specific factors:
- Workflow efficiency
- Data entry optimization
- Compliance and accessibility
- User training implications
""",
        "layout_analysis": "Focus on enterprise application layout patterns, data density, and information architecture.",
        "interaction_patterns": "Analyze enterprise user workflows, task completion patterns, and efficiency metrics.",
        "friction_detection": "Identify enterprise workflow friction points, data entry bottlenecks, and usability issues.",
        "heatmap_interpretation": "Interpret heatmap data in the context of enterprise user behavior and productivity.",
        "comprehensive": "Provide comprehensive enterprise-focused analysis with actionable recommendations.",
    },
    VLMProvider.MOCK: {
        "base": "Mock analysis prompt for testing purposes.",
        "layout_analysis": "Mock layout analysis.",
        "interaction_patterns": "Mock interaction patterns analysis.",
        "friction_detection": "Mock friction detection.",
        "heatmap_interpretation": "Mock heatmap interpretation.",
        "comprehensive": "Mock comprehensive analysis.",
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VLMConfig:
    """Configuration for VLM analysis."""
    provider: VLMProvider = VLMProvider.OPENAI
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: Optional[float] = None  # None = use provider optimal
    max_tokens: Optional[int] = None  # None = use provider optimal
    base_url: Optional[str] = None
    timeout: Optional[int] = None  # None = use provider optimal
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    api_version: Optional[str] = None  # For Azure
    deployment_name: Optional[str] = None  # For Azure
    endpoint: Optional[str] = None  # For Azure
    enable_cost_tracking: bool = True
    enable_metrics: bool = True


@dataclass
class VLMAnalysisResult:
    """Result of VLM analysis."""
    summary: str
    insights: List[str] = field(default_factory=list)
    friction_points: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    raw_response: str = ""
    cost_estimate: float = 0.0  # USD
    latency_ms: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "insights": self.insights,
            "friction_points": self.friction_points,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
        }


@dataclass
class VLMPerformanceMetrics:
    """Performance metrics for a VLM provider."""
    provider: VLMProvider
    model: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)

    def record_success(self, tokens: int, cost: float, latency_ms: float) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.successful_requests

    def record_failure(self, error_type: str = "unknown") -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.get_cache_hit_rate(),
            "error_types": self.error_types,
        }


# ============================================================================
# Provider Selection Recommendations
# ============================================================================

class ProviderRecommender:
    """Recommends the best VLM provider based on use case."""

    @staticmethod
    def recommend(
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
        priority: str = "balanced",  # "cost", "speed", "quality", "balanced"
        image_size: Optional[int] = None,
        budget_constraint: Optional[float] = None  # max cost per request in USD
    ) -> Tuple[VLMProvider, str]:
        """
        Recommend the best provider for the given use case.

        Returns:
            Tuple of (provider, reason)
        """
        providers = []

        for provider, capabilities in PROVIDER_CAPABILITIES.items():
            if provider == VLMProvider.MOCK:
                continue

            score = 0
            reasons = []

            # Check if analysis type is recommended
            if analysis_type.value in capabilities.recommended_for or "general_analysis" in capabilities.recommended_for:
                score += 10
                reasons.append("recommended for this analysis type")

            # Priority-based scoring
            if priority == "cost":
                pricing = PROVIDER_PRICING[provider]
                cost_score = 1 / (pricing.input_cost + pricing.output_cost + 0.001)
                score += cost_score * 5
                reasons.append("cost-effective")
            elif priority == "quality":
                if provider == VLMProvider.ANTHROPIC:
                    score += 15
                    reasons.append("highest quality output")
                elif provider == VLMProvider.OPENAI:
                    score += 12
                    reasons.append("high quality output")
            elif priority == "speed":
                if provider == VLMProvider.OPENAI:
                    score += 10
                    reasons.append("fast response times")
            elif priority == "balanced":
                score += 5
                reasons.append("balanced performance")

            # Image size considerations
            if image_size and image_size > capabilities.max_image_size:
                score -= 20  # Penalty for unsupported size
                reasons.append(f"image size exceeds max {capabilities.max_image_size}px")

            # Budget constraint
            if budget_constraint:
                pricing = PROVIDER_PRICING[provider]
                estimated_cost = (pricing.input_cost + pricing.output_cost) * 2  # Rough estimate
                if estimated_cost > budget_constraint:
                    score -= 15
                    reasons.append("exceeds budget constraint")

            providers.append((provider, score, ", ".join(reasons)))

        # Sort by score and return best
        providers.sort(key=lambda x: x[1], reverse=True)
        if providers:
            return providers[0][0], providers[0][2]

        return VLMProvider.OPENAI, "default recommendation"

    @staticmethod
    def get_provider_comparison() -> Dict[str, Any]:
        """Get comparison of all providers."""
        comparison = {}

        for provider in [VLMProvider.OPENAI, VLMProvider.ANTHROPIC, VLMProvider.GOOGLE, VLMProvider.AZURE]:
            config = PROVIDER_CONFIGS[provider]
            pricing = PROVIDER_PRICING[provider]
            capabilities = PROVIDER_CAPABILITIES[provider]

            comparison[provider.value] = {
                "optimal_temperature": config.temperature,
                "optimal_max_tokens": config.max_tokens,
                "input_cost_per_1k_tokens": pricing.input_cost,
                "output_cost_per_1k_tokens": pricing.output_cost,
                "image_cost": pricing.image_cost,
                "max_image_size": capabilities.max_image_size,
                "max_context_tokens": capabilities.max_context_tokens,
                "recommended_for": capabilities.recommended_for,
                "supports_streaming": capabilities.supports_streaming,
            }

        return comparison


# ============================================================================
# Retry and Error Handling Decorator
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_errors: Optional[List[str]] = None
):
    """
    Decorator for retrying failed API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        retryable_errors: List of error types that should trigger a retry
    """
    if retryable_errors is None:
        retryable_errors = [
            "timeout", "rate_limit", "server_error", "connection",
            "429", "500", "502", "503", "504"
        ]

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check if error is retryable
                    is_retryable = any(
                        err in error_str for err in retryable_errors
                    )

                    if not is_retryable or attempt == max_retries:
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

            raise last_error

        return wrapper
    return decorator


# ============================================================================
# VLM Client Base Class
# ============================================================================

class VLMClient(ABC):
    """Abstract base class for VLM clients with provider-specific optimizations."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._cache: Dict[str, tuple] = {}
        self._metrics = VLMPerformanceMetrics(
            provider=config.provider,
            model=config.model
        )
        self._optimal_config = PROVIDER_CONFIGS.get(config.provider, PROVIDER_CONFIGS[VLMProvider.OPENAI])
        self._pricing = PROVIDER_PRICING.get(config.provider, PROVIDER_PRICING[VLMProvider.OPENAI])

    @abstractmethod
    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """Analyze an image with the VLM."""
        pass

    def _get_cache_key(self, image_data: bytes, prompt: str) -> str:
        """Generate cache key."""
        import hashlib
        combined = image_data + prompt.encode()
        return hashlib.sha256(combined).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[VLMAnalysisResult]:
        """Get cached result if available and not expired."""
        if not self.config.enable_caching:
            return None
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if (timestamp + self.config.cache_ttl) > self._get_current_time():
                self._metrics.record_cache_hit()
                return result
            else:
                del self._cache[cache_key]
        self._metrics.record_cache_miss()
        return None

    def _cache_result(self, cache_key: str, result: VLMAnalysisResult) -> None:
        """Cache analysis result."""
        if self.config.enable_caching:
            self._cache[cache_key] = (result, self._get_current_time())

    @staticmethod
    def _get_current_time() -> int:
        """Get current timestamp in seconds."""
        return int(time.time())

    def _get_effective_config(self) -> ProviderOptimalConfig:
        """Get the effective configuration (user config overrides optimal)."""
        return ProviderOptimalConfig(
            temperature=self.config.temperature if self.config.temperature is not None else self._optimal_config.temperature,
            max_tokens=self.config.max_tokens if self.config.max_tokens is not None else self._optimal_config.max_tokens,
            top_p=self._optimal_config.top_p,
            top_k=self._optimal_config.top_k,
            image_detail=self._optimal_config.image_detail,
            timeout=self.config.timeout if self.config.timeout is not None else self._optimal_config.timeout,
            max_retries=self._optimal_config.max_retries,
            retry_delay=self._optimal_config.retry_delay,
            use_streaming=self._optimal_config.use_streaming,
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int, image_count: int = 1) -> float:
        """Calculate estimated cost in USD."""
        if not self.config.enable_cost_tracking:
            return 0.0

        input_cost = (input_tokens / 1000) * self._pricing.input_cost
        output_cost = (output_tokens / 1000) * self._pricing.output_cost
        image_cost = image_count * self._pricing.image_cost

        return input_cost + output_cost + image_cost

    def _build_provider_prompt(self, user_prompt: str, analysis_type: AnalysisType) -> str:
        """Build provider-specific prompt."""
        templates = PROVIDER_PROMPT_TEMPLATES.get(self.config.provider, PROVIDER_PROMPT_TEMPLATES[VLMProvider.OPENAI])
        base = templates.get("base", "")
        type_specific = templates.get(analysis_type.value, "")

        return f"{base}\n\n{type_specific}\n\n{user_prompt}"

    def get_metrics(self) -> VLMPerformanceMetrics:
        """Get performance metrics for this client."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = VLMPerformanceMetrics(
            provider=self.config.provider,
            model=self.config.model
        )


# ============================================================================
# OpenAI VLM Client
# ============================================================================

class OpenAIVLMClient(VLMClient):
    """OpenAI VLM client using GPT-4 Vision with provider-specific optimizations."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self._get_effective_config().timeout
            )
            logger.info(f"OpenAI VLM client initialized with model: {self.config.model}")
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._client = None

    @retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """Analyze image with OpenAI VLM."""
        if not self._client:
            result = self._get_error_result("OpenAI client not initialized")
            self._metrics.record_failure("client_not_initialized")
            return result

        cache_key = self._get_cache_key(image_data, prompt)
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        start_time = time.time()
        retry_count = 0

        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")

            # Get effective config
            effective_config = self._get_effective_config()

            # Build messages with provider-specific prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_provider_prompt(prompt, analysis_type)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": effective_config.image_detail
                            }
                        }
                    ]
                }
            ]

            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=effective_config.temperature,
                max_tokens=effective_config.max_tokens,
                top_p=effective_config.top_p
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0

            cost = self._calculate_cost(input_tokens, output_tokens)

            result = self._parse_response(content, total_tokens)
            result.provider = VLMProvider.OPENAI.value
            result.model = self.config.model
            result.raw_response = content
            result.cost_estimate = cost
            result.latency_ms = latency_ms
            result.retry_count = retry_count

            self._metrics.record_success(total_tokens, cost, latency_ms)
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"OpenAI VLM analysis failed: {e}")
            self._metrics.record_failure(type(e).__name__)

            result = self._get_error_result(str(e))
            result.latency_ms = latency_ms
            result.retry_count = retry_count
            return result

    def _parse_response(self, content: str, tokens_used: int) -> VLMAnalysisResult:
        """Parse the VLM response."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return VLMAnalysisResult(
                    summary=data.get("summary", content[:200]),
                    insights=data.get("insights", []),
                    friction_points=data.get("friction_points", []),
                    recommendations=data.get("recommendations", []),
                    confidence=data.get("confidence", 0.5),
                    tokens_used=tokens_used
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        return VLMAnalysisResult(
            summary=content[:500],
            insights=[content[:200]],
            tokens_used=tokens_used
        )

    def _get_error_result(self, error_message: str) -> VLMAnalysisResult:
        """Return an error result."""
        return VLMAnalysisResult(
            summary=f"Analysis failed: {error_message}",
            confidence=0.0,
            provider=VLMProvider.OPENAI.value
        )


# ============================================================================
# Anthropic VLM Client
# ============================================================================

class AnthropicVLMClient(VLMClient):
    """Anthropic Claude Vision client with provider-specific optimizations."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=self._get_effective_config().timeout
            )
            logger.info(f"Anthropic VLM client initialized with model: {self.config.model}")
        except ImportError:
            logger.warning("Anthropic package not installed. Install with: pip install anthropic")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self._client = None

    @retry_with_backoff(max_retries=3, initial_delay=1.5, backoff_factor=2.0)
    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """Analyze image with Claude Vision."""
        if not self._client:
            result = self._get_error_result("Anthropic client not initialized")
            self._metrics.record_failure("client_not_initialized")
            return result

        cache_key = self._get_cache_key(image_data, prompt)
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        start_time = time.time()
        retry_count = 0

        try:
            base64_image = base64.b64encode(image_data).decode("utf-8")
            full_prompt = self._build_provider_prompt(prompt, analysis_type)

            effective_config = self._get_effective_config()

            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=effective_config.max_tokens,
                temperature=effective_config.temperature,
                top_p=effective_config.top_p,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens

            cost = self._calculate_cost(input_tokens, output_tokens)

            result = self._parse_response(content, total_tokens)
            result.provider = VLMProvider.ANTHROPIC.value
            result.model = self.config.model
            result.raw_response = content
            result.cost_estimate = cost
            result.latency_ms = latency_ms
            result.retry_count = retry_count

            self._metrics.record_success(total_tokens, cost, latency_ms)
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Anthropic VLM analysis failed: {e}")
            self._metrics.record_failure(type(e).__name__)

            result = self._get_error_result(str(e))
            result.latency_ms = latency_ms
            result.retry_count = retry_count
            return result

    def _parse_response(self, content: str, tokens_used: int) -> VLMAnalysisResult:
        """Parse the VLM response."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return VLMAnalysisResult(
                    summary=data.get("summary", content[:200]),
                    insights=data.get("insights", []),
                    friction_points=data.get("friction_points", []),
                    recommendations=data.get("recommendations", []),
                    confidence=data.get("confidence", 0.5),
                    tokens_used=tokens_used
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        return VLMAnalysisResult(
            summary=content[:500],
            insights=[content[:200]],
            tokens_used=tokens_used
        )

    def _get_error_result(self, error_message: str) -> VLMAnalysisResult:
        """Return an error result."""
        return VLMAnalysisResult(
            summary=f"Analysis failed: {error_message}",
            confidence=0.0,
            provider=VLMProvider.ANTHROPIC.value
        )


# ============================================================================
# Google Gemini VLM Client
# ============================================================================

class GoogleVLMClient(VLMClient):
    """Google Gemini Vision client with provider-specific optimizations."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Google client."""
        try:
            import google.generativeai as genai
            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model)
            logger.info(f"Google VLM client initialized with model: {self.config.model}")
        except ImportError:
            logger.warning("Google Generative AI package not installed. Install with: pip install google-generativeai")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}")
            self._client = None

    @retry_with_backoff(max_retries=2, initial_delay=1.0, backoff_factor=2.0)
    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """Analyze image with Gemini Vision."""
        if not self._client:
            result = self._get_error_result("Google client not initialized")
            self._metrics.record_failure("client_not_initialized")
            return result

        cache_key = self._get_cache_key(image_data, prompt)
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        start_time = time.time()
        retry_count = 0

        try:
            import google.generativeai.types as types

            # Create image part
            image_part = types.image_from_bytes(image_data)

            # Build provider-specific prompt
            full_prompt = self._build_provider_prompt(prompt, analysis_type)

            effective_config = self._get_effective_config()

            # Generate content
            generation_config = genai.types.GenerationConfig(
                temperature=effective_config.temperature,
                max_output_tokens=effective_config.max_tokens,
                top_p=effective_config.top_p,
                top_k=effective_config.top_k,
            )

            response = await self._client.generate_content_async(
                [full_prompt, image_part],
                generation_config=generation_config
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.text if response.text else ""
            # Estimate tokens (Gemini doesn't always return token counts)
            estimated_tokens = len(content.split()) * 1.3  # Rough estimate

            cost = self._calculate_cost(int(estimated_tokens * 0.7), int(estimated_tokens * 0.3))

            result = self._parse_response(content, int(estimated_tokens))
            result.provider = VLMProvider.GOOGLE.value
            result.model = self.config.model
            result.raw_response = content
            result.cost_estimate = cost
            result.latency_ms = latency_ms
            result.retry_count = retry_count

            self._metrics.record_success(int(estimated_tokens), cost, latency_ms)
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Google VLM analysis failed: {e}")
            self._metrics.record_failure(type(e).__name__)

            result = self._get_error_result(str(e))
            result.latency_ms = latency_ms
            result.retry_count = retry_count
            return result

    def _parse_response(self, content: str, tokens_used: int) -> VLMAnalysisResult:
        """Parse the VLM response."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return VLMAnalysisResult(
                    summary=data.get("summary", content[:200]),
                    insights=data.get("insights", []),
                    friction_points=data.get("friction_points", []),
                    recommendations=data.get("recommendations", []),
                    confidence=data.get("confidence", 0.5),
                    tokens_used=tokens_used
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        return VLMAnalysisResult(
            summary=content[:500],
            insights=[content[:200]],
            tokens_used=tokens_used
        )

    def _get_error_result(self, error_message: str) -> VLMAnalysisResult:
        """Return an error result."""
        return VLMAnalysisResult(
            summary=f"Analysis failed: {error_message}",
            confidence=0.0,
            provider=VLMProvider.GOOGLE.value
        )


# ============================================================================
# Azure OpenAI VLM Client
# ============================================================================

class AzureVLMClient(VLMClient):
    """Azure OpenAI Vision client with provider-specific optimizations."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Azure OpenAI client."""
        try:
            from openai import AsyncAzureOpenAI
            api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = self.config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = self.config.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            deployment_name = self.config.deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

            if not api_key:
                raise ValueError("Azure OpenAI API key not provided")
            if not endpoint:
                raise ValueError("Azure OpenAI endpoint not provided")
            if not deployment_name:
                raise ValueError("Azure OpenAI deployment name not provided")

            self._client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                timeout=self._get_effective_config().timeout
            )
            self._deployment_name = deployment_name
            logger.info(f"Azure OpenAI VLM client initialized with deployment: {deployment_name}")
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            self._client = None

    @retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """Analyze image with Azure OpenAI VLM."""
        if not self._client:
            result = self._get_error_result("Azure OpenAI client not initialized")
            self._metrics.record_failure("client_not_initialized")
            return result

        cache_key = self._get_cache_key(image_data, prompt)
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        start_time = time.time()
        retry_count = 0

        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")

            # Get effective config
            effective_config = self._get_effective_config()

            # Build messages with provider-specific prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_provider_prompt(prompt, analysis_type)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": effective_config.image_detail
                            }
                        }
                    ]
                }
            ]

            response = await self._client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                temperature=effective_config.temperature,
                max_tokens=effective_config.max_tokens,
                top_p=effective_config.top_p
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0

            cost = self._calculate_cost(input_tokens, output_tokens)

            result = self._parse_response(content, total_tokens)
            result.provider = VLMProvider.AZURE.value
            result.model = self._deployment_name
            result.raw_response = content
            result.cost_estimate = cost
            result.latency_ms = latency_ms
            result.retry_count = retry_count

            self._metrics.record_success(total_tokens, cost, latency_ms)
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Azure OpenAI VLM analysis failed: {e}")
            self._metrics.record_failure(type(e).__name__)

            result = self._get_error_result(str(e))
            result.latency_ms = latency_ms
            result.retry_count = retry_count
            return result

    def _parse_response(self, content: str, tokens_used: int) -> VLMAnalysisResult:
        """Parse the VLM response."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return VLMAnalysisResult(
                    summary=data.get("summary", content[:200]),
                    insights=data.get("insights", []),
                    friction_points=data.get("friction_points", []),
                    recommendations=data.get("recommendations", []),
                    confidence=data.get("confidence", 0.5),
                    tokens_used=tokens_used
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        return VLMAnalysisResult(
            summary=content[:500],
            insights=[content[:200]],
            tokens_used=tokens_used
        )

    def _get_error_result(self, error_message: str) -> VLMAnalysisResult:
        """Return an error result."""
        return VLMAnalysisResult(
            summary=f"Analysis failed: {error_message}",
            confidence=0.0,
            provider=VLMProvider.AZURE.value
        )


# ============================================================================
# Mock VLM Client
# ============================================================================

class MockVLMClient(VLMClient):
    """Mock VLM client for testing without API access."""

    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """Return mock analysis results."""
        start_time = time.time()

        # Simulate some processing time
        await asyncio.sleep(0.1)

        latency_ms = (time.time() - start_time) * 1000

        result = VLMAnalysisResult(
            summary="Mock VLM analysis: Heatmap shows concentrated interactions in the center region.",
            insights=[
                "Primary interaction hotspot detected in the central navigation area",
                "Secondary activity observed in the right sidebar controls",
                "Low engagement in the footer section"
            ],
            friction_points=[
                "Users appear to hover repeatedly over the submit button",
                "Search input shows multiple focus events without submission"
            ],
            recommendations=[
                "Consider adding tooltips to clarify button functions",
                "Optimize search field for better discoverability",
                "Review footer content for relevance"
            ],
            confidence=0.8,
            provider=VLMProvider.MOCK.value,
            model="mock-model-v1",
            tokens_used=150,
            cost_estimate=0.0,
            latency_ms=latency_ms,
            retry_count=0
        )

        self._metrics.record_success(150, 0.0, latency_ms)
        return result


# ============================================================================
# Main VLM Analyzer
# ============================================================================

class VLMAnalyzer:
    """Main VLM analyzer class with provider-specific optimizations."""

    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize VLM analyzer.

        Args:
            config: VLM configuration. If None, loads from environment variables.
        """
        if config is None:
            config = self._load_config_from_env()
        self.config = config
        self._client = self._create_client()

    @staticmethod
    def _load_config_from_env() -> VLMConfig:
        """Load VLM configuration from environment variables."""
        provider_str = os.getenv("ICR_VLM_PROVIDER", "openai").lower()
        provider = VLMProvider(provider_str) if provider_str in [p.value for p in VLMProvider] else VLMProvider.OPENAI

        return VLMConfig(
            provider=provider,
            model=os.getenv("ICR_VLM_MODEL", "gpt-4o"),
            api_key=os.getenv("ICR_VLM_API_KEY"),
            temperature=float(os.getenv("ICR_VLM_TEMPERATURE", "0.0")) if os.getenv("ICR_VLM_TEMPERATURE") else None,
            max_tokens=int(os.getenv("ICR_VLM_MAX_TOKENS", "0")) if os.getenv("ICR_VLM_MAX_TOKENS") else None,
            base_url=os.getenv("ICR_VLM_BASE_URL"),
            timeout=int(os.getenv("ICR_VLM_TIMEOUT", "0")) if os.getenv("ICR_VLM_TIMEOUT") else None,
            enable_caching=os.getenv("ICR_VLM_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("ICR_VLM_CACHE_TTL", "3600")),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            enable_cost_tracking=os.getenv("ICR_VLM_COST_TRACKING", "true").lower() == "true",
            enable_metrics=os.getenv("ICR_VLM_METRICS", "true").lower() == "true",
        )

    def _create_client(self) -> Optional[VLMClient]:
        """Create the appropriate VLM client based on provider."""
        client_classes = {
            VLMProvider.OPENAI: OpenAIVLMClient,
            VLMProvider.ANTHROPIC: AnthropicVLMClient,
            VLMProvider.GOOGLE: GoogleVLMClient,
            VLMProvider.AZURE: AzureVLMClient,
            VLMProvider.MOCK: MockVLMClient,
        }

        client_class = client_classes.get(self.config.provider)
        if client_class:
            return client_class(self.config)

        logger.warning(f"Unsupported VLM provider: {self.config.provider}, using mock client")
        return MockVLMClient(self.config)

    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.LAYOUT_ANALYSIS
    ) -> VLMAnalysisResult:
        """
        Analyze an image with the configured VLM.

        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt
            analysis_type: Type of analysis to perform

        Returns:
            VLMAnalysisResult containing the analysis
        """
        if not self._client:
            return VLMAnalysisResult(
                summary="VLM client not available",
                confidence=0.0
            )

        return await self._client.analyze(image_data, prompt, analysis_type)

    def is_configured(self) -> bool:
        """Check if VLM is properly configured."""
        if self.config.provider == VLMProvider.MOCK:
            return True

        if self.config.provider == VLMProvider.OPENAI:
            return bool(self.config.api_key or os.getenv("OPENAI_API_KEY"))

        if self.config.provider == VLMProvider.ANTHROPIC:
            return bool(self.config.api_key or os.getenv("ANTHROPIC_API_KEY"))

        if self.config.provider == VLMProvider.GOOGLE:
            return bool(self.config.api_key or os.getenv("GOOGLE_API_KEY"))

        if self.config.provider == VLMProvider.AZURE:
            return bool(
                (self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")) and
                (self.config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")) and
                (self.config.deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
            )

        return False

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information (without sensitive data)."""
        info = {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "base_url": self.config.base_url,
            "enable_caching": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl,
            "enable_cost_tracking": self.config.enable_cost_tracking,
            "enable_metrics": self.config.enable_metrics,
            "configured": self.is_configured(),
        }

        # Add optimal config for reference
        if self._client:
            optimal = self._client._get_effective_config()
            info["effective_temperature"] = optimal.temperature
            info["effective_max_tokens"] = optimal.max_tokens
            info["effective_timeout"] = optimal.timeout

        return info

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics for the current client."""
        if self._client and self.config.enable_metrics:
            return self._client.get_metrics().to_dict()
        return None

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        if self._client:
            self._client.reset_metrics()

    @staticmethod
    def get_provider_capabilities() -> Dict[str, Any]:
        """Get capabilities for all providers."""
        return {
            provider.value: asdict(capabilities)
            for provider, capabilities in PROVIDER_CAPABILITIES.items()
        }

    @staticmethod
    def get_provider_pricing() -> Dict[str, Any]:
        """Get pricing information for all providers."""
        return {
            provider.value: asdict(pricing)
            for provider, pricing in PROVIDER_PRICING.items()
        }

    @staticmethod
    def recommend_provider(
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
        priority: str = "balanced",
        image_size: Optional[int] = None,
        budget_constraint: Optional[float] = None
    ) -> Tuple[str, str]:
        """
        Get a provider recommendation.

        Returns:
            Tuple of (provider_name, reason)
        """
        provider, reason = ProviderRecommender.recommend(
            analysis_type=analysis_type,
            priority=priority,
            image_size=image_size,
            budget_constraint=budget_constraint
        )
        return provider.value, reason

    @staticmethod
    def get_provider_comparison() -> Dict[str, Any]:
        """Get comparison of all providers."""
        return ProviderRecommender.get_provider_comparison()


# ============================================================================
# Convenience Functions
# ============================================================================

async def analyze_heatmap(
    image_data: bytes,
    prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick convenience function to analyze a heatmap image.

    Args:
        image_data: Raw image bytes
        prompt: Custom prompt (optional)
        provider: VLM provider override (optional)
        model: Model override (optional)

    Returns:
        Dictionary with analysis results
    """
    config = VLMAnalyzer._load_config_from_env()
    if provider:
        config.provider = VLMProvider(provider)
    if model:
        config.model = model

    analyzer = VLMAnalyzer(config)

    if not analyzer.is_configured():
        return {
            "error": "VLM not configured",
            "message": "Please set appropriate API key environment variable"
        }

    default_prompt = (
        "Analyze this UI snapshot with an interaction heatmap overlay.\n"
        "Identify cognitive friction points, confusing placements, and areas of repeated interaction.\n"
        "Provide concise, actionable UI refinement suggestions."
    )

    result = await analyzer.analyze(image_data, prompt or default_prompt)
    return result.to_dict()


async def analyze_with_recommendation(
    image_data: bytes,
    prompt: str,
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
    priority: str = "balanced"
) -> Dict[str, Any]:
    """
    Analyze heatmap with automatic provider recommendation.

    Args:
        image_data: Raw image bytes
        prompt: Analysis prompt
        analysis_type: Type of analysis
        priority: Priority for provider selection (cost, speed, quality, balanced)

    Returns:
        Dictionary with analysis results and provider info
    """
    provider, reason = VLMAnalyzer.recommend_provider(
        analysis_type=analysis_type,
        priority=priority
    )

    config = VLMAnalyzer._load_config_from_env()
    config.provider = VLMProvider(provider)

    analyzer = VLMAnalyzer(config)

    if not analyzer.is_configured():
        return {
            "error": "VLM not configured",
            "message": f"Recommended provider {provider} is not configured",
            "recommended_provider": provider,
            "recommendation_reason": reason
        }

    result = await analyzer.analyze(image_data, prompt, analysis_type)

    return {
        **result.to_dict(),
        "recommended_provider": provider,
        "recommendation_reason": reason
    }


# ============================================================================
# Exported symbols
# ============================================================================

__all__ = [
    # Enums
    "VLMProvider",
    "AnalysisType",
    # Data classes
    "VLMConfig",
    "VLMAnalysisResult",
    "VLMPerformanceMetrics",
    "ProviderOptimalConfig",
    "ProviderPricing",
    "ProviderCapabilities",
    # Clients
    "VLMClient",
    "OpenAIVLMClient",
    "AnthropicVLMClient",
    "GoogleVLMClient",
    "AzureVLMClient",
    "MockVLMClient",
    # Main analyzer
    "VLMAnalyzer",
    # Utilities
    "analyze_heatmap",
    "analyze_with_recommendation",
    "ProviderRecommender",
    # Constants
    "PROVIDER_CONFIGS",
    "PROVIDER_PRICING",
    "PROVIDER_CAPABILITIES",
    "PROVIDER_PROMPT_TEMPLATES",
]
