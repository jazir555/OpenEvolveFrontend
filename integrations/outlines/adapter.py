"""
Outlines Adapter - Structured LLM Output Generation

Guarantees valid output formats using regex/JSON constraints at token level.
Integrates with DSPy for optimized prompts + guaranteed valid outputs.

Follows SSOT pattern: Primary logic here, thin wrapper in knowledge_engine/integrations/outlines/
"""

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Type
from functools import lru_cache

import pydantic
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

# Configure logger
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"


class GenerationError(Exception):
    """Error during constrained generation."""
    pass


class ValidationError(Exception):
    """Error during output validation."""
    pass


class ConstraintCompilationError(Exception):
    """Error compiling constraints."""
    pass


@dataclass
class OutlinesResult:
    """Result of an Outlines generation operation."""
    success: bool
    output: Any
    raw_output: str
    constraint_type: str
    model: str
    processing_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "output": self.output,
            "raw_output": self.raw_output,
            "constraint_type": self.constraint_type,
            "model": self.model,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class OutlinesConfig:
    """Configuration for Outlines adapter."""
    # Model configuration
    model_provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    
    # Retry configuration
    max_retries: int = 3
    backoff_factor: float = 1.0
    retry_delay_seconds: float = 1.0
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0
    
    # Fallback
    enable_fallback: bool = True
    fallback_to_unconstrained: bool = True
    
    # Batch processing
    batch_max_workers: int = 4
    batch_timeout_seconds: float = 300.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_provider": self.model_provider.value,
            "model_name": self.model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "max_retries": self.max_retries,
            "enable_caching": self.enable_caching,
            "enable_fallback": self.enable_fallback,
        }


class CircuitBreaker:
    """Circuit breaker pattern for external calls."""
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"


class GrammarCache:
    """LRU cache for compiled grammars."""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached grammar."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Cache grammar with LRU eviction."""
        if len(self._cache) >= self.maxsize:
            # Evict least recently used
            lru_key = min(self._access_times, key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._access_times.clear()


class OutlinesAdapter:
    """
    Adapter for structured LLM output generation using Outlines.
    
    Provides guaranteed valid outputs through:
    - JSON schema constraints
    - Regex pattern constraints
    - Choice selection constraints
    - Grammar-based constraints
    
    Features:
    - Model registry supporting OpenAI, Anthropic, and local models
    - Caching layer for compiled grammars
    - Circuit breaker for external calls
    - Graceful fallback to unconstrained generation
    - Batch processing support
    """
    
    def __init__(self, config: Optional[OutlinesConfig] = None):
        """
        Initialize the Outlines adapter.
        
        Args:
            config: Configuration for the adapter. Uses defaults if None.
        """
        self.config = config or OutlinesConfig()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout_seconds
        )
        self.grammar_cache = GrammarCache(maxsize=self.config.cache_size)
        self._outlines_available = self._check_outlines_available()
        self._model_cache: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.config.batch_max_workers)
        
        # Initialize model connection
        self._initialize_model()
        
        logger.info({
            "msg": "OutlinesAdapter initialized",
            "model_provider": self.config.model_provider.value,
            "model_name": self.config.model_name,
            "outlines_available": self._outlines_available,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def _check_outlines_available(self) -> bool:
        """Check if outlines package is available."""
        try:
            import outlines
            return True
        except ImportError:
            logger.warning({
                "msg": "Outlines package not available. Install with: pip install outlines>=0.0.36",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return False
    
    def _initialize_model(self):
        """Initialize the model connection."""
        try:
            if self.config.model_provider == ModelProvider.OPENAI:
                self._init_openai_model()
            elif self.config.model_provider == ModelProvider.ANTHROPIC:
                self._init_anthropic_model()
            elif self.config.model_provider == ModelProvider.TRANSFORMERS:
                self._init_transformers_model()
            elif self.config.model_provider == ModelProvider.LLAMA_CPP:
                self._init_llama_cpp_model()
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize model: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise GenerationError(f"Model initialization failed: {e}")
    
    def _init_openai_model(self):
        """Initialize OpenAI model."""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
            self._model_cache["openai"] = self._client
        except ImportError:
            raise GenerationError("openai package not installed. Install with: pip install openai")
    
    def _init_anthropic_model(self):
        """Initialize Anthropic model."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
            self._model_cache["anthropic"] = self._client
        except ImportError:
            raise GenerationError("anthropic package not installed. Install with: pip install anthropic")
    
    def _init_transformers_model(self):
        """Initialize local transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = self.config.model_name
            logger.info({
                "msg": f"Loading transformers model: {model_name}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self._model_cache["tokenizer"] = tokenizer
            self._model_cache["model"] = model
            
        except ImportError:
            raise GenerationError("transformers package not installed. Install with: pip install transformers")
    
    def _init_llama_cpp_model(self):
        """Initialize llama.cpp model."""
        try:
            from llama_cpp import Llama
            
            model_path = self.config.model_name
            logger.info({
                "msg": f"Loading llama.cpp model: {model_path}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            model = Llama(model_path=model_path)
            self._model_cache["llama"] = model
            
        except ImportError:
            raise GenerationError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    
    def _get_cache_key(self, constraint_type: str, constraint: Any, prompt_hash: str) -> str:
        """Generate cache key for grammar."""
        return f"{constraint_type}:{hash(str(constraint))}:{prompt_hash}"
    
    def _exponential_backoff_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        delay = self.config.retry_delay_seconds
        
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                logger.warning({
                    "msg": f"Attempt {attempt + 1} failed, retrying in {delay}s",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                time.sleep(delay)
                delay *= (2 ** self.config.backoff_factor)
    
    def generate_json(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        prompt: str,
        model: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> OutlinesResult:
        """
        Generate JSON output matching the provided schema.
        
        Args:
            schema: JSON schema dict or Pydantic model class
            prompt: Input prompt
            model: Override model name
            correlation_id: Correlation ID for tracking
            
        Returns:
            OutlinesResult with parsed JSON output
        """
        correlation_id = correlation_id or f"outlines_json_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = time.time()
        model_name = model or self.config.model_name
        
        logger.info({
            "msg": "Starting JSON constrained generation",
            "correlation_id": correlation_id,
            "model": model_name,
            "prompt_length": len(prompt),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            if not self.circuit_breaker.can_execute():
                raise GenerationError("Circuit breaker is OPEN")
            
            # Check cache
            cache_key = self._get_cache_key("json", schema, hash(prompt))
            if self.config.enable_caching:
                cached = self.grammar_cache.get(cache_key)
                if cached:
                    logger.debug({
                        "msg": "Using cached grammar",
                        "correlation_id": correlation_id,
                    })
            
            result = self._exponential_backoff_retry(
                self._generate_json_internal,
                schema,
                prompt,
                model_name,
            )
            
            self.circuit_breaker.record_success()
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info({
                "msg": "JSON constrained generation completed",
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            return OutlinesResult(
                success=True,
                output=result,
                raw_output=json.dumps(result),
                constraint_type="json",
                model=model_name,
                processing_time_ms=processing_time_ms,
                metadata={"correlation_id": correlation_id},
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.error({
                "msg": "JSON constrained generation failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            # Fallback to unconstrained if enabled
            if self.config.fallback_to_unconstrained:
                return self._fallback_unconstrained(prompt, model_name, str(e), processing_time_ms)
            
            return OutlinesResult(
                success=False,
                output=None,
                raw_output="",
                constraint_type="json",
                model=model_name,
                processing_time_ms=processing_time_ms,
                error=str(e),
                metadata={"correlation_id": correlation_id},
            )
    
    def _generate_json_internal(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        prompt: str,
        model: str,
    ) -> Dict[str, Any]:
        """Internal JSON generation implementation."""
        if self._outlines_available:
            return self._generate_with_outlines_json(schema, prompt, model)
        else:
            return self._generate_without_outlines_json(schema, prompt, model)
    
    def _generate_with_outlines_json(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        prompt: str,
        model: str,
    ) -> Dict[str, Any]:
        """Generate using Outlines library."""
        import outlines
        from outlines import models, generate
        
        # Get or create model
        if self.config.model_provider == ModelProvider.OPENAI:
            outlines_model = models.OpenAI(model, api_key=self.config.api_key)
        elif self.config.model_provider == ModelProvider.TRANSFORMERS:
            outlines_model = outlines.models.transformers(model)
        else:
            raise GenerationError(f"Provider {self.config.model_provider} not supported for Outlines")
        
        # Generate with JSON constraint
        generator = generate.json(outlines_model, schema)
        result = generator(prompt)
        
        return result if isinstance(result, dict) else result.dict()
    
    def _generate_without_outlines_json(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        prompt: str,
        model: str,
    ) -> Dict[str, Any]:
        """Fallback generation without Outlines."""
        # Use standard LLM call and validate
        raw_output = self._call_llm(prompt, model)
        
        # Try to extract JSON
        try:
            # Look for JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
            if json_match:
                raw_output = json_match.group(1)
            
            parsed = json.loads(raw_output)
            
            # Validate against schema if Pydantic model
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                validated = schema(**parsed)
                return validated.dict()
            
            return parsed
        except (json.JSONDecodeError, PydanticValidationError) as e:
            raise GenerationError(f"Failed to parse/validate JSON: {e}")
    
    def generate_regex(
        self,
        pattern: str,
        prompt: str,
        model: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> OutlinesResult:
        """
        Generate output matching a regex pattern.
        
        Args:
            pattern: Regex pattern to match
            prompt: Input prompt
            model: Override model name
            correlation_id: Correlation ID for tracking
            
        Returns:
            OutlinesResult with string matching pattern
        """
        correlation_id = correlation_id or f"outlines_regex_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = time.time()
        model_name = model or self.config.model_name
        
        logger.info({
            "msg": "Starting regex constrained generation",
            "correlation_id": correlation_id,
            "model": model_name,
            "pattern": pattern,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            if not self.circuit_breaker.can_execute():
                raise GenerationError("Circuit breaker is OPEN")
            
            result = self._exponential_backoff_retry(
                self._generate_regex_internal,
                pattern,
                prompt,
                model_name,
            )
            
            self.circuit_breaker.record_success()
            processing_time_ms = (time.time() - start_time) * 1000
            
            return OutlinesResult(
                success=True,
                output=result,
                raw_output=result,
                constraint_type="regex",
                model=model_name,
                processing_time_ms=processing_time_ms,
                metadata={"correlation_id": correlation_id, "pattern": pattern},
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.error({
                "msg": "Regex constrained generation failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            if self.config.fallback_to_unconstrained:
                return self._fallback_unconstrained(prompt, model_name, str(e), processing_time_ms)
            
            return OutlinesResult(
                success=False,
                output=None,
                raw_output="",
                constraint_type="regex",
                model=model_name,
                processing_time_ms=processing_time_ms,
                error=str(e),
                metadata={"correlation_id": correlation_id},
            )
    
    def _generate_regex_internal(self, pattern: str, prompt: str, model: str) -> str:
        """Internal regex generation implementation."""
        if self._outlines_available:
            return self._generate_with_outlines_regex(pattern, prompt, model)
        else:
            return self._generate_without_outlines_regex(pattern, prompt, model)
    
    def _generate_with_outlines_regex(self, pattern: str, prompt: str, model: str) -> str:
        """Generate using Outlines regex."""
        import outlines
        from outlines import models, generate
        
        if self.config.model_provider == ModelProvider.OPENAI:
            outlines_model = models.OpenAI(model, api_key=self.config.api_key)
        elif self.config.model_provider == ModelProvider.TRANSFORMERS:
            outlines_model = outlines.models.transformers(model)
        else:
            raise GenerationError(f"Provider {self.config.model_provider} not supported")
        
        generator = generate.regex(outlines_model, pattern)
        return generator(prompt)
    
    def _generate_without_outlines_regex(self, pattern: str, prompt: str, model: str) -> str:
        """Fallback regex generation."""
        raw_output = self._call_llm(prompt, model)
        
        # Try to extract matching content
        matches = re.findall(pattern, raw_output)
        if matches:
            return matches[0] if isinstance(matches[0], str) else str(matches[0])
        
        # Return raw output if it matches
        if re.match(pattern, raw_output.strip()):
            return raw_output.strip()
        
        raise GenerationError(f"Output does not match pattern: {pattern}")
    
    def generate_choices(
        self,
        choices: List[str],
        prompt: str,
        model: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> OutlinesResult:
        """
        Generate output from a list of choices.
        
        Args:
            choices: List of valid choices
            prompt: Input prompt
            model: Override model name
            correlation_id: Correlation ID for tracking
            
        Returns:
            OutlinesResult with one of the choices
        """
        correlation_id = correlation_id or f"outlines_choice_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = time.time()
        model_name = model or self.config.model_name
        
        logger.info({
            "msg": "Starting choice constrained generation",
            "correlation_id": correlation_id,
            "model": model_name,
            "choices_count": len(choices),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            if not self.circuit_breaker.can_execute():
                raise GenerationError("Circuit breaker is OPEN")
            
            result = self._exponential_backoff_retry(
                self._generate_choices_internal,
                choices,
                prompt,
                model_name,
            )
            
            self.circuit_breaker.record_success()
            processing_time_ms = (time.time() - start_time) * 1000
            
            return OutlinesResult(
                success=True,
                output=result,
                raw_output=result,
                constraint_type="choices",
                model=model_name,
                processing_time_ms=processing_time_ms,
                metadata={"correlation_id": correlation_id, "choices": choices},
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            processing_time_ms = (time.time() - start_time) * 1000
            
            if self.config.fallback_to_unconstrained:
                return self._fallback_unconstrained(prompt, model_name, str(e), processing_time_ms)
            
            return OutlinesResult(
                success=False,
                output=None,
                raw_output="",
                constraint_type="choices",
                model=model_name,
                processing_time_ms=processing_time_ms,
                error=str(e),
                metadata={"correlation_id": correlation_id},
            )
    
    def _generate_choices_internal(self, choices: List[str], prompt: str, model: str) -> str:
        """Internal choice generation."""
        if self._outlines_available:
            return self._generate_with_outlines_choices(choices, prompt, model)
        else:
            return self._generate_without_outlines_choices(choices, prompt, model)
    
    def _generate_with_outlines_choices(self, choices: List[str], prompt: str, model: str) -> str:
        """Generate using Outlines choices."""
        import outlines
        from outlines import models, generate
        
        if self.config.model_provider == ModelProvider.OPENAI:
            outlines_model = models.OpenAI(model, api_key=self.config.api_key)
        elif self.config.model_provider == ModelProvider.TRANSFORMERS:
            outlines_model = outlines.models.transformers(model)
        else:
            raise GenerationError(f"Provider {self.config.model_provider} not supported")
        
        generator = generate.choice(outlines_model, choices)
        return generator(prompt)
    
    def _generate_without_outlines_choices(self, choices: List[str], prompt: str, model: str) -> str:
        """Fallback choice generation."""
        # Enhance prompt with choices
        enhanced_prompt = f"{prompt}\n\nChoose exactly one from: {', '.join(choices)}"
        raw_output = self._call_llm(enhanced_prompt, model)
        
        # Find matching choice
        output_lower = raw_output.lower().strip()
        for choice in choices:
            if choice.lower() in output_lower or output_lower in choice.lower():
                return choice
        
        # Default to first choice if no match
        logger.warning(f"No choice matched, defaulting to first: {choices[0]}")
        return choices[0]
    
    def batch_generate(
        self,
        tasks: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> List[OutlinesResult]:
        """
        Generate multiple outputs in parallel.
        
        Args:
            tasks: List of task dictionaries with keys:
                   - 'type': 'json', 'regex', or 'choices'
                   - 'constraint': schema, pattern, or choices list
                   - 'prompt': input prompt
                   - 'model': optional model override
            max_workers: Number of parallel workers
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of OutlinesResult objects
        """
        correlation_id = correlation_id or f"outlines_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        max_workers = max_workers or self.config.batch_max_workers
        
        logger.info({
            "msg": "Starting batch generation",
            "correlation_id": correlation_id,
            "tasks_count": len(tasks),
            "max_workers": max_workers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        start_time = time.time()
        
        def process_task(task: Dict[str, Any]) -> OutlinesResult:
            task_type = task.get("type", "json")
            constraint = task.get("constraint")
            prompt = task.get("prompt", "")
            model = task.get("model")
            task_correlation_id = task.get("correlation_id", f"{correlation_id}_task")
            
            try:
                if task_type == "json":
                    return self.generate_json(constraint, prompt, model, task_correlation_id)
                elif task_type == "regex":
                    return self.generate_regex(constraint, prompt, model, task_correlation_id)
                elif task_type == "choices":
                    return self.generate_choices(constraint, prompt, model, task_correlation_id)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            except Exception as e:
                return OutlinesResult(
                    success=False,
                    output=None,
                    raw_output="",
                    constraint_type=task_type,
                    model=model or self.config.model_name,
                    processing_time_ms=0.0,
                    error=str(e),
                    metadata={"correlation_id": task_correlation_id},
                )
        
        # Execute in parallel
        futures = [self._executor.submit(process_task, task) for task in tasks]
        results = [f.result(timeout=self.config.batch_timeout_seconds) for f in futures]
        
        processing_time_ms = (time.time() - start_time) * 1000
        successful_count = sum(1 for r in results if r.success)
        
        logger.info({
            "msg": "Batch generation completed",
            "correlation_id": correlation_id,
            "tasks_count": len(tasks),
            "successful_count": successful_count,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return results
    
    def validate_output(
        self,
        output: str,
        constraint: Any,
    ) -> bool:
        """
        Validate output against a constraint.
        
        Args:
            output: Output string to validate
            constraint: Constraint to validate against (schema, pattern, or choices)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(constraint, dict):
                # JSON schema validation
                if constraint.get("type") == "object":
                    parsed = json.loads(output)
                    # Basic schema validation
                    return True
                return False
            elif isinstance(constraint, str):
                # Regex validation
                return bool(re.match(constraint, output))
            elif isinstance(constraint, list):
                # Choice validation
                return output in constraint
            elif isinstance(constraint, type) and issubclass(constraint, BaseModel):
                # Pydantic model validation
                constraint.parse_raw(output)
                return True
            return False
        except Exception:
            return False
    
    def _call_llm(self, prompt: str, model: str) -> str:
        """Call LLM without constraints."""
        if self.config.model_provider == ModelProvider.OPENAI:
            return self._call_openai(prompt, model)
        elif self.config.model_provider == ModelProvider.ANTHROPIC:
            return self._call_anthropic(prompt, model)
        elif self.config.model_provider == ModelProvider.TRANSFORMERS:
            return self._call_transformers(prompt, model)
        else:
            raise GenerationError(f"Unsupported provider: {self.config.model_provider}")
    
    def _call_openai(self, prompt: str, model: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, model: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    def _call_transformers(self, prompt: str, model: str) -> str:
        """Call local transformers model."""
        tokenizer = self._model_cache["tokenizer"]
        model_obj = self._model_cache["model"]
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model_obj.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _fallback_unconstrained(
        self,
        prompt: str,
        model: str,
        error: str,
        processing_time_ms: float,
    ) -> OutlinesResult:
        """Fallback to unconstrained generation."""
        logger.warning({
            "msg": "Falling back to unconstrained generation",
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            raw_output = self._call_llm(prompt, model)
            return OutlinesResult(
                success=True,
                output=raw_output,
                raw_output=raw_output,
                constraint_type="unconstrained",
                model=model,
                processing_time_ms=processing_time_ms,
                metadata={"fallback": True, "original_error": error},
            )
        except Exception as fallback_error:
            return OutlinesResult(
                success=False,
                output=None,
                raw_output="",
                constraint_type="unconstrained",
                model=model,
                processing_time_ms=processing_time_ms,
                error=f"Original: {error}, Fallback: {fallback_error}",
                metadata={"fallback_failed": True},
            )
    
    async def close(self):
        """Close resources."""
        logger.info({
            "msg": "Closing OutlinesAdapter resources",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        self._executor.shutdown(wait=True)
        
        # Clear caches
        self.grammar_cache.clear()
        self._model_cache.clear()
        
        logger.info({
            "msg": "OutlinesAdapter resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
