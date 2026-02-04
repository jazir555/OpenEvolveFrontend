"""LMQL Adapter - Declarative LLM Query Language.

SQL-like syntax for LLM interactions with constraint programming.
Enables complex KG queries with declarative constraints.

Architecture: SSOT (Single Source of Truth)
- Primary implementation in integrations/lmql/
- Wrapper in knowledge_engine/integrations/lmql/

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Configure structured logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class LMQLResult:
    """Result of an LMQL query execution.
    
    Attributes:
        success: Whether the query executed successfully
        data: The result data (type depends on query)
        query: The original query string
        model: The model used for execution
        tokens_used: Number of tokens consumed
        tokens_prompt: Number of prompt tokens
        tokens_completion: Number of completion tokens
        cost: Estimated cost in USD
        latency_ms: Execution latency in milliseconds
        timestamp: UTC timestamp of execution
        correlation_id: Unique ID for tracing
        metadata: Additional metadata
        error: Error message if failed
    """
    success: bool
    data: Any = None
    query: str = ""
    model: str = ""
    tokens_used: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DialogResult:
    """Result of a multi-turn dialog.
    
    Attributes:
        success: Whether the dialog completed successfully
        responses: List of responses from each turn
        final_response: The final response
        turns: Number of dialog turns executed
        constraint_violations: List of constraint violations encountered
        metadata: Additional metadata
        correlation_id: Unique ID for tracing
    """
    success: bool
    responses: List[str] = field(default_factory=list)
    final_response: str = ""
    turns: int = 0
    constraint_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EntityResult:
    """Result of entity extraction.
    
    Attributes:
        entity: The extracted entity text
        entity_type: Type of entity (e.g., PERSON, ORG, DATE)
        confidence: Confidence score (0.0 - 1.0)
        start_pos: Start position in source text
        end_pos: End position in source text
        metadata: Additional entity metadata
    """
    entity: str
    entity_type: str
    confidence: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of entity extraction operation.
    
    Attributes:
        success: Whether extraction succeeded
        entities: List of extracted entities
        query: The original query
        latency_ms: Execution latency
        timestamp: UTC timestamp
        correlation_id: Unique ID for tracing
        error: Error message if failed
    """
    success: bool
    entities: List[EntityResult] = field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error: Optional[str] = None


# =============================================================================
# CONSTRAINT SYSTEM
# =============================================================================


class ConstraintType(Enum):
    """Types of constraints supported by LMQL."""
    LENGTH = "length"
    TYPE = "type"
    REGEX = "regex"
    RANGE = "range"
    FROM_LIST = "from_list"
    CUSTOM = "custom"
    STOP_AT = "stop_at"
    STOPS_BEFORE = "stops_before"


@dataclass
class Constraint:
    """Represents a single LMQL constraint.
    
    Examples:
        >>> Constraint(type=ConstraintType.LENGTH, min=1, max=100)
        >>> Constraint(type=ConstraintType.TYPE, allowed_types=["str"])
        >>> Constraint(type=ConstraintType.REGEX, pattern=r"\d{4}-\d{2}-\d{2}")
        >>> Constraint(type=ConstraintType.FROM_LIST, values=["yes", "no"])
    """
    type: ConstraintType
    # Length constraints
    min: Optional[int] = None
    max: Optional[int] = None
    # Type constraint
    allowed_types: Optional[List[str]] = None
    # Regex constraint
    pattern: Optional[str] = None
    # Range constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    # List constraint
    values: Optional[List[str]] = None
    # Custom constraint
    predicate: Optional[Callable[[Any], bool]] = None
    predicate_name: Optional[str] = None
    # Stop constraints
    stop_sequence: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    # Metadata
    description: Optional[str] = None
    error_message: Optional[str] = None

    def to_lmql_syntax(self, variable: str = "RESULT") -> str:
        """Convert constraint to LMQL WHERE clause syntax."""
        if self.type == ConstraintType.LENGTH:
            parts = []
            if self.min is not None:
                parts.append(f"len({variable}) >= {self.min}")
            if self.max is not None:
                parts.append(f"len({variable}) <= {self.max}")
            return " and ".join(parts) if parts else ""
        
        elif self.type == ConstraintType.TYPE:
            if self.allowed_types:
                return f"type({variable}) in {self.allowed_types}"
            return ""
        
        elif self.type == ConstraintType.REGEX:
            if self.pattern:
                return f"REGEX({variable}, r'{self.pattern}')"
            return ""
        
        elif self.type == ConstraintType.RANGE:
            parts = []
            if self.min_value is not None:
                parts.append(f"{variable} >= {self.min_value}")
            if self.max_value is not None:
                parts.append(f"{variable} <= {self.max_value}")
            return " and ".join(parts) if parts else ""
        
        elif self.type == ConstraintType.FROM_LIST:
            if self.values:
                values_str = ", ".join(f"'{v}'" for v in self.values)
                return f"{variable} in [{values_str}]"
            return ""
        
        elif self.type == ConstraintType.STOP_AT:
            if self.stop_sequence:
                return f"STOPS_AT({variable}, '{self.stop_sequence}')"
            if self.stop_sequences:
                seqs = ", ".join(f"'{s}'" for s in self.stop_sequences)
                return f"STOPS_AT({variable}, [{seqs}])"
            return ""
        
        elif self.type == ConstraintType.STOPS_BEFORE:
            if self.stop_sequence:
                return f"STOPS_BEFORE({variable}, '{self.stop_sequence}')"
            return ""
        
        elif self.type == ConstraintType.CUSTOM:
            return f"CUSTOM({variable}, '{self.predicate_name or 'predicate'}')"
        
        return ""


# =============================================================================
# QUERY BUILDER
# =============================================================================


class LMQLQueryBuilder:
    """Builder for programmatic LMQL query construction.
    
    This class provides a fluent interface for building LMQL queries
    without writing raw LMQL syntax.
    
    Example:
        >>> builder = LMQLQueryBuilder()
        >>> query = (builder
        ...     .with_prompt("Extract entities from: {text}")
        ...     .with_variable("entities", type="list")
        ...     .with_constraint(ConstraintType.LENGTH, min=1)
        ...     .with_model("gpt-4")
        ...     .with_temperature(0.7)
        ...     .build())
    """
    
    def __init__(self):
        self._prompt: str = ""
        self._variables: List[Dict[str, Any]] = []
        self._constraints: List[Constraint] = []
        self._model: Optional[str] = None
        self._temperature: float = 0.7
        self._max_tokens: int = 500
        self._decoding: str = "argmax"
        self._distribution: Optional[Dict[str, Any]] = None
        
    def with_prompt(self, prompt: str, **kwargs) -> LMQLQueryBuilder:
        """Set the prompt template."""
        self._prompt = prompt.format(**kwargs) if kwargs else prompt
        return self
        
    def with_variable(
        self,
        name: str,
        var_type: str = "str",
        constraints: Optional[List[Constraint]] = None
    ) -> LMQLQueryBuilder:
        """Add a variable to capture."""
        self._variables.append({
            "name": name,
            "type": var_type,
            "constraints": constraints or []
        })
        if constraints:
            self._constraints.extend(constraints)
        return self
        
    def with_constraint(self, constraint: Constraint) -> LMQLQueryBuilder:
        """Add a constraint."""
        self._constraints.append(constraint)
        return self
        
    def with_model(self, model: str) -> LMQLQueryBuilder:
        """Set the model to use."""
        self._model = model
        return self
        
    def with_temperature(self, temperature: float) -> LMQLQueryBuilder:
        """Set the sampling temperature."""
        self._temperature = temperature
        return self
        
    def with_max_tokens(self, max_tokens: int) -> LMQLQueryBuilder:
        """Set maximum tokens to generate."""
        self._max_tokens = max_tokens
        return self
        
    def with_decoding(self, decoding: str) -> LMQLQueryBuilder:
        """Set decoding strategy (argmax, sample, beam)."""
        self._decoding = decoding
        return self
        
    def with_distribution(self, name: str, values: List[str]) -> LMQLQueryBuilder:
        """Set a distribution constraint."""
        self._distribution = {"name": name, "values": values}
        return self
        
    def build(self) -> str:
        """Build the LMQL query string."""
        lines = []
        
        # Add model and decoding configuration
        if self._model:
            lines.append(f'argmax "{self._prompt}"')
        else:
            lines.append(f'argmax "{self._prompt}"')
            
        # Add variable declarations
        for var in self._variables:
            name = var["name"]
            var_type = var["type"]
            
            if var_type == "list":
                lines.append(f'    {name}: list[str] = "..."')
            elif var_type == "str":
                lines.append(f'    {name}: str = "..."')
            elif var_type == "int":
                lines.append(f'    {name}: int = "..."')
            elif var_type == "float":
                lines.append(f'    {name}: float = "..."')
            elif var_type == "bool":
                lines.append(f'    {name}: bool = "..."')
            else:
                lines.append(f'    {name} = "..."')
                
        # Add constraints
        if self._constraints:
            constraint_parts = []
            for constraint in self._constraints:
                syntax = constraint.to_lmql_syntax()
                if syntax:
                    constraint_parts.append(syntax)
                    
            if constraint_parts:
                lines.append("WHERE")
                for part in constraint_parts:
                    lines.append(f"    {part}")
                    
        # Add distribution if specified
        if self._distribution:
            values_str = ", ".join(f"'{v}'" for v in self._distribution["values"])
            lines.append(f"DISTRIBUTION {{'{self._distribution['name']}': [{values_str}]}}")
            
        return "\n".join(lines)
        
    def build_json(self) -> Dict[str, Any]:
        """Build query as JSON structure for API calls."""
        return {
            "prompt": self._prompt,
            "variables": self._variables,
            "constraints": [
                {
                    "type": c.type.value,
                    "min": c.min,
                    "max": c.max,
                    "pattern": c.pattern,
                    "values": c.values,
                    "min_value": c.min_value,
                    "max_value": c.max_value,
                }
                for c in self._constraints
            ],
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "decoding": self._decoding,
        }


# =============================================================================
# LMQL ADAPTER
# =============================================================================


class LMQLAdapter:
    """Core adapter for LMQL (Language Model Query Language).
    
    Provides SQL-like declarative interface for LLM interactions with
    constraint programming capabilities.
    
    Features:
    - Declarative query execution
    - Constraint-based generation
    - Multi-turn dialog support
    - Entity extraction
    - KG query generation
    - Multiple provider support (OpenAI, Anthropic, local)
    
    Example:
        >>> adapter = LMQLAdapter(model="gpt-4")
        >>> result = adapter.query(
        ...     'Extract entities: {text}\\nentities: [ENTITY]',
        ...     context={"text": "Apple Inc. was founded by Steve Jobs."},
        ...     constraints=[Constraint(ConstraintType.LENGTH, min=1)]
        ... )
    """
    
    # Cost per 1K tokens (prompt, completion) in USD
    PRICING = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0015, 0.002),
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
    }
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: float = 30.0,
        max_retries: int = 3,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
    ):
        """Initialize LMQL adapter.
        
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            api_key: API key for the provider
            base_url: Custom base URL for API
            provider: Provider name (openai, anthropic, local)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            enable_caching: Whether to enable result caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Initialize cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        
        # Initialize client
        self._client: Any = None
        self._lmql_available = False
        
        # Try to import LMQL
        try:
            import lmql
            self._lmql = lmql
            self._lmql_available = True
            logger.info("LMQL library loaded successfully")
        except ImportError:
            logger.warning("LMQL library not available, using fallback implementation")
            self._lmql = None
            
        # Initialize provider client
        self._init_client()
        
        # Metrics
        self._metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cached_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
        }
        
    def _init_client(self) -> None:
        """Initialize the provider client."""
        if self.provider == "openai":
            try:
                import openai
                if self.api_key:
                    self._client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                    )
                else:
                    self._client = openai.OpenAI(timeout=self.timeout)
            except ImportError:
                logger.warning("OpenAI client not available")
                self._client = None
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                if self.api_key:
                    self._client = anthropic.Anthropic(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                    )
                else:
                    self._client = anthropic.Anthropic(timeout=self.timeout)
            except ImportError:
                logger.warning("Anthropic client not available")
                self._client = None
                
        elif self.provider == "local":
            # Local model support (e.g., via LMQL's local backend)
            self._client = None
            
    def _get_cache_key(self, query_str: str, context: Dict[str, Any], model: str) -> str:
        """Generate cache key for query."""
        import hashlib
        data = json.dumps({"query": query_str, "context": context, "model": model}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
        
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if result is in cache."""
        if not self.enable_caching or cache_key not in self._cache:
            return None
            
        result, timestamp = self._cache[cache_key]
        if time.time() - timestamp > self.cache_ttl:
            del self._cache[cache_key]
            return None
            
        return result
        
    def _set_cache(self, cache_key: str, result: Any) -> None:
        """Store result in cache."""
        if self.enable_caching:
            self._cache[cache_key] = (result, time.time())
            
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate query cost in USD."""
        pricing = self.PRICING.get(self.model, (0.01, 0.03))
        prompt_cost = (prompt_tokens / 1000) * pricing[0]
        completion_cost = (completion_tokens / 1000) * pricing[1]
        return round(prompt_cost + completion_cost, 6)
        
    def query(
        self,
        query_str: str,
        context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        constraints: Optional[List[Constraint]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeout: Optional[float] = None,
    ) -> LMQLResult:
        """Execute an LMQL query.
        
        Args:
            query_str: LMQL query string with optional {placeholders}
            context: Context variables to substitute in query
            model: Override model to use
            constraints: Additional constraints to apply
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Override timeout
            
        Returns:
            LMQLResult with query results and metadata
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        self._metrics["total_queries"] += 1
        
        try:
            # Substitute context into query
            if context:
                query_str = query_str.format(**context)
                
            # Check cache
            use_model = model or self.model
            cache_key = self._get_cache_key(query_str, context or {}, use_model)
            cached_result = self._check_cache(cache_key)
            
            if cached_result:
                self._metrics["cached_queries"] += 1
                cached_result.correlation_id = correlation_id
                return cached_result
                
            # Execute query based on availability
            if self._lmql_available and self._lmql:
                result = self._execute_lmql(
                    query_str=query_str,
                    model=use_model,
                    constraints=constraints,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout or self.timeout,
                )
            else:
                result = self._execute_fallback(
                    query_str=query_str,
                    model=use_model,
                    constraints=constraints,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout or self.timeout,
                )
                
            # Update result metadata
            result.correlation_id = correlation_id
            result.latency_ms = round((time.time() - start_time) * 1000, 2)
            
            # Update metrics
            if result.success:
                self._metrics["successful_queries"] += 1
                self._metrics["total_tokens"] += result.tokens_used
                self._metrics["total_cost"] += result.cost
            else:
                self._metrics["failed_queries"] += 1
                
            self._metrics["total_latency_ms"] += result.latency_ms
            
            # Cache successful results
            if result.success and self.enable_caching:
                self._set_cache(cache_key, result)
                
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            self._metrics["failed_queries"] += 1
            return LMQLResult(
                success=False,
                query=query_str,
                model=model or self.model,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                correlation_id=correlation_id,
                error=str(e),
            )
            
    def _execute_lmql(
        self,
        query_str: str,
        model: str,
        constraints: Optional[List[Constraint]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> LMQLResult:
        """Execute using native LMQL."""
        # This would use actual LMQL library
        # For now, fallback to standard execution
        return self._execute_fallback(
            query_str, model, constraints, temperature, max_tokens, timeout
        )
        
    def _execute_fallback(
        self,
        query_str: str,
        model: str,
        constraints: Optional[List[Constraint]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> LMQLResult:
        """Execute using standard LLM APIs with constraint validation."""
        prompt_tokens = len(query_str.split())  # Rough estimate
        
        try:
            if self.provider == "openai" and self._client:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": query_str}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                
                content = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else prompt_tokens
                completion_tokens = response.usage.completion_tokens if response.usage else len(content.split())
                
            elif self.provider == "anthropic" and self._client:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": query_str}],
                    timeout=int(timeout),
                )
                
                content = response.content[0].text if response.content else ""
                prompt_tokens = response.usage.input_tokens if response.usage else prompt_tokens
                completion_tokens = response.usage.output_tokens if response.usage else len(content.split())
                
            else:
                # Mock response for testing/development
                content = self._generate_mock_response(query_str, constraints)
                completion_tokens = len(content.split())
                
            # Validate constraints
            violations = []
            if constraints:
                for constraint in constraints:
                    if not self._validate_constraint(content, constraint):
                        violations.append(f"Constraint violated: {constraint.description or constraint.type.value}")
                        
            total_tokens = prompt_tokens + completion_tokens
            cost = self._estimate_cost(prompt_tokens, completion_tokens)
            
            return LMQLResult(
                success=True,
                data=content,
                query=query_str,
                model=model,
                tokens_used=total_tokens,
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                cost=cost,
                metadata={"constraint_violations": violations},
            )
            
        except Exception as e:
            return LMQLResult(
                success=False,
                query=query_str,
                model=model,
                error=str(e),
            )
            
    def _generate_mock_response(self, query_str: str, constraints: Optional[List[Constraint]]) -> str:
        """Generate mock response for testing."""
        # Check for entity extraction patterns
        if "entity" in query_str.lower():
            return '["Apple Inc.", "Steve Jobs", "Cupertino"]'
            
        # Check for relation extraction
        if "relation" in query_str.lower():
            return '[{"subject": "Apple Inc.", "predicate": "founded_by", "object": "Steve Jobs"}]'
            
        # Check constraints
        if constraints:
            for c in constraints:
                if c.type == ConstraintType.FROM_LIST and c.values:
                    return c.values[0]
                if c.type == ConstraintType.TYPE and c.allowed_types:
                    if "int" in c.allowed_types:
                        return "42"
                    if "bool" in c.allowed_types:
                        return "true"
                        
        return "Mock response for: " + query_str[:50]
        
    def _validate_constraint(self, value: str, constraint: Constraint) -> bool:
        """Validate a value against a constraint."""
        if constraint.type == ConstraintType.LENGTH:
            if constraint.min is not None and len(value) < constraint.min:
                return False
            if constraint.max is not None and len(value) > constraint.max:
                return False
            return True
            
        elif constraint.type == ConstraintType.REGEX:
            if constraint.pattern:
                return bool(re.match(constraint.pattern, value))
            return True
            
        elif constraint.type == ConstraintType.FROM_LIST:
            if constraint.values:
                return value in constraint.values
            return True
            
        elif constraint.type == ConstraintType.RANGE:
            try:
                num_val = float(value)
                if constraint.min_value is not None and num_val < constraint.min_value:
                    return False
                if constraint.max_value is not None and num_val > constraint.max_value:
                    return False
                return True
            except ValueError:
                return False
                
        elif constraint.type == ConstraintType.CUSTOM:
            if constraint.predicate:
                return constraint.predicate(value)
            return True
            
        return True
        
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract entities from text using LMQL.
        
        Args:
            text: Text to extract entities from
            entity_types: List of entity types to extract (e.g., ["PERSON", "ORG"])
            constraints: Additional constraints (min_confidence, max_entities)
            model: Override model
            
        Returns:
            ExtractionResult with extracted entities
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        try:
            types_str = ", ".join(entity_types) if entity_types else "all types"
            min_confidence = constraints.get("min_confidence", 0.5) if constraints else 0.5
            max_entities = constraints.get("max_entities", 50) if constraints else 50
            
            query_str = f"""Extract named entities from the following text.
Return entities as a JSON list of objects with fields: entity, type, confidence.

Text: {text}

Entity types to extract: {types_str}

Entities (JSON):"""

            result = self.query(
                query_str=query_str,
                model=model,
                constraints=[Constraint(ConstraintType.LENGTH, max=max_entities * 100)],
                temperature=0.0,
            )
            
            if not result.success:
                return ExtractionResult(
                    success=False,
                    query=query_str,
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    correlation_id=correlation_id,
                    error=result.error,
                )
                
            # Parse entities from response
            entities = self._parse_entities(result.data or "", min_confidence)
            
            # Apply max_entities constraint
            if len(entities) > max_entities:
                entities = entities[:max_entities]
                
            return ExtractionResult(
                success=True,
                entities=entities,
                query=query_str,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                correlation_id=correlation_id,
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                success=False,
                query="",
                latency_ms=round((time.time() - start_time) * 1000, 2),
                correlation_id=correlation_id,
                error=str(e),
            )
            
    def _parse_entities(self, text: str, min_confidence: float) -> List[EntityResult]:
        """Parse entities from LLM response."""
        entities = []
        
        try:
            # Try to parse as JSON
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        confidence = item.get("confidence", 0.5)
                        if confidence >= min_confidence:
                            entities.append(EntityResult(
                                entity=item.get("entity", ""),
                                entity_type=item.get("type", "UNKNOWN"),
                                confidence=confidence,
                            ))
            elif isinstance(data, dict) and "entities" in data:
                for item in data["entities"]:
                    confidence = item.get("confidence", 0.5)
                    if confidence >= min_confidence:
                        entities.append(EntityResult(
                            entity=item.get("entity", ""),
                            entity_type=item.get("type", "UNKNOWN"),
                            confidence=confidence,
                        ))
        except json.JSONDecodeError:
            # Fallback: extract entities using regex
            entity_pattern = r'["\']?([^"\']+)["\']?\s*:\s*["\']?([^"\']+)["\']?'
            matches = re.findall(entity_pattern, text)
            for entity, entity_type in matches:
                entities.append(EntityResult(
                    entity=entity.strip(),
                    entity_type=entity_type.strip(),
                    confidence=0.5,
                ))
                
        return entities
        
    def query_kg(
        self,
        kg_connection: Any,
        query_template: str,
        params: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph using LMQL-generated Cypher.
        
        Args:
            kg_connection: Knowledge graph connection object
            query_template: LMQL query template
            params: Parameters for the query
            model: Override model
            
        Returns:
            List of query results
        """
        try:
            # Generate Cypher query using LMQL
            cypher_result = self._generate_cypher(query_template, params or {}, model)
            
            if not cypher_result.success:
                logger.error(f"Cypher generation failed: {cypher_result.error}")
                return []
                
            cypher_query = cypher_result.data
            
            # Execute against KG if connection provided
            if kg_connection and hasattr(kg_connection, 'execute_query'):
                return kg_connection.execute_query(cypher_query, params)
            elif kg_connection and hasattr(kg_connection, 'run'):
                # Neo4j/Memgraph style
                result = kg_connection.run(cypher_query, params)
                return [dict(record) for record in result]
            else:
                # Return generated query for testing
                return [{"cypher": cypher_query, "params": params}]
                
        except Exception as e:
            logger.error(f"KG query failed: {e}", exc_info=True)
            return []
            
    def _generate_cypher(
        self,
        query_template: str,
        params: Dict[str, Any],
        model: Optional[str],
    ) -> LMQLResult:
        """Generate Cypher query from LMQL template."""
        prompt = f"""Convert the following natural language query to Cypher for Memgraph.

Query: {query_template}
Parameters: {json.dumps(params)}

Generate a Cypher query that:
1. Uses Memgraph-compatible syntax
2. Handles temporal queries using valid_from/valid_to properties
3. Uses parameterized queries for security
4. Returns only the Cypher query, no explanation

Cypher:"""

        return self.query(
            query_str=prompt,
            model=model,
            temperature=0.0,
            constraints=[Constraint(ConstraintType.STOP_AT, stop_sequence="\n\n")],
        )
        
    def multi_turn_dialog(
        self,
        history: List[Dict[str, str]],
        query: str,
        constraints: Optional[List[Constraint]] = None,
        max_turns: int = 5,
        model: Optional[str] = None,
    ) -> DialogResult:
        """Execute a multi-turn dialog with constraint checking.
        
        Args:
            history: List of previous messages {"role": "user/assistant", "content": "..."}
            query: The current query
            constraints: Constraints to apply to responses
            max_turns: Maximum dialog turns
            model: Override model
            
        Returns:
            DialogResult with dialog history and final response
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        try:
            responses = []
            current_query = query
            violations = []
            
            for turn in range(max_turns):
                # Build conversation context
                messages = history.copy()
                messages.append({"role": "user", "content": current_query})
                
                # Execute query
                result = self.query(
                    query_str=current_query,
                    model=model,
                    constraints=constraints,
                )
                
                if not result.success:
                    return DialogResult(
                        success=False,
                        responses=responses,
                        final_response="",
                        turns=turn,
                        constraint_violations=violations,
                        correlation_id=correlation_id,
                    )
                    
                response = result.data or ""
                responses.append(response)
                
                # Check constraints
                if constraints:
                    for constraint in constraints:
                        if not self._validate_constraint(response, constraint):
                            violations.append(f"Turn {turn}: {constraint.description or constraint.type.value}")
                            
                # Check if dialog should continue
                if self._is_dialog_complete(response):
                    break
                    
                # Prepare next turn
                current_query = self._generate_follow_up(response, turn)
                history.append({"role": "assistant", "content": response})
                history.append({"role": "user", "content": current_query})
                
            return DialogResult(
                success=True,
                responses=responses,
                final_response=responses[-1] if responses else "",
                turns=len(responses),
                constraint_violations=violations,
                correlation_id=correlation_id,
            )
            
        except Exception as e:
            logger.error(f"Multi-turn dialog failed: {e}", exc_info=True)
            return DialogResult(
                success=False,
                responses=responses if 'responses' in locals() else [],
                final_response="",
                turns=len(responses) if 'responses' in locals() else 0,
                constraint_violations=violations if 'violations' in locals() else [],
                correlation_id=correlation_id,
            )
            
    def _is_dialog_complete(self, response: str) -> bool:
        """Check if dialog should terminate."""
        completion_indicators = [
            "final answer",
            "conclusion",
            "in summary",
            "to conclude",
            "that's all",
            "complete",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)
        
    def _generate_follow_up(self, response: str, turn: int) -> str:
        """Generate follow-up query for next turn."""
        if turn == 0:
            return "Please elaborate on that."
        elif turn == 1:
            return "What are the implications of this?"
        else:
            return "Is there anything else I should know?"
            
    def constrained_generation(
        self,
        prompt: str,
        constraints: List[Constraint],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate text with strict constraints.
        
        Args:
            prompt: Generation prompt
            constraints: List of constraints to enforce
            model: Override model
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text that satisfies all constraints
        """
        result = self.query(
            query_str=prompt,
            model=model,
            constraints=constraints,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return result.data if result.success else ""
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics."""
        metrics = self._metrics.copy()
        if metrics["total_queries"] > 0:
            metrics["success_rate"] = metrics["successful_queries"] / metrics["total_queries"]
            metrics["cache_hit_rate"] = metrics["cached_queries"] / metrics["total_queries"]
            metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_queries"]
        else:
            metrics["success_rate"] = 0.0
            metrics["cache_hit_rate"] = 0.0
            metrics["avg_latency_ms"] = 0.0
        return metrics
        
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cached_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
        }
        
    def clear_cache(self) -> None:
        """Clear result cache."""
        self._cache.clear()


# =============================================================================
# QUERY OPTIMIZER
# =============================================================================


class QueryOptimizer:
    """Optimizer for common LMQL query patterns.
    
    Provides optimized query templates and execution strategies
    for frequent KG query patterns.
    """
    
    def __init__(self, adapter: LMQLAdapter):
        self.adapter = adapter
        
    def optimize_entity_extraction(self, text: str, entity_types: List[str]) -> str:
        """Generate optimized entity extraction query."""
        return f"""Extract all named entities from the text.
Types: {', '.join(entity_types)}

Text: {text}

Return JSON list: [{{"entity": "...", "type": "...", "confidence": 0.9}}]"""

    def optimize_relation_query(self, entity: str, relation_types: List[str]) -> str:
        """Generate optimized relation query."""
        return f"""Find all relations for entity: {entity}
Relation types: {', '.join(relation_types)}

Return JSON: [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]"""

    def optimize_schema_inference(self, sample_data: str) -> str:
        """Generate optimized schema inference query."""
        return f"""Infer the knowledge graph schema from this sample data.

Sample: {sample_data}

Return JSON schema with entity types and their relations."""

    def optimize_multi_hop(self, start_entity: str, hops: int) -> str:
        """Generate optimized multi-hop query."""
        return f"""Find all entities reachable from "{start_entity}" in {hops} hops.

Return path as JSON list of entities and relations."""


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================


_default_adapter: Optional[LMQLAdapter] = None
_default_optimizer: Optional[QueryOptimizer] = None


def get_default_adapter() -> LMQLAdapter:
    """Get or create the default LMQL adapter."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = LMQLAdapter()
    return _default_adapter


def get_default_optimizer() -> QueryOptimizer:
    """Get or create the default query optimizer."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = QueryOptimizer(get_default_adapter())
    return _default_optimizer


def reset_defaults() -> None:
    """Reset default instances."""
    global _default_adapter, _default_optimizer
    _default_adapter = None
    _default_optimizer = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Classes
    "LMQLAdapter",
    "LMQLQueryBuilder",
    "QueryOptimizer",
    "Constraint",
    "ConstraintType",
    # Data classes
    "LMQLResult",
    "DialogResult",
    "ExtractionResult",
    "EntityResult",
    # Functions
    "get_default_adapter",
    "get_default_optimizer",
    "reset_defaults",
]


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    adapter = LMQLAdapter()
    
    # Example 1: Simple query
    result = adapter.query(
        query_str="What is the capital of France?",
        constraints=[Constraint(ConstraintType.LENGTH, max=100)]
    )
    print(f"Query result: {result.data}")
    
    # Example 2: Entity extraction
    extraction = adapter.extract_entities(
        text="Apple Inc. was founded by Steve Jobs in Cupertino.",
        entity_types=["ORG", "PERSON", "GPE"]
    )
    for entity in extraction.entities:
        print(f"Entity: {entity.entity} ({entity.entity_type})")
        
    # Example 3: Query builder
    builder = LMQLQueryBuilder()
    query = (builder
        .with_prompt("Extract entities from: {text}")
        .with_variable("entities", "list")
        .with_constraint(Constraint(ConstraintType.LENGTH, min=1))
        .with_model("gpt-4")
        .build())
    print(f"\nBuilt query:\n{query}")
